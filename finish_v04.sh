#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG=v04_finish.log
exec >> "$LOG" 2>&1

stage() {
    echo
    echo "------ $1 ------ $(date) ------"
}
run() {
    echo "+ $@"
    "$@"
    echo "exit: $?"
}

echo "=== v0.4 finish pipeline started: $(date) ==="

stage "PR metadata refresh (delta only)"
run uv run python pr_metadata.py --workers 4

stage "Join PR metadata"
run uv run python join_pr_metadata.py

stage "License metadata refresh"
run uv run python add_licenses.py --workers 8

stage "Enrich v0.3 (MinHash + frontmatter) on combined corpus"
run uv run python enrich_v03.py

stage "Curator training subset"
run uv run python curator_subset.py

stage "Skill linter sanity report"
run uv run python skill_linter.py --report

stage "Final stats"
run uv run python -c "
import pyarrow.parquet as pq
from pathlib import Path
for p in sorted(Path('data/release').glob('*.parquet')):
    md = pq.read_metadata(p)
    size_mb = p.stat().st_size / 1_000_000
    print(f'  {p.name:<32} rows={md.num_rows:>10,}  {size_mb:>8.1f} MB')
"

echo
echo "=== v0.4 finish pipeline done: $(date) ==="
