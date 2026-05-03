#!/bin/bash
# After batch_v04.py finishes scraping OpenClaw repos, this orchestrates the
# integration into existing release parquets and re-runs all enrichment.
#
# Idempotent: re-running is safe — pr_metadata uses caches, add_licenses
# and enrich_v03 are now drop-and-rewrite (no duplicate columns).

set -e
cd "$(dirname "$0")"

LOG=v041_openclaw.log
exec >> "$LOG" 2>&1

echo
echo "=========================================="
echo "v0.4.1 OpenClaw integration: $(date)"
echo "=========================================="

stage() {
    echo
    echo "------ $1 ------ $(date) ------"
}
run() {
    echo "+ $@"
    "$@"
    echo "exit: $?"
}

# Stage: wait for OpenClaw batch to finish
stage "wait for OpenClaw batch"
while pgrep -f "batch_v04.py.*--platform openclaw_skill" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] OpenClaw running: $(tail -1 batch_openclaw.log 2>/dev/null | head -c 200)"
done
echo "OpenClaw batch finished."

# Stage: integrate openclaw raw into existing release parquets
stage "Add OpenClaw platform to release parquets"
run uv run python add_platform.py --platform openclaw_skill --raw-dir data/raw_openclaw_skill

# Stage: PR metadata refresh (delta — existing cache covers most repos)
stage "PR metadata refresh"
run uv run python pr_metadata.py --workers 4

# Stage: Join PR metadata
stage "Join PR metadata"
run uv run python join_pr_metadata.py

# Stage: License metadata refresh (idempotent, fetches delta only)
stage "License metadata refresh"
run uv run python add_licenses.py --workers 8

# Stage: Re-run enrich_v03 (MinHash on combined corpus — idempotent now)
stage "Enrich v0.3 features (MinHash + frontmatter) on full corpus"
run uv run python enrich_v03.py

# Stage: Regenerate curator training subset
stage "Curator training subset"
run uv run python curator_subset.py

# Stage: Skill linter sanity report
stage "Skill linter sanity report (full corpus)"
run uv run python skill_linter.py --report

# Stage: Final stats
stage "Final stats"
run uv run python -c "
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path
print()
for p in sorted(Path('data/release').glob('*.parquet')):
    md = pq.read_metadata(p)
    size_mb = p.stat().st_size / 1_000_000
    print(f'  {p.name:<32} rows={md.num_rows:>10,}  {size_mb:>8.1f} MB')
print()
print('Platform breakdown (clean):')
t = pq.read_table('data/release/diffs_clean.parquet', columns=['platform'])
plats = Counter(t['platform'].to_pylist())
for p, c in sorted(plats.items(), key=lambda x: -x[1]):
    print(f'  {p:<22} {c:>8,}  ({100*c/t.num_rows:.1f}%)')
"

echo
echo "=========================================="
echo "v0.4.1 OpenClaw integration done: $(date)"
echo "=========================================="
echo "Next: review numbers, update READMEs, run upload_hf.py, git commit"
