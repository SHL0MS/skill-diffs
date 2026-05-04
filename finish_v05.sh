#!/bin/bash
# Run after embeddings + bundled refresh + baselines complete.
# Adds Tier 2.2 (semantic diff), Tier 3.1 (injection tag), Tier 3.3 (quality score),
# regenerates curator subsets, runs final stats.

set -e
cd "$(dirname "$0")"

LOG=v05_finish.log
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

echo "=========================================="
echo "v0.5 finish chain started: $(date)"
echo "=========================================="

# Wait for any running batch / pipeline first
stage "wait for tier 1 background processes"
while pgrep -f "extract_bundled.py" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] bundled still running"
done
while pgrep -f "embed_cluster.py" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] embed still running"
done
while pgrep -f "run_baselines_v05" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] baselines still running"
done
echo "tier 1 background work all done"

# After bundled JSONL is written for new platform repos, aggregate
stage "aggregate bundled.parquet"
run uv run python aggregate_bundled.py

# After embeddings done, merge semantic cluster IDs into release parquets
stage "add semantic clusters to release parquets"
if [ -f data/semantic_clusters.parquet ]; then
    run uv run python add_semantic_clusters.py
else
    echo "  (skip: data/semantic_clusters.parquet not found — embed_cluster.py may have failed)"
fi

# Tier 2.2: structural diff_summary column
stage "add semantic diff structure column"
run uv run python add_semantic_diff.py

# Tier 3.1: prompt_injection_pattern tag
stage "add prompt-injection tag"
run uv run python add_injection_tag.py

# Regenerate curator subsets with all new columns + tags
stage "regenerate curator_training (default + strict)"
run uv run python curator_subset.py
run uv run python curator_subset.py --strict

# Tier 3.3: quality_score (must come last — uses everything else)
stage "add aggregate quality_score column"
run uv run python add_quality_score.py

# Final stats
stage "final stats"
run uv run python -c "
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path
print()
print(f'{'\"'}file{'\"':<32} {'\"'}rows{'\"':>10} {'\"'}cols{'\"':>5} {'\"'}size{'\"':>10}')
print('-' * 60)
for p in sorted(Path('data/release').glob('*.parquet')):
    md = pq.read_metadata(p)
    size_mb = p.stat().st_size / 1_000_000
    print(f'  {p.name:<30} {md.num_rows:>10,} {md.num_columns:>5}  {size_mb:>7.1f} MB')
"

echo
echo "=========================================="
echo "v0.5 finish chain done: $(date)"
echo "=========================================="
echo "Next: review numbers, update READMEs, upload + commit"
