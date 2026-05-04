#!/bin/bash
# Optimized finish chain — runs aggregate_bundled immediately, waits for
# embeddings only when needed, doesn't block on baselines (they run
# independently and produce data/eval_results/ which is used at ship time).

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

echo
echo "=========================================="
echo "v0.5 finish chain (v2 — non-blocking) restarted: $(date)"
echo "=========================================="

# Aggregate bundled — already finished extraction
stage "aggregate bundled.parquet"
run uv run python aggregate_bundled.py

# Wait for embeddings to finish before merging clusters into release parquets
stage "wait for embed_cluster.py"
while pgrep -f "embed_cluster.py" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] embed still running"
done

stage "add semantic clusters to release parquets"
if [ -f data/semantic_clusters.parquet ]; then
    run uv run python add_semantic_clusters.py
else
    echo "  WARNING: data/semantic_clusters.parquet not found — embed_cluster.py may have failed"
fi

# These don't depend on baselines or embeddings — could even have run
# in parallel with aggregate_bundled, but sequencing keeps disk IO sane
stage "add semantic diff structure column"
run uv run python add_semantic_diff.py

stage "add prompt-injection tag"
run uv run python add_injection_tag.py

stage "regenerate curator_training (default + strict)"
run uv run python curator_subset.py
run uv run python curator_subset.py --strict

stage "add aggregate quality_score column"
run uv run python add_quality_score.py

# Final stats
stage "final stats"
run uv run python -c "
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path
print()
for p in sorted(Path('data/release').glob('*.parquet')):
    md = pq.read_metadata(p)
    size_mb = p.stat().st_size / 1_000_000
    print(f'  {p.name:<32} rows={md.num_rows:>10,} cols={md.num_columns:>3}  {size_mb:>8.1f} MB')
print()
print('Platform breakdown (clean tier):')
t = pq.read_table('data/release/diffs_clean.parquet', columns=['platform'])
plats = Counter(t['platform'].to_pylist())
for p, c in sorted(plats.items(), key=lambda x: -x[1]):
    print(f'  {p:<22} {c:>8,}  ({100*c/t.num_rows:.1f}%)')
"

echo
echo "=========================================="
echo "v0.5 finish chain done: $(date)"
echo "Baselines may still be running independently \u2014 check baselines_v05.log"
echo "=========================================="
