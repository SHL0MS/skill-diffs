#!/bin/bash
# v0.4 pipeline orchestrator — runs the remaining batches and enrichment
# stages sequentially, with progress logging.
#
# Resumable: each stage skips work already done.
#
# Usage:
#   ./run_v04_pipeline.sh                  # run from current state
#   ./run_v04_pipeline.sh --skip-batches   # only run consolidation onwards

set -e
cd "$(dirname "$0")"

LOG=v04_pipeline.log
exec >> "$LOG" 2>&1

echo
echo "=========================================="
echo "v0.4 pipeline run started: $(date)"
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

# Stage: wait for any in-flight batch processes from launch
stage "wait for OpenCode batch"
while pgrep -f "batch_v04.py.*--platform opencode_skill" > /dev/null; do
    sleep 30
    echo "[$(date +%H:%M:%S)] OpenCode still running: $(tail -1 batch_opencode.log 2>/dev/null)"
done
echo "OpenCode done."

# Stage: OpenClaw batch
stage "OpenClaw batch"
if [ ! -f data/manifest_openclaw_skill.jsonl ] || \
   [ "$(grep -c '"status": "ok"' data/manifest_openclaw_skill.jsonl 2>/dev/null)" -lt 1700 ]; then
    run uv run python batch_v04.py --repos data/openclaw_repos.txt \
        --platform openclaw_skill --extractor skill_md --workers 16
else
    echo "  (already done)"
fi

# Stage: Cursor batch
stage "Cursor batch"
if [ ! -f data/manifest_cursor_rule.jsonl ] || \
   [ "$(grep -c '"status": "ok"' data/manifest_cursor_rule.jsonl 2>/dev/null)" -lt 1700 ]; then
    run uv run python batch_v04.py --repos data/cursor_repos.txt \
        --platform cursor_rule --extractor cursor --workers 16
else
    echo "  (already done)"
fi

# Stage: Consolidate v0.4
stage "Consolidate v0.4"
run uv run python consolidate_v04.py

# Stage: PR metadata for new repos (delta, existing cache reused)
stage "PR metadata (delta for new repos)"
# Build combined repo list
cat data/huzey_repos.txt data/expansion_repos.txt \
    data/opencode_repos.txt data/hermes_repos.txt \
    data/openclaw_repos.txt data/cursor_repos.txt 2>/dev/null \
    | sort -u > data/all_repos_v04.txt
echo "Total repos: $(wc -l < data/all_repos_v04.txt)"
# pr_metadata.py reads repos.parquet by default; we need to make it use the new repos.parquet
# which now has ALL platforms.
run uv run python pr_metadata.py --workers 4

# Stage: Join PR metadata
stage "Join PR metadata"
run uv run python join_pr_metadata.py

# Stage: License metadata for new repos
stage "License metadata"
run uv run python add_licenses.py --workers 8

# Stage: Re-run enrich_v03 (MinHash + frontmatter + same-author dedup)
stage "Enrich v0.3 features (MinHash + frontmatter)"
run uv run python enrich_v03.py

# Stage: Embed + semantic cluster
stage "Embed + semantic cluster"
run uv run python embed_cluster.py

# Stage: Add semantic clusters to release parquets
stage "Add semantic clusters to release parquets"
run uv run python add_semantic_clusters.py

# Stage: Final stats
stage "Final stats"
run uv run python -c "
import pyarrow.parquet as pq
from pathlib import Path
for p in sorted(Path('data/release').glob('*.parquet')):
    t = pq.read_metadata(p)
    size_mb = p.stat().st_size / 1_000_000
    print(f'  {p.name:<32} rows={t.num_rows:>10,}  {size_mb:>8.1f} MB')
"

echo
echo "=========================================="
echo "v0.4 pipeline run finished: $(date)"
echo "=========================================="
