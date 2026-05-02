#!/bin/bash
# v0.4 TRIMMED pipeline — Curator-focused scope.
# Skips OpenClaw + Cursor batches and embedding/semantic clustering.
# Adds: curator_subset.py + skill_linter.py
#
# Resumable: each stage skips work already done.

set -e
cd "$(dirname "$0")"

LOG=v04_trimmed.log
exec >> "$LOG" 2>&1

echo
echo "=========================================="
echo "v0.4 TRIMMED pipeline started: $(date)"
echo "Scope: Hermes + OpenCode + existing Anthropic"
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

# Stage: wait for OpenCode batch to finish
stage "wait for OpenCode batch"
while pgrep -f "batch_v04.py.*--platform opencode_skill" > /dev/null; do
    sleep 30
    echo "[$(date +%H:%M:%S)] OpenCode running: $(tail -1 batch_opencode.log 2>/dev/null | head -c 200)"
done
echo "OpenCode done."

# Stage: Consolidate v0.4 (only data/raw, raw_hermes_skill, raw_opencode_skill)
stage "Consolidate v0.4"
run uv run python consolidate_v04.py

# Stage: PR metadata refresh (delta only — existing cache covers v0.3)
stage "PR metadata refresh"
run uv run python pr_metadata.py --workers 4

# Stage: Join PR metadata
stage "Join PR metadata"
run uv run python join_pr_metadata.py

# Stage: License metadata refresh
stage "License metadata refresh"
run uv run python add_licenses.py --workers 8

# Stage: Re-run enrich_v03 (MinHash + frontmatter validation)
stage "Enrich v0.3 features (MinHash + frontmatter)"
run uv run python enrich_v03.py

# Stage: Derive Curator training subset
stage "Curator training subset"
run uv run python curator_subset.py

# Stage: Run skill linter (sanity report)
stage "Skill linter sanity report"
run uv run python skill_linter.py --report

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
echo "v0.4 TRIMMED pipeline finished: $(date)"
echo "=========================================="
echo
echo "Next: review parquets, update READMEs, run upload_hf.py"
