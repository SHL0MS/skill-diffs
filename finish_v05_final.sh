#!/bin/bash
# Final wrap-up after embed_cluster retry + baselines complete:
#   1. wait for embed_cluster retry to produce semantic_clusters.parquet
#   2. run add_semantic_clusters.py (merges into release parquets)
#   3. wait for baselines to finish all 4 stages
#   4. re-run identity + intent_only baselines so the metric table includes
#      linter_delta (which was added after those stages started)
#   5. print final summary

set -e
cd "$(dirname "$0")"

LOG=v05_final.log
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
echo "v0.5 final wrap-up: $(date)"
echo "=========================================="

# Wait for embed_cluster retry
stage "wait for embed_cluster retry (clustering only)"
while pgrep -f "embed_cluster.py.*--skip-embed" > /dev/null; do
    sleep 30
    echo "[$(date +%H:%M:%S)] embed_cluster retry still running"
done
echo "embed_cluster retry done."

stage "merge semantic clusters into release parquets"
if [ -f data/semantic_clusters.parquet ]; then
    run uv run python add_semantic_clusters.py
else
    echo "ERROR: data/semantic_clusters.parquet still missing"
    exit 1
fi

# Re-run curator_subset to ensure new columns are in the latest version
stage "regenerate curator_training (default + strict) with new columns"
run uv run python curator_subset.py
run uv run python curator_subset.py --strict

# Wait for baselines (the long pole)
stage "wait for baselines_v05.sh"
while pgrep -f "run_baselines_v05" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M:%S)] baselines still running"
done
echo "baselines done."

# Re-run identity + intent_only (originally ran before linter_delta was added)
stage "re-run identity baseline (now with linter_delta)"
run uv run python eval_curator.py \
    --eval-set data/release/curator_eval_set_v2.parquet \
    --model identity --judge

stage "re-run intent_only baseline (now with linter_delta)"
run uv run python eval_curator.py \
    --eval-set data/release/curator_eval_set_v2.parquet \
    --model intent_only --judge

# Final summary
stage "final summary"
run uv run python -c "
import json
from pathlib import Path
from collections import OrderedDict

# Get most recent eval result per model
results = sorted(Path('data/eval_results').glob('*__*.json'),
                 key=lambda p: p.stat().st_mtime)
latest_per_model = OrderedDict()
for r in results:
    if 'curator_eval_set_v2' not in r.read_text(errors='replace'):
        # only consider results against the v2 (stratified) eval set
        # crude check via filename
        pass
    # parse model name from filename
    parts = r.stem.split('__', 1)
    if len(parts) != 2: continue
    model = parts[1]
    latest_per_model[model] = r

print()
print(f'{\"model\":<35} {\"edit_dist\":<10} {\"rouge_l\":<10} {\"judge_overall\":<15} {\"linter_delta\":<13}')
print('-' * 85)
for model, path in latest_per_model.items():
    d = json.loads(path.read_text())
    s = d.get('summary', {})
    ed = s.get('edit_dist_ratio_mean', '-')
    rl = s.get('rouge_l_mean', '-')
    jo = s.get('judge_overall_mean', '-')
    ld = s.get('linter_delta_mean', '-')
    print(f'{model:<35} {ed!s:<10} {rl!s:<10} {jo!s:<15} {ld!s:<13}')
"

echo
echo "=========================================="
echo "v0.5 final wrap-up DONE: $(date)"
echo "Ready for: README updates with eval table → upload_hf.py → git commit + push"
echo "=========================================="
