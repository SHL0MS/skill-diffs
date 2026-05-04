#!/bin/bash
# Run all baseline evals against the new stratified eval set with LLM-as-judge.
#
# Estimated cost: ~$50 total (Sonnet judge dominates).
# Estimated time: ~3-4 hours wall (sequential).

set -e
cd "$(dirname "$0")"

LOG=baselines_v05.log
exec >> "$LOG" 2>&1

EVAL_SET=data/release/curator_eval_set_v2.parquet

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
echo "v0.5 baseline eval suite started: $(date)"
echo "=========================================="

stage "identity baseline (free, fast)"
run uv run python eval_curator.py --eval-set "$EVAL_SET" --model identity --judge

stage "intent_only baseline (free, fast)"
run uv run python eval_curator.py --eval-set "$EVAL_SET" --model intent_only --judge

stage "claude-haiku-4-5 (real model, ~\$1)"
run uv run python eval_curator.py --eval-set "$EVAL_SET" \
    --model anthropic:claude-haiku-4-5 --judge --output-predictions

stage "claude-sonnet-4-5 (smartest baseline, ~\$10)"
run uv run python eval_curator.py --eval-set "$EVAL_SET" \
    --model anthropic:claude-sonnet-4-5 --judge --output-predictions

stage "Final summary"
run uv run python -c "
import json
from pathlib import Path
results = sorted(Path('data/eval_results').glob('*.json'))
recent = [r for r in results if 'curator_eval_set_v2' in str(r) or '20260503' in str(r)]
print()
for r in recent[-4:]:
    d = json.loads(r.read_text())
    s = d['summary']
    print(f'  {s[\"model\"]:<35} judge_overall={s.get(\"judge_overall_mean\", \"-\")} edit_dist={s.get(\"edit_dist_ratio_mean\", \"-\")}')
"

echo
echo "=========================================="
echo "v0.5 baseline eval suite done: $(date)"
echo "=========================================="
