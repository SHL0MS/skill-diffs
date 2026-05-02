#!/bin/bash
# Live monitor for v0.4 pipeline. Run from skill-diffs root.
# Shows orchestrator stage + active batch progress + process state, refreshing every 5s.
# Ctrl-C to exit.

cd "$(dirname "$0")"

while true; do
    clear
    echo "================================================================"
    echo " v0.4 pipeline monitor   $(date '+%a %H:%M:%S')"
    echo "================================================================"
    echo

    # --- orchestrator stage (prefer most recent log) ---
    for log in v04_finish.log v04_trimmed.log v04_pipeline.log; do
        if [ -f "$log" ]; then
            echo "── orchestrator (last 10 lines of $log) ─"
            tail -10 "$log"
            echo
            break
        fi
    done

    # --- active batch (most recently modified batch_*.log) ---
    latest=$(ls -t batch_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "── active batch [$latest] ─"
        tail -5 "$latest"
        echo
    fi

    # --- all batch summaries ---
    echo "── batch summaries ──────────────────────────────────"
    for f in batch_hermes.log batch_opencode.log; do
        if [ -f "$f" ]; then
            done_line=$(grep -E "^Done in" "$f" | tail -1)
            if [ -n "$done_line" ]; then
                printf "  %-25s DONE  %s\n" "$f" "$(echo "$done_line" | cut -c1-60)"
            else
                last=$(tail -1 "$f" 2>/dev/null)
                # Extract progress fragment if present
                frag=$(echo "$last" | grep -oE '\[[0-9,]+/[0-9,]+\].*eta=[0-9]+s' | head -1)
                if [ -n "$frag" ]; then
                    printf "  %-25s RUN   %s\n" "$f" "$frag"
                else
                    printf "  %-25s RUN   %s\n" "$f" "$(echo "$last" | cut -c1-60)"
                fi
            fi
        else
            printf "  %-25s pending\n" "$f"
        fi
    done
    echo

    # --- running processes ---
    echo "── running v0.4 processes ───────────────────────────"
    procs=$(ps -ef | grep -E "(run_v04_pipeline|batch_v04|consolidate_v04|enrich_v03|embed_cluster|pr_metadata|add_licenses|add_semantic|join_pr_meta)" | grep -v grep | grep -v monitor.sh)
    if [ -n "$procs" ]; then
        echo "$procs" | awk '{
            cmd=""
            for(i=8; i<=NF && i<=14; i++) cmd = cmd " " $i
            printf "  pid=%-7s %s\n", $2, cmd
        }'
    else
        echo "  (none — orchestrator finished or not started)"
    fi
    echo

    # --- raw output dir sizes ---
    echo "── raw output sizes ─────────────────────────────────"
    for d in data/raw_hermes_skill data/raw_opencode_skill; do
        if [ -d "$d" ]; then
            n=$(ls "$d" 2>/dev/null | wc -l | tr -d ' ')
            printf "  %-32s %6s files\n" "$d" "$n"
        fi
    done
    echo

    # --- gh rate limit (once per minute to avoid burning rate) ---
    sec=$(date +%S)
    if [ "$sec" -lt "06" ]; then
        rl=$(gh api rate_limit --jq '.resources.core | "\(.remaining)/\(.limit) reset_in=\((.reset - now) | floor)s"' 2>/dev/null)
        echo "── gh core rate: $rl"
        echo
    fi

    # --- full pipeline ETA (recovery + finish pipeline) ---
    echo "── full pipeline ETA (recovery + finish) ─────────────"
    python3 - <<'EOF'
import os, re, time
from pathlib import Path

# Batches done. Recovery + finish stages.
STAGES = [
    ("hermes_batch",     "batch_hermes.log"),
    ("opencode_batch",   "batch_opencode.log"),
    ("merge_v04",        None),           # already done synchronously
    ("pr_metadata",      "v04_finish.log"),
    ("join_pr",          "v04_finish.log"),
    ("add_licenses",     "v04_finish.log"),
    ("enrich_v03",       "v04_finish.log"),
    ("curator_subset",   "v04_finish.log"),
    ("skill_linter",     "v04_finish.log"),
]

# Estimated time per stage (from observed runs)
STAGE_SECONDS = {
    "pr_metadata": 8 * 60,
    "join_pr": 1 * 60,
    "add_licenses": 6 * 60,
    "enrich_v03": 18 * 60,    # MinHash on 577k skills ~15-20 min
    "curator_subset": 1 * 60,
    "skill_linter": 4 * 60,
}

# Parse v04_finish.log for stage transitions
log_path = Path("v04_finish.log")
done_stages = set()
current_stage = None

if log_path.exists():
    txt = log_path.read_text()
    # Find which "------ STAGE_NAME ------" markers are followed by "exit: 0"
    stage_blocks = re.findall(
        r"------ (.+?) ------ .+?\n(.*?)(?=------ |\Z)",
        txt, re.DOTALL,
    )
    for stage_name, body in stage_blocks:
        if "exit: 0" in body:
            done_stages.add(stage_name.strip())
        else:
            current_stage = stage_name.strip()

# Check if final stats stage is done (signals full completion)
finished = "exit: 0" in (log_path.read_text() if log_path.exists() else "") and \
           "Final stats" in done_stages

# Map stage name to estimate
STAGE_NAME_TO_KEY = {
    "PR metadata refresh (delta only)": "pr_metadata",
    "Join PR metadata": "join_pr",
    "License metadata refresh": "add_licenses",
    "Enrich v0.3 (MinHash + frontmatter) on combined corpus": "enrich_v03",
    "Curator training subset": "curator_subset",
    "Skill linter sanity report": "skill_linter",
}

total_remaining = 0
print(f"  hermes_batch    DONE  (567 repos, 65k records)")
print(f"  opencode_batch  DONE  (1285/1302 repos, 137k records, killed stragglers)")
print(f"  merge_v04       DONE  (864k records combined)")
for stage_label, key in STAGE_NAME_TO_KEY.items():
    if stage_label in done_stages:
        print(f"  {key:<15} DONE")
    elif stage_label == current_stage:
        # Try to extract progress from latest line
        latest_lines = txt.splitlines()[-10:] if log_path.exists() else []
        progress = ""
        for line in reversed(latest_lines):
            m = re.search(r'\[(\d[\d,]*)/(\d[\d,]*)\].*eta=(\d+)s', line)
            if m:
                progress = f" ({m.group(1)}/{m.group(2)} eta={m.group(3)}s)"
                total_remaining += int(m.group(3))
                break
        if not progress:
            total_remaining += STAGE_SECONDS.get(key, 5*60)
        print(f"  {key:<15} RUN  {progress}")
    else:
        secs = STAGE_SECONDS.get(key, 5*60)
        total_remaining += secs
        print(f"  {key:<15} queued (est {secs//60}m)")

def fmt(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m = s // 60
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"

print()
if finished:
    print(f"  PIPELINE COMPLETE \u2014 ready for README + upload")
else:
    print(f"  TOTAL REMAINING: {fmt(total_remaining)}")
    finish = time.localtime(time.time() + total_remaining)
    print(f"  ESTIMATED FINISH: {time.strftime('%a %H:%M', finish)}")
EOF
    echo

    # --- footer ---
    echo "(refreshing every 5s — Ctrl-C to exit)"
    sleep 5
done
