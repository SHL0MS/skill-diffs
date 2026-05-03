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
    for log in v041_openclaw.log v04_finish.log v04_trimmed.log v04_pipeline.log; do
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
    for entry in "hermes:batch_hermes.log:data/raw_hermes_skill" \
                 "opencode:batch_opencode.log:data/raw_opencode_skill" \
                 "openclaw:batch_openclaw.log:data/raw_openclaw_skill"; do
        name=$(echo "$entry" | cut -d: -f1)
        f=$(echo "$entry" | cut -d: -f2)
        d=$(echo "$entry" | cut -d: -f3)
        if [ -f "$f" ]; then
            done_line=$(grep -E "^Done in" "$f" | tail -1)
            if [ -n "$done_line" ]; then
                printf "  %-12s DONE  %s\n" "$name" "$(echo "$done_line" | cut -c1-60)"
            else
                last=$(tail -1 "$f" 2>/dev/null)
                frag=$(echo "$last" | grep -oE '\[[0-9,]+/[0-9,]+\].*eta=[0-9]+s' | head -1)
                if [ -n "$frag" ]; then
                    printf "  %-12s RUN   %s\n" "$name" "$frag"
                else
                    printf "  %-12s RUN   %s\n" "$name" "$(echo "$last" | cut -c1-60)"
                fi
            fi
        elif [ -d "$d" ] && [ "$(ls "$d" 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
            n=$(ls "$d" 2>/dev/null | wc -l | tr -d ' ')
            printf "  %-12s DONE  (log gone, %s raw jsonl files)\n" "$name" "$n"
        else
            printf "  %-12s pending\n" "$name"
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
    for d in data/raw_hermes_skill data/raw_opencode_skill data/raw_openclaw_skill; do
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

# v0.4.1 OpenClaw integration stages
STAGES = [
    ("openclaw_batch",      "batch_openclaw.log"),
    ("add_platform",        "v041_openclaw.log"),
    ("pr_metadata",         "v041_openclaw.log"),
    ("join_pr",             "v041_openclaw.log"),
    ("add_licenses",        "v041_openclaw.log"),
    ("enrich_v03",          "v041_openclaw.log"),
    ("curator_subset",      "v041_openclaw.log"),
    ("skill_linter",        "v041_openclaw.log"),
]

# Estimated time per stage (from observed runs)
STAGE_SECONDS = {
    "add_platform":   3 * 60,
    "pr_metadata":    8 * 60,
    "join_pr":        1 * 60,
    "add_licenses":   8 * 60,
    "enrich_v03":     25 * 60,    # MinHash on now-larger combined corpus
    "curator_subset": 1 * 60,
    "skill_linter":   4 * 60,
}

# Parse v041_openclaw.log for stage transitions
log_path = Path("v041_openclaw.log")
done_stages = set()
current_stage = None

if log_path.exists():
    txt = log_path.read_text()
    stage_blocks = re.findall(
        r"------ (.+?) ------ .+?\n(.*?)(?=------ |\Z)",
        txt, re.DOTALL,
    )
    for stage_name, body in stage_blocks:
        if "exit: 0" in body:
            done_stages.add(stage_name.strip())
        else:
            current_stage = stage_name.strip()

finished = "v0.4.1 OpenClaw integration done" in (log_path.read_text() if log_path.exists() else "")

# Map stage label to short key
STAGE_NAME_TO_KEY = {
    "Add OpenClaw platform to release parquets": "add_platform",
    "PR metadata refresh": "pr_metadata",
    "Join PR metadata": "join_pr",
    "License metadata refresh": "add_licenses",
    "Enrich v0.3 features (MinHash + frontmatter) on full corpus": "enrich_v03",
    "Curator training subset": "curator_subset",
    "Skill linter sanity report (full corpus)": "skill_linter",
}

def fmt(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m = s // 60
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"

total_remaining = 0
# Past-tense summary of completed batches
print(f"  hermes_batch    DONE  (567 repos, 65k records)")
print(f"  opencode_batch  DONE  (1285/1302 repos, 137k records)")

# OpenClaw batch — check if running
oc_batch_log = Path("batch_openclaw.log")
oc_done = False
if oc_batch_log.exists():
    txt = oc_batch_log.read_text()
    if "Done in" in txt:
        oc_done = True
        m = re.search(r"Done in (\d+)s.*ok=(\d+).*err=(\d+)", txt)
        if m:
            print(f"  openclaw_batch  DONE  ({m.group(2)} ok, {m.group(3)} err in {m.group(1)}s)")
        else:
            print(f"  openclaw_batch  DONE")
    else:
        # Find latest [N/M] eta=Xs progress line
        last_eta = None
        for line in reversed(txt.splitlines()):
            m = re.search(r'\[(\d[\d,]*)/(\d[\d,]*)\].*eta=(\d+)s', line)
            if m:
                last_eta = (m.group(1), m.group(2), int(m.group(3)))
                break
        if last_eta:
            cur, tot, eta = last_eta
            total_remaining += eta
            print(f"  openclaw_batch  RUN   {cur}/{tot}   remaining: {fmt(eta)}")
        else:
            print(f"  openclaw_batch  starting")
else:
    print(f"  openclaw_batch  not started yet")

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
