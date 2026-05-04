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
    for log in v05_finish.log v041_openclaw.log v04_finish.log v04_trimmed.log v04_pipeline.log; do
        if [ -f "$log" ]; then
            echo "── orchestrator (last 10 lines of $log) ─"
            tail -10 "$log"
            echo
            break
        fi
    done

    # --- v0.5 in-flight pipelines (the things actually running right now) ---
    echo "── v0.5 in-flight ───────────────────────────────────"
    for entry in "bundled_refresh:bundled_refresh.log:extract_bundled.py" \
                 "embed_cluster:embed_cluster.log:embed_cluster.py" \
                 "baselines_v05:baselines_v05.log:run_baselines_v05.sh"; do
        name=$(echo "$entry" | cut -d: -f1)
        f=$(echo "$entry" | cut -d: -f2)
        proc_pat=$(echo "$entry" | cut -d: -f3)
        running=$(pgrep -f "$proc_pat" > /dev/null && echo yes || echo no)
        if [ "$running" = "yes" ] && [ -f "$f" ]; then
            # Try to extract progress + eta
            last=$(tail -1 "$f" 2>/dev/null)
            frag=$(echo "$last" | grep -oE '\[[0-9,]+/[0-9,]+\].*eta=[0-9]+s' | head -1)
            if [ -z "$frag" ]; then
                # tqdm progress for embed_cluster
                frag=$(echo "$last" | grep -oE '[0-9]+%[^]]*]' | tail -1)
            fi
            if [ -n "$frag" ]; then
                printf "  %-18s RUN   %s\n" "$name" "$frag"
            else
                printf "  %-18s RUN\n" "$name"
            fi
        elif [ -f "$f" ]; then
            done_line=$(grep -E "Done|done|exit: 0" "$f" 2>/dev/null | tail -1)
            if [ -n "$done_line" ]; then
                printf "  %-18s DONE\n" "$name"
            else
                printf "  %-18s ??\n" "$name"
            fi
        else
            printf "  %-18s pending\n" "$name"
        fi
    done
    echo

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

    # --- full pipeline ETA ---
    echo "── full pipeline ETA (v0.5) ──────────────────────────"
    python3 - <<'EOF'
import os, re, subprocess, time
from pathlib import Path

def fmt(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m = s // 60
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"

def is_running(pat):
    try:
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
    except Exception:
        return False

def extract_eta(logf, regex_patterns):
    """Return (current, total, eta_seconds) from latest progress line, or None."""
    if not Path(logf).exists():
        return None
    txt = Path(logf).read_text(errors="replace")
    for line in reversed(txt.splitlines()):
        for pat in regex_patterns:
            m = re.search(pat, line)
            if m:
                try:
                    cur = int(m.group(1).replace(",", ""))
                    tot = int(m.group(2).replace(",", ""))
                    if m.lastindex >= 3:
                        eta = int(m.group(3))
                    else:
                        eta = None
                    return (cur, tot, eta, line)
                except (ValueError, IndexError):
                    continue
    return None

# Tier 1 pipelines
TIER1 = [
    ("bundled_refresh", "bundled_refresh.log", "extract_bundled.py",
     [r'\[(\d+)/(\d+)\].*eta=(\d+)s']),
    ("embed_cluster", "embed_cluster.log", "embed_cluster.py",
     # tqdm output: "Batches:  43%|████| 4473/10389 [47:28<1:02:46,  1.57it/s]"
     [r'(\d+)/(\d+) \[(?:\d+:)?\d+:\d+<(\d+):(\d+):?(\d+)?']),
    ("baselines_v05", "baselines_v05.log", "run_baselines_v05",
     [r'\[(\d+)/(\d+)\].*eta=(\d+)s']),
]

# Tier 2/3 finish chain stages
FINISH_STAGES = [
    ("aggregate bundled.parquet",                  "aggregate_bundled",      3 * 60),
    ("add semantic clusters",                       "add_semantic_clusters",  60),
    ("add semantic diff structure column",          "add_semantic_diff",     12 * 60),
    ("add prompt-injection tag",                    "add_injection_tag",      8 * 60),
    ("regenerate curator_training (default + strict)", "curator_subset",      90),
    ("add aggregate quality_score column",          "add_quality_score",      6 * 60),
]

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

total_remaining = 0
# Past-tense summary of completed batches
print(f"  hermes_batch    DONE  (567 repos, 65k records)")
print(f"  opencode_batch  DONE  (1285/1302 repos, 137k records)")

print(f"  openclaw_batch  DONE  (1631 ok, 91 err)")
print(f"  v0.4.1 base     DONE  (consolidate + pr + license + enrich + curator)")
print()
print("  -- Tier 1 (in flight) --")

# Tier 1 status
tier1_running = []
tier1_eta_max = 0
for name, logf, proc_pat, regexes in TIER1:
    running = is_running(proc_pat)
    eta_info = extract_eta(logf, regexes)
    if running and eta_info:
        cur, tot, _eta_s_unused, line = eta_info
        # For embed_cluster the regex captures cur/tot but eta is parsed
        # below from the tqdm time format
        eta_s = None
        if name == "embed_cluster":
            # tqdm format: "<1:02:46," (HH:MM:SS) or "<59:43," (MM:SS)
            m = re.search(r'<(?:(\d+):)?(\d+):(\d+),', line)
            if m:
                h = int(m.group(1) or 0)
                mn = int(m.group(2) or 0)
                sc = int(m.group(3) or 0)
                eta_s = h * 3600 + mn * 60 + sc
        else:
            # batch_v04 / extract_bundled / eval_curator: "eta=NNNs"
            m = re.search(r'eta=(\d+)s', line)
            if m:
                eta_s = int(m.group(1))
        if eta_s is None:
            eta_s = 0

        # Special case for baselines_v05: 4 sequential stages
        # (identity / intent_only / claude-haiku-4-5 / claude-sonnet-4-5)
        if name == "baselines_v05":
            # Count completed stages from log
            txt = Path(logf).read_text(errors="replace") if Path(logf).exists() else ""
            stage_markers = len(re.findall(r"^------ ", txt, re.MULTILINE))
            stages_done = txt.count("exit: 0")
            # Estimated remaining stage durations (sec): identity~33m, intent_only~33m,
            # haiku~55m, sonnet~75m
            stage_estimates = [33*60, 33*60, 55*60, 75*60]
            total_est = sum(stage_estimates)
            # Remaining stages from index `stages_done`
            remaining_future = sum(stage_estimates[stages_done+1:]) if stages_done < 4 else 0
            # Plus eta for current stage
            full_eta = eta_s + remaining_future
            pct = 100 * cur / tot if tot else 0
            print(f"  {name:<18} RUN   stage {stages_done+1}/4  {cur:>4,}/{tot}  "
                  f"current-stage eta {fmt(eta_s)}  total-remaining {fmt(full_eta)}")
            tier1_eta_max = max(tier1_eta_max, full_eta)
            tier1_running.append(name)
            continue

        pct = 100 * cur / tot if tot else 0
        print(f"  {name:<18} RUN   {cur:>6,}/{tot:<6,} ({pct:.0f}%)  eta {fmt(eta_s)}")
        tier1_running.append(name)
        tier1_eta_max = max(tier1_eta_max, eta_s)
    elif running:
        print(f"  {name:<18} RUN   (starting up)")
        tier1_running.append(name)
    else:
        # Done check via Path
        if Path(logf).exists():
            print(f"  {name:<18} DONE")
        else:
            print(f"  {name:<18} pending")

print()
print("  -- Tier 2/3 finish chain (waits on all of Tier 1) --")

# Parse v05_finish.log for completed stages
finish_log = Path("v05_finish.log")
done_stages = set()
current_stage = None
finish_finished = False
if finish_log.exists():
    txt = finish_log.read_text(errors="replace")
    if "v0.5 finish chain done" in txt:
        finish_finished = True
    stage_blocks = re.findall(
        r"------ (.+?) ------ .+?\n(.*?)(?=------ |\Z)",
        txt, re.DOTALL,
    )
    for stage_name, body in stage_blocks:
        if "exit: 0" in body:
            done_stages.add(stage_name.strip())
        else:
            current_stage = stage_name.strip()

finish_remaining = 0
for label, key, secs in FINISH_STAGES:
    if label in done_stages:
        print(f"    {key:<22} DONE")
    elif label == current_stage:
        finish_remaining += secs // 2  # halfway estimate
        print(f"    {key:<22} RUN")
    else:
        finish_remaining += secs
        print(f"    {key:<22} queued (est {secs//60}m)")

# Total ETA = max(tier1) + finish_chain (since finish waits)
total_remaining = tier1_eta_max + finish_remaining

print()

baselines_done = not is_running("run_baselines_v05")
all_done = finish_finished and baselines_done
embed_done = not is_running("embed_cluster.py")
bundled_done = not is_running("extract_bundled.py")

if all_done:
    print(f"  PIPELINE COMPLETE — both finish chain and baselines done.")
    print(f"  Ready for README review + HF upload + git commit.")
elif finish_finished:
    # Finish chain done, baselines still running. Dataset itself is ready;
    # only the eval baselines table for the data card is pending.
    if tier1_running:
        # Sum up remaining ETAs for things still in flight
        remaining = tier1_eta_max
        print(f"  DATASET READY (finish chain done at {time.strftime('%H:%M', time.localtime())}).")
        print(f"  Baselines still running ({fmt(remaining)} remaining for full eval table).")
        print()
        print(f"  Options:")
        print(f"    A) Ship dataset now, add baseline numbers as a follow-up (~10 min ship)")
        print(f"    B) Wait for baselines to fill the data card eval table")
        finish = time.localtime(time.time() + remaining)
        print(f"  IF WAITING: ESTIMATED FINISH: {time.strftime('%a %H:%M', finish)}")
    else:
        print(f"  PIPELINE COMPLETE — both finish chain and baselines done.")
elif tier1_running:
    print(f"  TOTAL REMAINING: ~{fmt(total_remaining)} (tier1 long-pole {fmt(tier1_eta_max)}, then ~{fmt(finish_remaining)} finish)")
    finish = time.localtime(time.time() + total_remaining)
    print(f"  ESTIMATED FINISH: {time.strftime('%a %H:%M', finish)}")
else:
    print(f"  TOTAL REMAINING (finish only): {fmt(finish_remaining)}")
    finish = time.localtime(time.time() + finish_remaining)
    print(f"  ESTIMATED FINISH: {time.strftime('%a %H:%M', finish)}")
EOF
    echo

    # --- footer ---
    echo "(refreshing every 5s — Ctrl-C to exit)"
    sleep 5
done
