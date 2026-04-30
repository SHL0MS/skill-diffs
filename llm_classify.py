#!/usr/bin/env python3
"""LLM classification of `intent_class == "unknown"` records.

Reads data/release/diffs.parquet, filters to records whose regex classifier
gave up, batches them through Claude Haiku, and writes back updated parquet
files (diffs, diffs_clean, skills_initial) with new intent fields.

Usage:
    uv run python llm_classify.py [--batch-size N] [--workers N] [--limit N] [--dry-run]

Reads ANTHROPIC_API_KEY from macOS Keychain (opencode/ANTHROPIC_WORK).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from anthropic import Anthropic, APIError, RateLimitError


VALID_CLASSES = {
    "feat", "fix", "docs", "style", "refactor", "perf", "test",
    "build", "ci", "chore", "revert", "merge", "initial", "other",
}

MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = """You classify commit subjects for changes to AI agent skill files (SKILL.md). Pick ONE label for each commit subject from this exact list:

feat: a new feature, capability, instruction, or section added
fix: a bug fix, error correction, or behavioral fix
docs: documentation, comment, or wording-only changes
style: formatting, whitespace, lint, code-style only
refactor: restructuring or rewording without behavior change
perf: performance improvement
test: tests added or changed
build: build system, dependencies, packaging
ci: CI/CD configuration changes
chore: routine maintenance (rename, cleanup, version bump, etc.)
revert: reverts a prior change
merge: merge commit
initial: initial creation of the file
other: does not fit any of the above

Output rules:
- Output ONE LINE per input, in the same order, with format: "<n>: <label>"
- Use the exact lowercase label, no quotes, no extra text"""


def get_anthropic_key():
    result = subprocess.run(
        ["security", "find-generic-password", "-a", "opencode", "-s", "ANTHROPIC_WORK", "-w"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print("ERROR: ANTHROPIC_WORK key not found in keychain", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def classify_batch(client, subjects):
    """Classify a batch of commit subjects. Returns list of labels (same order)."""
    numbered = "\n".join(f"{i+1}: {s}" for i, s in enumerate(subjects))
    user_msg = f"Classify these {len(subjects)} commit subjects:\n\n{numbered}"

    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
            return parse_response(text, len(subjects))
        except RateLimitError:
            time.sleep(2 ** attempt)
        except APIError as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return ["other"] * len(subjects)


def parse_response(text, expected):
    """Parse '1: feat\\n2: fix\\n...' into list of N labels."""
    labels = ["other"] * expected
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)[:.\)]\s*([a-z]+)\s*$", line, re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        label = m.group(2).lower()
        if 0 <= idx < expected and label in VALID_CLASSES:
            labels[idx] = label
    return labels


def collect_unknown_records(source_path):
    """Return list of (pair_id, commit_subject) for records with intent_class='unknown'."""
    t = pq.read_table(source_path, columns=["pair_id", "intent_class", "commit_subject"])
    df = t.to_pylist()
    return [(r["pair_id"], r["commit_subject"] or "") for r in df if r["intent_class"] == "unknown"]


def update_parquet(in_path, out_path, mapping):
    """Read parquet, update intent_* fields per mapping, write back."""
    t = pq.read_table(in_path)
    df = t.to_pylist()
    n_updated = 0
    for r in df:
        if r.get("intent_class") == "unknown":
            new = mapping.get(r["pair_id"])
            if new is not None:
                r["intent_class"] = new
                r["intent_confidence"] = 0.85
                r["intent_source"] = "llm"
                n_updated += 1
    schema = t.schema
    out_path_tmp = out_path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(df, schema=schema), out_path_tmp, compression="zstd")
    out_path_tmp.replace(out_path)
    return n_updated


def main():
    parser = argparse.ArgumentParser(description="LLM classify unknown intents.")
    parser.add_argument("--release-dir", default="data/release")
    parser.add_argument("--source", default="diffs_clean",
                        choices=["diffs_clean", "diffs", "skills_initial"],
                        help="Which parquet's unknowns to classify (mapping is then applied to ALL files)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N unknowns (for cost control / testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually call API or write parquet")
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    source_path = release_dir / f"{args.source}.parquet"
    if not source_path.exists():
        print(f"ERROR: {source_path} not found. Run consolidate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading unknown records from {source_path.name}...", file=sys.stderr)
    unknowns = collect_unknown_records(source_path)
    print(f"Found {len(unknowns):,} records with intent_class='unknown' in {args.source}",
          file=sys.stderr)

    if args.limit:
        unknowns = unknowns[: args.limit]
        print(f"Limited to {len(unknowns):,}", file=sys.stderr)

    # Build batches
    batches = []
    for i in range(0, len(unknowns), args.batch_size):
        batch = unknowns[i:i + args.batch_size]
        batches.append(batch)
    print(f"Will run {len(batches):,} API calls of up to {args.batch_size} subjects each",
          file=sys.stderr)

    # Cost estimate
    avg_input = 250 + args.batch_size * 30
    avg_output = args.batch_size * 8
    total_in = len(batches) * avg_input
    total_out = len(batches) * avg_output
    cost = (total_in / 1e6) * 1.0 + (total_out / 1e6) * 5.0
    print(f"Rough cost estimate: ~${cost:.2f} ({total_in:,} input + {total_out:,} output tokens)",
          file=sys.stderr)

    if args.dry_run:
        print("[dry-run] not calling API", file=sys.stderr)
        return

    api_key = get_anthropic_key()
    client = Anthropic(api_key=api_key)

    print(f"\nDispatching with {args.workers} concurrent workers...", file=sys.stderr)
    started = time.time()
    mapping = {}
    n_done = 0

    def process_batch(batch):
        subjects = [s for _, s in batch]
        labels = classify_batch(client, subjects)
        return [(pid, lbl) for (pid, _), lbl in zip(batch, labels)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            try:
                results = fut.result()
                for pid, lbl in results:
                    mapping[pid] = lbl
            except Exception as e:
                print(f"  batch failed: {e}", file=sys.stderr)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(batches):
                elapsed = time.time() - started
                rate = n_done / elapsed if elapsed else 0
                eta = (len(batches) - n_done) / rate if rate else 0
                print(f"  [{n_done}/{len(batches)}] {len(mapping):,} classified | eta={int(eta)}s",
                      file=sys.stderr)

    print(f"\nClassified {len(mapping):,} of {len(unknowns):,} unknowns "
          f"in {int(time.time() - started)}s", file=sys.stderr)

    # Save mapping for inspection / re-runs
    mapping_path = release_dir / "llm_classifications.json"
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"Saved mapping to {mapping_path}", file=sys.stderr)

    # Update each parquet file in place
    print("\nUpdating parquet files...", file=sys.stderr)
    for name in ("diffs.parquet", "diffs_clean.parquet", "skills_initial.parquet"):
        p = release_dir / name
        if not p.exists():
            continue
        n = update_parquet(p, p, mapping)
        print(f"  {name}: updated {n:,} records", file=sys.stderr)


if __name__ == "__main__":
    main()
