#!/usr/bin/env python3
"""Batch-extract SKILL.md commit history from a list of GitHub repos.

Usage:
    uv run python batch.py [--repos PATH] [--workers N] [--limit N]

Reads owner/repo per line from --repos (default: data/huzey_repos.txt).
Writes one JSONL per repo to data/raw/<owner>__<repo>.jsonl, skipping
already-processed repos. Maintains data/manifest.jsonl with per-repo stats.
"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Reuse the extract function from our prototype
from extract import extract_repo, DEFAULT_REPO_TIMEOUT_S


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "manifest.jsonl"


def output_path(repo_full):
    owner, repo = repo_full.split("/", 1)
    safe = f"{owner}__{repo}".replace("/", "_")
    return RAW_DIR / f"{safe}.jsonl"


def already_processed(repo_full, manifest_index):
    """True if this repo has a successful entry in the manifest."""
    entry = manifest_index.get(repo_full)
    return entry is not None and entry.get("status") == "ok"


def load_manifest_index():
    """Map repo_full -> latest manifest entry."""
    if not MANIFEST.exists():
        return {}
    index = {}
    with open(MANIFEST) as f:
        for line in f:
            try:
                entry = json.loads(line)
                index[entry["repo"]] = entry
            except json.JSONDecodeError:
                continue
    return index


def append_manifest(entry):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry) + "\n")


def process_one(args):
    """Worker: extract one repo, return manifest entry."""
    if isinstance(args, tuple):
        repo_full, timeout = args
    else:
        repo_full, timeout = args, DEFAULT_REPO_TIMEOUT_S
    started = time.time()
    out = output_path(repo_full)
    try:
        repo_url = f"https://github.com/{repo_full}.git"
        records = extract_repo(repo_url, out, quiet=True, timeout=timeout)
        elapsed = time.time() - started
        return {
            "repo": repo_full,
            "status": "ok",
            "records": records,
            "elapsed_s": round(elapsed, 2),
            "output": str(out),
        }
    except Exception as e:
        if out.exists():
            out.unlink()
        is_timeout = type(e).__name__ == "RepoTimeoutError"
        return {
            "repo": repo_full,
            "status": "timeout" if is_timeout else "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
            "elapsed_s": round(time.time() - started, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch SKILL.md extraction.")
    parser.add_argument(
        "--repos", default=str(DATA_DIR / "huzey_repos.txt"),
        help="Path to file with one owner/repo per line",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only consider first N repos from input file")
    parser.add_argument("--max-new", type=int, default=None,
                        help="Process at most this many NEW (unprocessed) repos this run")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry repos that previously failed")
    parser.add_argument("--timeout", type=int, default=DEFAULT_REPO_TIMEOUT_S,
                        help=f"Per-repo timeout in seconds (default: {DEFAULT_REPO_TIMEOUT_S}s = 30 min)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    repo_list = Path(args.repos).read_text().splitlines()
    repo_list = [r.strip() for r in repo_list if r.strip()]
    if args.limit:
        repo_list = repo_list[: args.limit]

    manifest_index = load_manifest_index()

    pending = []
    skipped_ok = 0
    skipped_err = 0
    for r in repo_list:
        if already_processed(r, manifest_index):
            skipped_ok += 1
            continue
        if r in manifest_index and not args.retry_errors:
            skipped_err += 1
            continue
        pending.append(r)

    if args.max_new is not None:
        pending = pending[: args.max_new]

    print(f"Repos in list: {len(repo_list)}", file=sys.stderr)
    print(f"  Already done: {skipped_ok}", file=sys.stderr)
    print(f"  Previously failed (skip): {skipped_err}", file=sys.stderr)
    print(f"  To process now: {len(pending)}", file=sys.stderr)
    print(f"Workers: {args.workers}\n", file=sys.stderr)

    started = time.time()
    n_ok = 0
    n_err = 0
    total_records = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, (r, args.timeout)): r for r in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            append_manifest(entry)
            if entry["status"] == "ok":
                n_ok += 1
                total_records += entry["records"]
                marker = "OK"
                detail = f"{entry['records']} records"
            elif entry["status"] == "timeout":
                n_err += 1
                marker = "TIMEOUT"
                detail = f"{entry['elapsed_s']}s"
            else:
                n_err += 1
                marker = "ERR"
                detail = entry["error"][:80]
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate if rate > 0 else 0
            print(
                f"[{i}/{len(pending)}] {marker} {entry['repo']}: "
                f"{detail} ({entry['elapsed_s']}s) "
                f"| ok={n_ok} err={n_err} | eta={int(eta)}s",
                file=sys.stderr,
            )

    print(
        f"\nDone in {int(time.time() - started)}s. "
        f"ok={n_ok} err={n_err} total_records={total_records}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
