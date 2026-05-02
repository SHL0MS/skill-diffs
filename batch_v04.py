#!/usr/bin/env python3
"""Batch-extract for v0.4 (OpenCode/Hermes/OpenClaw/Cursor).

Like batch.py but parameterized by extractor and output dir, and tags the
manifest with `platform` for downstream consolidate.

Usage:
    uv run python batch_v04.py \\
        --repos data/opencode_repos.txt \\
        --platform opencode_skill \\
        --extractor skill_md \\
        --workers 16

    uv run python batch_v04.py \\
        --repos data/cursor_repos.txt \\
        --platform cursor_rule \\
        --extractor cursor \\
        --workers 16
"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract import extract_repo as extract_skill_md
from extract_cursor import extract_repo as extract_cursor


EXTRACTORS = {
    "skill_md": extract_skill_md,
    "cursor": extract_cursor,
}


def output_path(raw_dir, repo_full):
    safe = repo_full.replace("/", "__")
    return raw_dir / f"{safe}.jsonl"


def load_manifest_index(manifest):
    if not manifest.exists():
        return {}
    index = {}
    with open(manifest) as f:
        for line in f:
            try:
                entry = json.loads(line)
                index[entry["repo"]] = entry
            except json.JSONDecodeError:
                continue
    return index


def append_manifest(manifest, entry):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a") as f:
        f.write(json.dumps(entry) + "\n")


def process_one(args):
    """Worker: extract one repo. args = (repo_full, raw_dir_str, extractor_name, platform)."""
    repo_full, raw_dir_str, extractor_name, platform = args
    raw_dir = Path(raw_dir_str)
    started = time.time()
    out = output_path(raw_dir, repo_full)
    try:
        repo_url = f"https://github.com/{repo_full}.git"
        extractor = EXTRACTORS[extractor_name]
        records = extractor(repo_url, out, quiet=True)
        return {
            "repo": repo_full,
            "platform": platform,
            "status": "ok",
            "records": records,
            "elapsed_s": round(time.time() - started, 2),
            "output": str(out),
        }
    except Exception as e:
        if out.exists():
            out.unlink()
        return {
            "repo": repo_full,
            "platform": platform,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
            "elapsed_s": round(time.time() - started, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="v0.4 batch extraction.")
    parser.add_argument("--repos", required=True,
                        help="Path to file with one owner/repo per line")
    parser.add_argument("--platform", required=True,
                        choices=["opencode_skill", "hermes_skill",
                                 "openclaw_skill", "cursor_rule"],
                        help="Platform tag for manifest entries")
    parser.add_argument("--extractor", required=True,
                        choices=list(EXTRACTORS.keys()),
                        help="Which extractor to use")
    parser.add_argument("--output-dir", default=None,
                        help="Raw JSONL output dir (default: data/raw_<platform>)")
    parser.add_argument("--manifest", default=None,
                        help="Manifest path (default: data/manifest_<platform>.jsonl)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.output_dir) if args.output_dir else Path(f"data/raw_{args.platform}")
    manifest = Path(args.manifest) if args.manifest else Path(f"data/manifest_{args.platform}.jsonl")
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_list = Path(args.repos).read_text().splitlines()
    repo_list = [r.strip() for r in repo_list if r.strip()]
    if args.limit:
        repo_list = repo_list[: args.limit]

    manifest_index = load_manifest_index(manifest)
    pending = []
    skipped_ok = skipped_err = 0
    for r in repo_list:
        e = manifest_index.get(r)
        if e and e.get("status") == "ok":
            skipped_ok += 1
            continue
        if e and not args.retry_errors:
            skipped_err += 1
            continue
        pending.append(r)

    if args.max_new is not None:
        pending = pending[: args.max_new]

    print(f"Platform:        {args.platform}", file=sys.stderr)
    print(f"Extractor:       {args.extractor}", file=sys.stderr)
    print(f"Raw output dir:  {raw_dir}", file=sys.stderr)
    print(f"Manifest:        {manifest}", file=sys.stderr)
    print(f"Repos in list:   {len(repo_list):,}", file=sys.stderr)
    print(f"  Already done:  {skipped_ok:,}", file=sys.stderr)
    print(f"  Prev failed:   {skipped_err:,}", file=sys.stderr)
    print(f"  To process:    {len(pending):,}", file=sys.stderr)
    print(f"Workers: {args.workers}\n", file=sys.stderr)

    if not pending:
        return

    started = time.time()
    n_ok = n_err = 0
    total_records = 0

    worker_args = [
        (r, str(raw_dir), args.extractor, args.platform) for r in pending
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, w): w[0] for w in worker_args}
        for i, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            append_manifest(manifest, entry)
            if entry["status"] == "ok":
                n_ok += 1
                total_records += entry["records"]
                marker = "OK"
                detail = f"{entry['records']} records"
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

    print(f"\nDone in {int(time.time() - started)}s. "
          f"ok={n_ok} err={n_err} total_records={total_records}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
