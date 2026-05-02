#!/usr/bin/env python3
"""Fetch PR metadata for every repo in the dataset.

Strategy (fast tier):
  For each repo, paginate `/repos/{repo}/pulls?state=all&per_page=100` and
  extract (number, title, body, state, merged_at, merge_commit_sha, head_sha).
  Build a sha->PR map keyed on BOTH head_sha and merge_commit_sha.

  This catches:
    - PRs that were squash-merged (after_sha in our diffs == merge_commit_sha)
    - PRs whose head commit appears as the after_sha (fast-forward / rebase merge)
  Misses:
    - Commits in the middle of a multi-commit PR (would need per-PR /commits call)

Per-repo results cached to data/pr_cache/<owner>__<repo>.json so the script is
resumable. PR titles/bodies are deliberately preserved verbatim (not truncated)
since we want them as training-quality intent labels.

Usage:
    uv run python pr_metadata.py [--workers N] [--limit N] [--retry-errors]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


REPOS_PATH = Path("data/release/repos.parquet")
CACHE_DIR = Path("data/pr_cache")

# Fields we keep from each PR
PR_FIELDS = ("number", "title", "body", "state", "merged_at", "merge_commit_sha")


# Coarse rate-limit guard. Counter is incremented per gh call across threads;
# when it crosses a threshold we re-check the actual GH rate limit and sleep
# until reset if needed. Avoids one rate_limit call per request.
_call_counter = 0
_rate_lock = threading.Lock()


def _maybe_check_rate_limit():
    global _call_counter
    with _rate_lock:
        _call_counter += 1
        if _call_counter % 50 != 0:
            return
        # Every 50 calls, peek at remaining budget
        try:
            res = subprocess.run(
                ["gh", "api", "rate_limit", "--jq",
                 ".resources.core | \"\\(.remaining) \\(.reset)\""],
                capture_output=True, text=True, check=True, timeout=10,
            )
            remaining_s, reset_s = res.stdout.strip().split()
            remaining = int(remaining_s)
            reset_at = int(reset_s)
            if remaining < 50:
                wait = max(0, reset_at - int(time.time())) + 5
                print(f"  [rate-limit] {remaining} remaining; sleeping {wait}s "
                      f"until reset...", file=sys.stderr)
                time.sleep(wait)
        except Exception:
            # If rate_limit check fails, fall through. Per-call errors will be caught.
            pass


def gh_api_paged(path, max_pages=20):
    """Paginate a gh api endpoint that returns a JSON array. Returns list."""
    out = []
    for page in range(1, max_pages + 1):
        _maybe_check_rate_limit()
        sep = "&" if "?" in path else "?"
        url = f"{path}{sep}per_page=100&page={page}"
        result = subprocess.run(
            ["gh", "api", url],
            capture_output=True, text=True, check=False, timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.lower()
            if "404" in err or "not found" in err:
                return {"_status": "not_found", "items": []}
            if "rate limit" in err or "secondary rate" in err:
                # Hard wait for reset, then retry once
                print(f"  rate-limit hit on {path} pg {page}; waiting 60s",
                      file=sys.stderr)
                time.sleep(60)
                result = subprocess.run(
                    ["gh", "api", url],
                    capture_output=True, text=True, check=False, timeout=60,
                )
                if result.returncode != 0:
                    return {"_status": "error",
                            "_msg": result.stderr.strip()[:200], "items": out}
            else:
                return {"_status": "error",
                        "_msg": result.stderr.strip()[:200], "items": out}
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"_status": "error", "_msg": "json decode", "items": out}
        if not isinstance(data, list):
            return {"_status": "error", "_msg": "not a list", "items": out}
        out.extend(data)
        if len(data) < 100:
            break
    return {"_status": "ok", "items": out}


def fetch_repo_prs(repo_full):
    """Fetch all PRs for a repo. Return cache record."""
    cache_file = CACHE_DIR / f"{repo_full.replace('/', '__')}.json"
    if cache_file.exists():
        try:
            existing = json.loads(cache_file.read_text())
            if existing.get("status") in ("ok", "not_found"):
                return existing
        except json.JSONDecodeError:
            pass

    started = time.time()
    result = gh_api_paged(f"/repos/{repo_full}/pulls?state=all")
    status = result.get("_status", "error")
    raw_items = result.get("items", [])

    # Trim to fields we care about
    prs = []
    for it in raw_items:
        head = it.get("head") or {}
        prs.append({
            "number": it.get("number"),
            "title": it.get("title"),
            "body": it.get("body"),
            "state": it.get("state"),
            "merged_at": it.get("merged_at"),
            "merge_commit_sha": it.get("merge_commit_sha"),
            "head_sha": head.get("sha"),
            "html_url": it.get("html_url"),
        })

    record = {
        "repo": repo_full,
        "status": status,
        "n_prs": len(prs),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 2),
        "prs": prs,
    }
    if status == "error":
        record["error"] = result.get("_msg", "")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(record, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser(description="Fetch PR metadata per repo.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-fetch repos whose cache shows status=error")
    args = parser.parse_args()

    t = pq.read_table(REPOS_PATH, columns=["repo"])
    repos = t["repo"].to_pylist()
    if args.limit:
        repos = repos[: args.limit]

    # Determine pending
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    cached_ok = 0
    cached_err = 0
    for r in repos:
        cache_file = CACHE_DIR / f"{r.replace('/', '__')}.json"
        if cache_file.exists():
            try:
                existing = json.loads(cache_file.read_text())
                st = existing.get("status")
                if st == "ok" or st == "not_found":
                    cached_ok += 1
                    continue
                if st == "error" and not args.retry_errors:
                    cached_err += 1
                    continue
            except json.JSONDecodeError:
                pass
        pending.append(r)

    print(f"Repos total:      {len(repos):,}", file=sys.stderr)
    print(f"  cached ok:      {cached_ok:,}", file=sys.stderr)
    print(f"  cached error:   {cached_err:,} (use --retry-errors to redo)",
          file=sys.stderr)
    print(f"  to fetch:       {len(pending):,}", file=sys.stderr)
    print(f"Workers: {args.workers}\n", file=sys.stderr)

    if not pending:
        print("Nothing to do.", file=sys.stderr)
        return

    started = time.time()
    n_ok = n_err = n_nf = 0
    total_prs = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_repo_prs, r): r for r in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"repo": futures[fut], "status": "error", "error": str(e),
                       "n_prs": 0, "elapsed_s": 0.0}
            st = rec.get("status")
            n_prs = rec.get("n_prs", 0)
            if st == "ok":
                n_ok += 1
                total_prs += n_prs
            elif st == "not_found":
                n_nf += 1
            else:
                n_err += 1
            if i % 25 == 0 or i == len(pending):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(pending) - i) / rate if rate > 0 else 0
                print(f"  [{i:>5}/{len(pending):,}] ok={n_ok} nf={n_nf} err={n_err} "
                      f"prs={total_prs:,} | {rate:.1f}/s eta={int(eta)}s",
                      file=sys.stderr)

    print(f"\nDone in {int(time.time() - started)}s. "
          f"ok={n_ok} not_found={n_nf} err={n_err} total_prs={total_prs:,}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
