#!/usr/bin/env python3
"""Add license metadata to repos.parquet via GitHub API.

For each repo in repos.parquet, fetch its license SPDX identifier and
default branch + stars + last-pushed timestamp (free metadata while we're
making the request anyway).

Output: repos.parquet updated in place with new columns:
    license_spdx (string, nullable)
    license_name (string, nullable)
    stars (int, nullable)
    default_branch (string, nullable)
    pushed_at (string, nullable, ISO 8601)
    fetched_at (string, when this row was fetched)
    fetch_status (string: ok | not_found | error)
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPOS_PATH = Path("data/release/repos.parquet")


def gh_api(path):
    """Run gh api <path> and return parsed JSON or None on error."""
    result = subprocess.run(
        ["gh", "api", path, "--cache", "1h"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        # Detect 404 vs other errors
        err = result.stderr.lower()
        if "not found" in err or "404" in err:
            return {"_status": "not_found"}
        return {"_status": "error", "_msg": result.stderr.strip()[:200]}
    try:
        data = json.loads(result.stdout)
        data["_status"] = "ok"
        return data
    except json.JSONDecodeError:
        return {"_status": "error", "_msg": "json decode"}


def fetch_one(repo_full):
    info = gh_api(f"repos/{repo_full}")
    status = info.get("_status", "error")
    license_obj = info.get("license") or {}
    return {
        "repo": repo_full,
        "license_spdx": license_obj.get("spdx_id") if license_obj else None,
        "license_name": license_obj.get("name") if license_obj else None,
        "stars": info.get("stargazers_count"),
        "default_branch": info.get("default_branch"),
        "pushed_at": info.get("pushed_at"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fetch_status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Add license metadata to repos.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    t = pq.read_table(REPOS_PATH)
    repos = t["repo"].to_pylist()
    if args.limit:
        repos = repos[: args.limit]
    print(f"Fetching metadata for {len(repos):,} repos...", file=sys.stderr)

    started = time.time()
    results = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, r): r for r in repos}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["repo"]] = res
            n_done += 1
            if n_done % 100 == 0 or n_done == len(repos):
                elapsed = time.time() - started
                rate = n_done / elapsed if elapsed else 0
                eta = (len(repos) - n_done) / rate if rate else 0
                ok = sum(1 for r in results.values() if r["fetch_status"] == "ok")
                err = sum(1 for r in results.values() if r["fetch_status"] != "ok")
                print(f"  [{n_done:,}/{len(repos):,}] ok={ok} err={err} eta={int(eta)}s",
                      file=sys.stderr)

    # Merge into existing repos.parquet
    print("\nMerging into repos.parquet...", file=sys.stderr)
    rows = t.to_pylist()
    for r in rows:
        meta = results.get(r["repo"])
        if meta:
            r.update({
                "license_spdx": meta.get("license_spdx"),
                "license_name": meta.get("license_name"),
                "stars": meta.get("stars"),
                "default_branch": meta.get("default_branch"),
                "pushed_at": meta.get("pushed_at"),
                "fetched_at": meta.get("fetched_at"),
                "fetch_status": meta.get("fetch_status"),
            })
        else:
            r.update({
                "license_spdx": None, "license_name": None,
                "stars": None, "default_branch": None,
                "pushed_at": None, "fetched_at": None,
                "fetch_status": "not_attempted",
            })

    # Idempotent: if license columns already exist (from a prior run), drop
    # them from the input schema so we don't end up with duplicate fields.
    license_field_names = {
        "license_spdx", "license_name", "stars", "default_branch",
        "pushed_at", "fetched_at", "fetch_status",
    }
    new_fields = [f for f in t.schema if f.name not in license_field_names]
    new_fields.extend([
        pa.field("license_spdx", pa.string()),
        pa.field("license_name", pa.string()),
        pa.field("stars", pa.int32()),
        pa.field("default_branch", pa.string()),
        pa.field("pushed_at", pa.string()),
        pa.field("fetched_at", pa.string()),
        pa.field("fetch_status", pa.string()),
    ])
    new_schema = pa.schema(new_fields)
    out_tmp = REPOS_PATH.with_suffix(".tmp.parquet")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=new_schema),
        out_tmp, compression="zstd",
    )
    out_tmp.replace(REPOS_PATH)
    print(f"Wrote {REPOS_PATH}", file=sys.stderr)

    # Quick stats
    ok = sum(1 for r in results.values() if r["fetch_status"] == "ok")
    nf = sum(1 for r in results.values() if r["fetch_status"] == "not_found")
    er = sum(1 for r in results.values() if r["fetch_status"] == "error")
    licensed = sum(1 for r in results.values() if r.get("license_spdx"))
    print(f"\nStatus:")
    print(f"  ok:        {ok:,}")
    print(f"  not_found: {nf:,}")
    print(f"  error:     {er:,}")
    print(f"  with SPDX: {licensed:,}")

    from collections import Counter
    licenses = Counter(r.get("license_spdx") for r in results.values() if r.get("license_spdx"))
    print(f"\nTop 10 licenses by repo count:")
    for lic, n in licenses.most_common(10):
        print(f"  {lic:<20} {n:,}")


if __name__ == "__main__":
    main()
