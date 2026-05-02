#!/usr/bin/env python3
"""Join PR metadata from data/pr_cache/ into release parquets.

Reads all per-repo PR caches and builds a global (repo, sha) -> PR map keyed
on both head_sha and merge_commit_sha. For each row in
diffs/diffs_clean/skills_initial.parquet, attaches PR fields:

    pr_number       (int32, nullable)
    pr_title        (string, nullable)
    pr_body         (string, nullable)
    pr_state        (string, nullable: open|closed|merged)
    pr_merged_at    (string, nullable)
    pr_url          (string, nullable)
    pr_match_kind   (string, nullable: head_sha|merge_commit_sha|none)

Adds these columns in place to the existing parquets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


RELEASE_DIR = Path("data/release")
CACHE_DIR = Path("data/pr_cache")
TARGET_FILES = ["diffs.parquet", "diffs_clean.parquet", "skills_initial.parquet"]

NEW_FIELDS = [
    pa.field("pr_number", pa.int32()),
    pa.field("pr_title", pa.string()),
    pa.field("pr_body", pa.string()),
    pa.field("pr_state", pa.string()),
    pa.field("pr_merged_at", pa.string()),
    pa.field("pr_url", pa.string()),
    pa.field("pr_match_kind", pa.string()),
]


def build_sha_map():
    """Walk pr_cache/ and build (repo, sha) -> (pr_dict, match_kind)."""
    cache_files = sorted(CACHE_DIR.glob("*.json"))
    print(f"Loading {len(cache_files):,} cache files...", file=sys.stderr)
    sha_map = {}
    n_prs_total = 0
    n_repos_with_prs = 0
    n_repos_err = 0
    started = time.time()

    for i, cf in enumerate(cache_files):
        try:
            data = json.loads(cf.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("status") != "ok":
            if data.get("status") == "error":
                n_repos_err += 1
            continue
        repo = data["repo"]
        prs = data.get("prs", [])
        if prs:
            n_repos_with_prs += 1
        for pr in prs:
            n_prs_total += 1
            entry = {
                "pr_number": pr.get("number"),
                "pr_title": pr.get("title"),
                "pr_body": pr.get("body"),
                "pr_state": pr.get("state"),
                "pr_merged_at": pr.get("merged_at"),
                "pr_url": pr.get("html_url"),
            }
            head_sha = pr.get("head_sha")
            merge_sha = pr.get("merge_commit_sha")
            # Prefer merge_commit_sha as primary key (squash-merge case is most
            # common). Only set head_sha key if not already present (don't
            # overwrite a more authoritative merge_commit_sha hit).
            if merge_sha:
                key = (repo, merge_sha)
                if key not in sha_map:
                    sha_map[key] = (entry, "merge_commit_sha")
            if head_sha:
                key = (repo, head_sha)
                if key not in sha_map:
                    sha_map[key] = (entry, "head_sha")

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:,}/{len(cache_files):,}] "
                  f"prs={n_prs_total:,} ({time.time() - started:.0f}s)",
                  file=sys.stderr)

    print(f"  Loaded {n_prs_total:,} PRs from {n_repos_with_prs:,} repos "
          f"({n_repos_err} errors).", file=sys.stderr)
    print(f"  Map size: {len(sha_map):,} unique (repo, sha) keys.", file=sys.stderr)
    return sha_map


def enrich_table(path, sha_map):
    print(f"\n=== {path.name} ===", file=sys.stderr)
    t = pq.read_table(path)
    n = t.num_rows
    print(f"  rows: {n:,}", file=sys.stderr)

    repos = t["repo"].to_pylist()
    afters = t["after_sha"].to_pylist()

    pr_number = []
    pr_title = []
    pr_body = []
    pr_state = []
    pr_merged_at = []
    pr_url = []
    pr_match_kind = []
    n_match = 0
    n_match_merge = 0
    n_match_head = 0

    for repo, sha in zip(repos, afters):
        if sha is None:
            pr_number.append(None); pr_title.append(None); pr_body.append(None)
            pr_state.append(None); pr_merged_at.append(None); pr_url.append(None)
            pr_match_kind.append(None)
            continue
        hit = sha_map.get((repo, sha))
        if hit is None:
            pr_number.append(None); pr_title.append(None); pr_body.append(None)
            pr_state.append(None); pr_merged_at.append(None); pr_url.append(None)
            pr_match_kind.append(None)
        else:
            entry, kind = hit
            pr_number.append(entry["pr_number"])
            pr_title.append(entry["pr_title"])
            pr_body.append(entry["pr_body"])
            pr_state.append(entry["pr_state"])
            pr_merged_at.append(entry["pr_merged_at"])
            pr_url.append(entry["pr_url"])
            pr_match_kind.append(kind)
            n_match += 1
            if kind == "merge_commit_sha":
                n_match_merge += 1
            else:
                n_match_head += 1

    pct = 100.0 * n_match / n if n else 0
    print(f"  matched: {n_match:,} ({pct:.1f}%) "
          f"[merge_sha={n_match_merge:,} head_sha={n_match_head:,}]",
          file=sys.stderr)

    # Build new table by appending columns
    new_table = t
    for fld, col in zip(
        NEW_FIELDS,
        [pr_number, pr_title, pr_body, pr_state, pr_merged_at, pr_url, pr_match_kind],
    ):
        # Skip if column already exists (idempotent re-runs)
        if fld.name in new_table.schema.names:
            new_table = new_table.set_column(
                new_table.schema.get_field_index(fld.name),
                fld, pa.array(col, type=fld.type),
            )
        else:
            new_table = new_table.append_column(fld, pa.array(col, type=fld.type))

    out_tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(new_table, out_tmp, compression="zstd")
    out_tmp.replace(path)
    print(f"  wrote {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Join PR metadata into release parquets.")
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    args = parser.parse_args()

    sha_map = build_sha_map()
    if not sha_map:
        print("ERROR: empty sha_map. Did you run pr_metadata.py?", file=sys.stderr)
        sys.exit(1)

    rdir = Path(args.release_dir)
    for fname in TARGET_FILES:
        p = rdir / fname
        if p.exists():
            enrich_table(p, sha_map)
        else:
            print(f"  (skip missing: {p})", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
