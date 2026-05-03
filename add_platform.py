#!/usr/bin/env python3
"""Append a new platform's raw JSONL data to existing release parquets.

Use case: you ran batch_v04 for a new platform (e.g. openclaw_skill) and
want to add it to the already-enriched data/release/ parquets without
rebuilding everything from data/raw (which may not exist).

Steps:
  1. Read existing release parquets (data/release/{diffs,diffs_clean,
     skills_initial,repos}.parquet)
  2. Load new platform JSONL from data/raw_<platform>/, classify + filter
     using the same logic as consolidate_v04 (so quality_tags etc. match)
  3. Add platform column, concat with existing
  4. Write back

After this script: you should re-run pr_metadata.py + join_pr_metadata.py
+ add_licenses.py + enrich_v03.py (in that order) to populate the
delta-fetched columns and recompute global MinHash clustering. All those
scripts are idempotent so re-running is safe.

Usage:
    uv run python add_platform.py --platform openclaw_skill --raw-dir data/raw_openclaw_skill
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from classify import classify_record
from filter_quality import (
    DEFAULT_DISQUALIFYING,
    is_bot_email,
    content_hash,
    pair_hash,
)


RELEASE_DIR = Path("data/release")
HUZEY_PATH = Path("data/huzey_repos.txt")
EXPANSION_PATH = Path("data/expansion_repos.txt")


def stable_id(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def enrich_records(records, after_hash_seen, pair_hash_seen, source_seed,
                   platform):
    """Enrich one repo's records (chronologically ordered)."""
    by_skill = {}
    for i, r in enumerate(records):
        by_skill.setdefault(r["skill_path"], []).append(i)

    pre_revert_idx = set()
    for chain in by_skill.values():
        for j in range(len(chain) - 1):
            curr_idx = chain[j]
            next_idx = chain[j + 1]
            nxt_subj = (records[next_idx].get("commit_subject") or "").strip().lower()
            if nxt_subj.startswith("revert"):
                pre_revert_idx.add(curr_idx)

    for i, rec in enumerate(records):
        klass, conf, source = classify_record(rec)
        tags = []
        if is_bot_email(rec.get("commit_email")):
            tags.append("bot_author")
        if klass == "whitespace":
            tags.append("whitespace_change")
        if klass == "merge":
            tags.append("merge_commit")
        subj = (rec.get("commit_subject") or "").strip()
        if subj.lower().startswith("revert"):
            tags.append("revert_subject")
        if rec.get("is_initial"):
            tags.append("initial_commit")
        after = rec.get("after_content") or ""
        before = rec.get("before_content") or ""
        if len(after) < 500 and not rec.get("is_initial"):
            tags.append("short_skill")
        if len(before) > 200_000 or len(after) > 200_000:
            tags.append("large_blob")
        if "\ufffd" in after or "\ufffd" in before:
            tags.append("non_utf8_clean")
        added = rec.get("lines_added", 0) or 0
        removed = rec.get("lines_removed", 0) or 0
        char_delta = abs(rec.get("char_delta", 0) or 0)
        if not rec.get("is_initial") and added <= 2 and removed <= 2 and char_delta < 40:
            tags.append("micro_edit")
        ah = content_hash(after)
        if ah in after_hash_seen:
            tags.append("duplicate_after")
        after_hash_seen.add(ah)
        ph = pair_hash(before, after)
        if ph in pair_hash_seen:
            tags.append("duplicate_pair")
        pair_hash_seen.add(ph)
        if i in pre_revert_idx:
            tags.append("pre_revert")

        repo = rec["repo"]
        skill_path = rec["skill_path"]
        skill_id = stable_id(repo, skill_path)
        pair_id = stable_id(repo, skill_path, rec.get("before_sha") or "", rec["after_sha"])
        yield {
            "pair_id": pair_id,
            "skill_id": skill_id,
            "repo": repo,
            "source_seed": source_seed,
            "platform": platform,
            "skill_path": skill_path,
            "skill_name": rec.get("skill_name") or "",
            "before_sha": rec.get("before_sha"),
            "after_sha": rec["after_sha"],
            "before_content": rec.get("before_content"),
            "after_content": rec.get("after_content") or "",
            "commit_subject": subj,
            "commit_author": rec.get("commit_author") or "",
            "commit_email": rec.get("commit_email") or "",
            "commit_date": rec.get("commit_date") or "",
            "lines_added": int(added),
            "lines_removed": int(removed),
            "char_delta": int(rec.get("char_delta", 0) or 0),
            "is_initial": bool(rec.get("is_initial")),
            "intent_class": klass,
            "intent_confidence": float(round(conf, 2)),
            "intent_source": source,
            "quality_tags": tags,
        }


def main():
    parser = argparse.ArgumentParser(description="Append a platform to existing release parquets.")
    parser.add_argument("--platform", required=True,
                        help="Platform tag for new records (e.g. openclaw_skill)")
    parser.add_argument("--raw-dir", required=True,
                        help="Directory containing per-repo JSONL files")
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    args = parser.parse_args()

    rdir = Path(args.release_dir)
    raw_dir = Path(args.raw_dir)

    # ---- Load existing release parquets ----
    print("Loading existing release parquets...", file=sys.stderr)
    existing_diffs = pq.read_table(rdir / "diffs.parquet")
    existing_clean = pq.read_table(rdir / "diffs_clean.parquet")
    existing_init = pq.read_table(rdir / "skills_initial.parquet")
    existing_repos = pq.read_table(rdir / "repos.parquet")
    print(f"  diffs:           {existing_diffs.num_rows:,}", file=sys.stderr)
    print(f"  diffs_clean:     {existing_clean.num_rows:,}", file=sys.stderr)
    print(f"  skills_initial:  {existing_init.num_rows:,}", file=sys.stderr)
    print(f"  repos:           {existing_repos.num_rows:,}", file=sys.stderr)

    # Check this platform isn't already in there
    existing_platforms = set(existing_diffs["platform"].to_pylist())
    if args.platform in existing_platforms:
        print(f"\nWARNING: platform '{args.platform}' is ALREADY in diffs.parquet "
              f"({sum(1 for p in existing_diffs['platform'].to_pylist() if p == args.platform):,} rows)",
              file=sys.stderr)
        print("If you want to replace it, manually filter it out first.",
              file=sys.stderr)
        sys.exit(1)

    # Load existing dedup state from current parquets so duplicate_after /
    # duplicate_pair are computed against the global corpus (not just the
    # new platform). This ensures consistency.
    print("\nBuilding dedup hash set from existing corpus...", file=sys.stderr)
    after_hash_seen = set()
    pair_hash_seen = set()
    bf = existing_diffs["before_content"].to_pylist()
    af = existing_diffs["after_content"].to_pylist()
    for b, a in zip(bf, af):
        if a is not None:
            after_hash_seen.add(content_hash(a))
        pair_hash_seen.add(pair_hash(b or "", a or ""))
    print(f"  hash sets: {len(after_hash_seen):,} after, "
          f"{len(pair_hash_seen):,} pair", file=sys.stderr)

    # ---- Process new raw dir ----
    huzey_set = set(HUZEY_PATH.read_text().splitlines()) if HUZEY_PATH.exists() else set()
    expansion_set = set(EXPANSION_PATH.read_text().splitlines()) if EXPANSION_PATH.exists() else set()

    print(f"\nReading {raw_dir}...", file=sys.stderr)
    files = sorted(raw_dir.glob("*.jsonl"))
    print(f"  {len(files):,} JSONL files", file=sys.stderr)

    started = time.time()
    new_full = []
    new_clean = []
    new_init = []
    new_repos = []

    for fi, f in enumerate(files):
        records = []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not records:
            continue
        repo = records[0]["repo"]
        seed = ("huzey" if repo in huzey_set
                else ("expansion" if repo in expansion_set else "other"))

        n_records_repo = 0
        n_diff_repo = 0
        n_clean_diff_repo = 0
        skills = set()

        for enriched in enrich_records(
            records, after_hash_seen, pair_hash_seen, seed, args.platform,
        ):
            new_full.append(enriched)
            n_records_repo += 1
            skills.add(enriched["skill_path"])
            if enriched["is_initial"]:
                new_init.append(enriched)
            else:
                n_diff_repo += 1
                tags = set(enriched["quality_tags"])
                if not (tags & DEFAULT_DISQUALIFYING):
                    new_clean.append(enriched)
                    n_clean_diff_repo += 1

        new_repos.append({
            "repo": repo,
            "source_seed": seed,
            "platform": args.platform,
            "n_skills": len(skills),
            "n_records": n_records_repo,
            "n_diff_pairs": n_diff_repo,
            "n_clean_diff_pairs": n_clean_diff_repo,
        })

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - started
            print(f"  [{fi+1:,}/{len(files):,}] new_records={len(new_full):,} "
                  f"({elapsed:.0f}s)", file=sys.stderr)

    print(f"\n  Processed: {len(new_full):,} records, "
          f"{len(new_clean):,} clean, {len(new_init):,} initial, "
          f"{len(new_repos):,} repos", file=sys.stderr)

    # ---- Build new tables aligned to existing schema ----
    def to_aligned_table(records, schema):
        # Add null values for any column in schema that's missing from records
        names = schema.names
        rows = []
        for r in records:
            row = {n: r.get(n) for n in names}
            rows.append(row)
        return pa.Table.from_pylist(rows, schema=schema)

    print("\nWriting back to release parquets...", file=sys.stderr)
    diffs_schema = existing_diffs.schema
    clean_schema = existing_clean.schema
    init_schema = existing_init.schema
    repos_schema = existing_repos.schema

    new_diffs_t = to_aligned_table(new_full, diffs_schema)
    new_clean_t = to_aligned_table(new_clean, clean_schema)
    new_init_t = to_aligned_table(new_init, init_schema)
    new_repos_t = to_aligned_table(new_repos, repos_schema)

    combined_diffs = pa.concat_tables([existing_diffs, new_diffs_t])
    combined_clean = pa.concat_tables([existing_clean, new_clean_t])
    combined_init = pa.concat_tables([existing_init, new_init_t])
    combined_repos = pa.concat_tables([existing_repos, new_repos_t])

    for name, t in [
        ("diffs.parquet", combined_diffs),
        ("diffs_clean.parquet", combined_clean),
        ("skills_initial.parquet", combined_init),
        ("repos.parquet", combined_repos),
    ]:
        out_tmp = (rdir / name).with_suffix(".tmp.parquet")
        pq.write_table(t, out_tmp, compression="zstd")
        out_tmp.replace(rdir / name)
        print(f"  {name}: {t.num_rows:,} rows", file=sys.stderr)

    print("\nDone. Now re-run pr_metadata.py / join_pr_metadata.py / add_licenses.py / enrich_v03.py / curator_subset.py to update derived columns.", file=sys.stderr)


if __name__ == "__main__":
    main()
