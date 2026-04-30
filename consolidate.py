#!/usr/bin/env python3
"""Streaming consolidation: data/raw/*.jsonl  →  data/release/*.parquet

Applies in a single pass:
  - intent classification (regex)
  - quality tagging
  - pre-revert detection (per-skill chain)
  - global content-hash dedup
  - schema normalization with stable IDs
  - zstd-compressed parquet output

Drops the need for data/classified, data/filtered, data/clean intermediates.
After verifying parquet output, you can `rm -rf data/raw` and reclaim ~10 GB.

Outputs (in data/release/):
  diffs.parquet       — every record with intent + quality_tags
  diffs_clean.parquet — subset that survives default disqualifying filters
                        AND is not an initial commit (true diff pairs only)
  skills_initial.parquet — initial-commit records only (skill creation snapshots)
  repos.parquet       — per-repo provenance / counts
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
    BOT_EMAIL_PATTERNS,
    DEFAULT_DISQUALIFYING,
    is_bot_email,
    content_hash,
    pair_hash,
)


def stable_id(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def diff_schema():
    return pa.schema([
        ("pair_id", pa.string()),
        ("skill_id", pa.string()),
        ("repo", pa.string()),
        ("source_seed", pa.string()),  # huzey | expansion | other
        ("skill_path", pa.string()),
        ("skill_name", pa.string()),
        ("before_sha", pa.string()),
        ("after_sha", pa.string()),
        ("before_content", pa.large_string()),
        ("after_content", pa.large_string()),
        ("commit_subject", pa.string()),
        ("commit_author", pa.string()),
        ("commit_email", pa.string()),
        ("commit_date", pa.string()),
        ("lines_added", pa.int32()),
        ("lines_removed", pa.int32()),
        ("char_delta", pa.int32()),
        ("is_initial", pa.bool_()),
        ("intent_class", pa.string()),
        ("intent_confidence", pa.float32()),
        ("intent_source", pa.string()),
        ("quality_tags", pa.list_(pa.string())),
    ])


def repos_schema():
    return pa.schema([
        ("repo", pa.string()),
        ("source_seed", pa.string()),
        ("n_skills", pa.int32()),
        ("n_records", pa.int32()),
        ("n_diff_pairs", pa.int32()),
        ("n_clean_diff_pairs", pa.int32()),
    ])


def enrich_records_for_repo(records, after_hash_seen, pair_hash_seen, source_seed):
    """Yield enriched records for one repo's worth of raw records.
    Detects pre_revert within this repo. records assumed in extract.py order
    (per-skill chronological).
    """
    # Group by skill_path to detect pre-reverts
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
        # Intent classification
        klass, conf, source = classify_record(rec)
        # Quality tags
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
    parser = argparse.ArgumentParser(description="Streaming consolidation to parquet.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/release")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    huzey_set = set(Path("data/huzey_repos.txt").read_text().splitlines())
    expansion_set = set()
    if Path("data/expansion_repos.txt").exists():
        expansion_set = set(Path("data/expansion_repos.txt").read_text().splitlines())

    schema = diff_schema()
    full_writer = pq.ParquetWriter(out_dir / "diffs.parquet", schema, compression="zstd")
    clean_writer = pq.ParquetWriter(out_dir / "diffs_clean.parquet", schema, compression="zstd")
    initial_writer = pq.ParquetWriter(out_dir / "skills_initial.parquet", schema, compression="zstd")

    full_batch, clean_batch, initial_batch = [], [], []

    after_hash_seen = set()
    pair_hash_seen = set()

    n_total = n_clean = n_initial = 0
    repo_stats = {}
    files = sorted(in_dir.glob("*.jsonl"))
    print(f"Consolidating {len(files):,} per-repo shards...", file=sys.stderr)
    started = time.time()

    for fi, f in enumerate(files):
        # Load all records for this repo (preserves chronological order)
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
        seed = "huzey" if repo in huzey_set else ("expansion" if repo in expansion_set else "other")
        n_records_repo = 0
        n_diff_repo = 0
        n_clean_diff_repo = 0
        skills = set()

        for enriched in enrich_records_for_repo(records, after_hash_seen, pair_hash_seen, seed):
            full_batch.append(enriched)
            n_total += 1
            n_records_repo += 1
            skills.add(enriched["skill_path"])

            is_init = enriched["is_initial"]
            if is_init:
                initial_batch.append(enriched)
                n_initial += 1
            else:
                n_diff_repo += 1
                # Clean = no disqualifying tag AND not initial
                tags = set(enriched["quality_tags"])
                if not (tags & DEFAULT_DISQUALIFYING):
                    clean_batch.append(enriched)
                    n_clean += 1
                    n_clean_diff_repo += 1

            # Flush batches
            if len(full_batch) >= args.batch_size:
                full_writer.write_table(pa.Table.from_pylist(full_batch, schema=schema))
                full_batch.clear()
            if len(clean_batch) >= args.batch_size:
                clean_writer.write_table(pa.Table.from_pylist(clean_batch, schema=schema))
                clean_batch.clear()
            if len(initial_batch) >= args.batch_size:
                initial_writer.write_table(pa.Table.from_pylist(initial_batch, schema=schema))
                initial_batch.clear()

        repo_stats[repo] = {
            "repo": repo,
            "source_seed": seed,
            "n_skills": len(skills),
            "n_records": n_records_repo,
            "n_diff_pairs": n_diff_repo,
            "n_clean_diff_pairs": n_clean_diff_repo,
        }

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - started
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - fi - 1) / rate if rate > 0 else 0
            print(f"  [{fi+1}/{len(files)}] {n_total:,} records, "
                  f"{n_clean:,} clean diffs | eta={int(eta)}s", file=sys.stderr)

    # Flush remaining
    if full_batch:
        full_writer.write_table(pa.Table.from_pylist(full_batch, schema=schema))
    if clean_batch:
        clean_writer.write_table(pa.Table.from_pylist(clean_batch, schema=schema))
    if initial_batch:
        initial_writer.write_table(pa.Table.from_pylist(initial_batch, schema=schema))
    full_writer.close()
    clean_writer.close()
    initial_writer.close()

    # Repos parquet
    repo_rows = list(repo_stats.values())
    pq.write_table(
        pa.Table.from_pylist(repo_rows, schema=repos_schema()),
        out_dir / "repos.parquet", compression="zstd",
    )

    elapsed = int(time.time() - started)
    print(f"\nDone in {elapsed}s.", file=sys.stderr)
    print(f"  diffs.parquet:           {n_total:,} records", file=sys.stderr)
    print(f"  diffs_clean.parquet:     {n_clean:,} records", file=sys.stderr)
    print(f"  skills_initial.parquet:  {n_initial:,} records", file=sys.stderr)
    print(f"  repos.parquet:           {len(repo_rows):,} rows", file=sys.stderr)
    print()
    print("Output sizes:", file=sys.stderr)
    for p in sorted(out_dir.glob("*.parquet")):
        size_mb = p.stat().st_size / 1_000_000
        print(f"  {p.name:<28} {size_mb:>8.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
