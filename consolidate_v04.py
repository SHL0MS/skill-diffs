#!/usr/bin/env python3
"""v0.4 consolidation: merge multiple raw dirs into per-format release parquets.

Reads:
  data/raw/                       -> SKILL.md (legacy claude_skill platform)
  data/raw_opencode_skill/        -> SKILL.md (opencode platform)
  data/raw_hermes_skill/          -> SKILL.md (hermes platform)
  data/raw_openclaw_skill/        -> SKILL.md (openclaw platform)
  data/raw_cursor_rule/           -> cursor rule format

Writes:
  data/release/diffs.parquet            (SKILL.md across all 4 platforms, `platform` col)
  data/release/diffs_clean.parquet      (clean SKILL.md subset)
  data/release/skills_initial.parquet   (initial commits, SKILL.md)
  data/release/cursor_diffs.parquet     (cursor rules, all)
  data/release/cursor_diffs_clean.parquet
  data/release/cursor_rules_initial.parquet
  data/release/repos.parquet            (per-repo provenance + platform)

Re-uses classify + filter_quality logic from consolidate.py, plus adds
'platform' column derived from input dir.
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


# Maps raw dir name -> (platform, format)
PLATFORM_DIRS = {
    "raw":                "claude_skill",        # legacy v0.1-v0.3 corpus
    "raw_opencode_skill": "opencode_skill",
    "raw_hermes_skill":   "hermes_skill",
    "raw_openclaw_skill": "openclaw_skill",
    "raw_cursor_rule":    "cursor_rule",
}

# Format groups for output: format -> set of platforms
SKILL_MD_PLATFORMS = {
    "claude_skill", "opencode_skill", "hermes_skill", "openclaw_skill",
}
CURSOR_PLATFORMS = {"cursor_rule"}


def stable_id(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def diff_schema(include_cluster_cols=False):
    cols = [
        ("pair_id", pa.string()),
        ("skill_id", pa.string()),
        ("repo", pa.string()),
        ("source_seed", pa.string()),
        ("platform", pa.string()),
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
    ]
    if include_cluster_cols:
        # Will be filled by enrich pass; keep nullable now
        cols += [
            ("skill_cluster_id", pa.string()),
            ("is_canonical", pa.bool_()),
        ]
    return pa.schema(cols)


def repos_schema():
    return pa.schema([
        ("repo", pa.string()),
        ("source_seed", pa.string()),
        ("platform", pa.string()),
        ("format", pa.string()),
        ("n_skills", pa.int32()),
        ("n_records", pa.int32()),
        ("n_diff_pairs", pa.int32()),
        ("n_clean_diff_pairs", pa.int32()),
    ])


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
    parser = argparse.ArgumentParser(description="v0.4 streaming consolidation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/release")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    huzey_set = set(Path("data/huzey_repos.txt").read_text().splitlines())
    expansion_set = set()
    if Path("data/expansion_repos.txt").exists():
        expansion_set = set(Path("data/expansion_repos.txt").read_text().splitlines())

    # SKILL.md schema for all skill_md platforms (claude/oc/hermes/openclaw)
    skill_schema = diff_schema()
    skill_full = pq.ParquetWriter(out_dir / "diffs.parquet", skill_schema, compression="zstd")
    skill_clean = pq.ParquetWriter(out_dir / "diffs_clean.parquet", skill_schema, compression="zstd")
    skill_init = pq.ParquetWriter(out_dir / "skills_initial.parquet", skill_schema, compression="zstd")

    # Cursor schema (same shape, separate file)
    cursor_full = pq.ParquetWriter(out_dir / "cursor_diffs.parquet", skill_schema, compression="zstd")
    cursor_clean = pq.ParquetWriter(out_dir / "cursor_diffs_clean.parquet", skill_schema, compression="zstd")
    cursor_init = pq.ParquetWriter(out_dir / "cursor_rules_initial.parquet", skill_schema, compression="zstd")

    # Two parallel dedup-state spaces (one per format, since cross-format
    # duplicates aren't really duplicates conceptually)
    dedup = {
        "skill_md": {"after": set(), "pair": set()},
        "cursor_rule": {"after": set(), "pair": set()},
    }

    repo_stats = {}
    n_total = {"skill_md": 0, "cursor_rule": 0}
    n_clean = {"skill_md": 0, "cursor_rule": 0}
    n_init = {"skill_md": 0, "cursor_rule": 0}

    started = time.time()
    BATCH = args.batch_size
    skill_full_buf, skill_clean_buf, skill_init_buf = [], [], []
    cursor_full_buf, cursor_clean_buf, cursor_init_buf = [], [], []

    def flush(buf, writer):
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=skill_schema))
            buf.clear()

    for raw_subdir, platform in PLATFORM_DIRS.items():
        in_dir = data_root / raw_subdir
        if not in_dir.exists():
            print(f"  (skip missing dir: {in_dir})", file=sys.stderr)
            continue
        files = sorted(in_dir.glob("*.jsonl"))
        if not files:
            print(f"  (skip empty dir: {in_dir})", file=sys.stderr)
            continue
        print(f"\n=== {raw_subdir}  platform={platform}  files={len(files):,} ===",
              file=sys.stderr)

        format_group = "cursor_rule" if platform in CURSOR_PLATFORMS else "skill_md"

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
                records,
                dedup[format_group]["after"],
                dedup[format_group]["pair"],
                seed, platform,
            ):
                n_total[format_group] += 1
                n_records_repo += 1
                skills.add(enriched["skill_path"])

                if format_group == "skill_md":
                    full_buf, clean_buf, init_buf = (
                        skill_full_buf, skill_clean_buf, skill_init_buf)
                    full_writer, clean_writer, init_writer = (
                        skill_full, skill_clean, skill_init)
                else:
                    full_buf, clean_buf, init_buf = (
                        cursor_full_buf, cursor_clean_buf, cursor_init_buf)
                    full_writer, clean_writer, init_writer = (
                        cursor_full, cursor_clean, cursor_init)

                full_buf.append(enriched)
                if enriched["is_initial"]:
                    init_buf.append(enriched)
                    n_init[format_group] += 1
                else:
                    n_diff_repo += 1
                    tags = set(enriched["quality_tags"])
                    if not (tags & DEFAULT_DISQUALIFYING):
                        clean_buf.append(enriched)
                        n_clean[format_group] += 1
                        n_clean_diff_repo += 1

                if len(full_buf) >= BATCH:
                    flush(full_buf, full_writer)
                if len(clean_buf) >= BATCH:
                    flush(clean_buf, clean_writer)
                if len(init_buf) >= BATCH:
                    flush(init_buf, init_writer)

            repo_stats[repo] = {
                "repo": repo,
                "source_seed": seed,
                "platform": platform,
                "format": format_group,
                "n_skills": len(skills),
                "n_records": n_records_repo,
                "n_diff_pairs": n_diff_repo,
                "n_clean_diff_pairs": n_clean_diff_repo,
            }

            if (fi + 1) % 200 == 0:
                elapsed = time.time() - started
                print(f"  [{fi+1}/{len(files)}] {raw_subdir}: "
                      f"skill_md={n_total['skill_md']:,} "
                      f"cursor={n_total['cursor_rule']:,} "
                      f"({elapsed:.0f}s)", file=sys.stderr)

    # Final flush
    flush(skill_full_buf, skill_full)
    flush(skill_clean_buf, skill_clean)
    flush(skill_init_buf, skill_init)
    flush(cursor_full_buf, cursor_full)
    flush(cursor_clean_buf, cursor_clean)
    flush(cursor_init_buf, cursor_init)

    skill_full.close(); skill_clean.close(); skill_init.close()
    cursor_full.close(); cursor_clean.close(); cursor_init.close()

    # repos.parquet
    repo_rows = list(repo_stats.values())
    pq.write_table(
        pa.Table.from_pylist(repo_rows, schema=repos_schema()),
        out_dir / "repos.parquet", compression="zstd",
    )

    elapsed = int(time.time() - started)
    print(f"\nDone in {elapsed}s.", file=sys.stderr)
    print(f"  SKILL.md format:", file=sys.stderr)
    print(f"    diffs.parquet:           {n_total['skill_md']:,}", file=sys.stderr)
    print(f"    diffs_clean.parquet:     {n_clean['skill_md']:,}", file=sys.stderr)
    print(f"    skills_initial.parquet:  {n_init['skill_md']:,}", file=sys.stderr)
    print(f"  Cursor rule format:", file=sys.stderr)
    print(f"    cursor_diffs.parquet:        {n_total['cursor_rule']:,}", file=sys.stderr)
    print(f"    cursor_diffs_clean.parquet:  {n_clean['cursor_rule']:,}", file=sys.stderr)
    print(f"    cursor_rules_initial:        {n_init['cursor_rule']:,}", file=sys.stderr)
    print(f"  repos.parquet: {len(repo_rows):,} rows", file=sys.stderr)
    print(file=sys.stderr)
    print("Output sizes:", file=sys.stderr)
    for p in sorted(out_dir.glob("*.parquet")):
        size_mb = p.stat().st_size / 1_000_000
        print(f"  {p.name:<32} {size_mb:>8.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
