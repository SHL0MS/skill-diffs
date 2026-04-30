#!/usr/bin/env python3
"""Aggregate per-repo JSONL shards into parquet files for the final dataset.

Produces (in data/release/):
  diffs.parquet           — all diff records with classification + quality tags
  diffs_clean.parquet     — only "clean" subset (passes default filters)
  bundled.parquet         — bundled resource snapshots (skill folder at HEAD)
  repos.parquet           — per-repo provenance (record counts, license, etc.)
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def stable_id(*parts):
    """Stable SHA1 hash of joined parts."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def stream_records(in_dir):
    """Yield records from every JSONL shard."""
    for f in sorted(Path(in_dir).glob("*.jsonl")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def normalize_diff(rec):
    """Produce final-schema diff record with stable IDs."""
    repo = rec["repo"]
    skill_path = rec["skill_path"]
    skill_id = stable_id(repo, skill_path)
    pair_id = stable_id(repo, skill_path, rec["before_sha"] or "", rec["after_sha"])
    return {
        "pair_id": pair_id,
        "skill_id": skill_id,
        "repo": repo,
        "skill_path": skill_path,
        "skill_name": rec.get("skill_name") or "",
        "before_sha": rec.get("before_sha"),
        "after_sha": rec["after_sha"],
        "before_content": rec.get("before_content"),
        "after_content": rec.get("after_content") or "",
        "commit_subject": rec.get("commit_subject") or "",
        "commit_author": rec.get("commit_author") or "",
        "commit_email": rec.get("commit_email") or "",
        "commit_date": rec.get("commit_date") or "",
        "lines_added": int(rec.get("lines_added", 0)),
        "lines_removed": int(rec.get("lines_removed", 0)),
        "char_delta": int(rec.get("char_delta", 0)),
        "is_initial": bool(rec.get("is_initial")),
        "intent_class": rec.get("intent_class") or "unknown",
        "intent_confidence": float(rec.get("intent_confidence", 0.0)),
        "intent_source": rec.get("intent_source") or "regex",
        "quality_tags": rec.get("quality_tags") or [],
    }


def normalize_bundled(rec):
    repo = rec["repo"]
    skill_path = rec["skill_path"]
    skill_id = stable_id(repo, skill_path)
    return {
        "skill_id": skill_id,
        "repo": repo,
        "skill_path": skill_path,
        "skill_dir": rec.get("skill_dir") or "",
        "skill_name": rec.get("skill_name") or "",
        "head_sha": rec.get("head_sha") or "",
        "bundled_count": int(rec.get("bundled_count", 0)),
        "bundled_text_count": int(rec.get("bundled_text_count", 0)),
        "bundled_files": [
            {
                "path": b.get("path", ""),
                "size": int(b.get("size", 0)),
                "content": b.get("content"),
                "binary_or_oversize": bool(b.get("binary_or_oversize", False)),
            }
            for b in (rec.get("bundled_files") or [])
        ],
    }


def write_parquet(records, schema, out_path, batch_size=2000):
    """Stream records into parquet, batched, with given schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    batch = []
    n = 0
    for r in records:
        batch.append(r)
        if len(batch) >= batch_size:
            tbl = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(tbl)
            n += len(batch)
            batch = []
    if batch:
        tbl = pa.Table.from_pylist(batch, schema=schema)
        writer.write_table(tbl)
        n += len(batch)
    writer.close()
    return n


def diff_schema():
    return pa.schema([
        ("pair_id", pa.string()),
        ("skill_id", pa.string()),
        ("repo", pa.string()),
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


def bundled_schema():
    file_struct = pa.struct([
        ("path", pa.string()),
        ("size", pa.int64()),
        ("content", pa.large_string()),
        ("binary_or_oversize", pa.bool_()),
    ])
    return pa.schema([
        ("skill_id", pa.string()),
        ("repo", pa.string()),
        ("skill_path", pa.string()),
        ("skill_dir", pa.string()),
        ("skill_name", pa.string()),
        ("head_sha", pa.string()),
        ("bundled_count", pa.int32()),
        ("bundled_text_count", pa.int32()),
        ("bundled_files", pa.list_(file_struct)),
    ])


def repos_schema():
    return pa.schema([
        ("repo", pa.string()),
        ("source_seed", pa.string()),  # huzey / expansion
        ("n_skills", pa.int32()),
        ("n_records", pa.int32()),
        ("n_diff_pairs", pa.int32()),
        ("n_clean_diff_pairs", pa.int32()),
    ])


def build_repo_provenance(diff_in_dir, clean_in_dir):
    """Aggregate per-repo stats from the per-file JSONL shards."""
    huzey = set(Path("data/huzey_repos.txt").read_text().splitlines())
    expansion = set()
    if Path("data/expansion_repos.txt").exists():
        expansion = set(Path("data/expansion_repos.txt").read_text().splitlines())

    by_repo = {}
    for f in sorted(Path(diff_in_dir).glob("*.jsonl")):
        repo = None
        n_records = 0
        n_diff = 0
        skills = set()
        with open(f) as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if repo is None:
                    repo = r["repo"]
                n_records += 1
                if not r.get("is_initial"):
                    n_diff += 1
                skills.add(r["skill_path"])
        if repo is None:
            continue
        n_clean = 0
        clean_path = Path(clean_in_dir) / f.name
        if clean_path.exists():
            with open(clean_path) as fp:
                for line in fp:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not r.get("is_initial"):
                        n_clean += 1
        by_repo[repo] = {
            "repo": repo,
            "source_seed": "huzey" if repo in huzey else ("expansion" if repo in expansion else "other"),
            "n_skills": len(skills),
            "n_records": n_records,
            "n_diff_pairs": n_diff,
            "n_clean_diff_pairs": n_clean,
        }
    return list(by_repo.values())


def main():
    parser = argparse.ArgumentParser(description="Build final parquet dataset.")
    parser.add_argument("--diff-input", default="data/filtered",
                        help="Per-repo JSONL with quality_tags applied")
    parser.add_argument("--clean-input", default="data/clean",
                        help="Per-repo clean JSONL (already filtered)")
    parser.add_argument("--bundled-input", default="data/bundled")
    parser.add_argument("--output-dir", default="data/release")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # All diffs
    print("Writing diffs.parquet...", file=sys.stderr)
    n_all = write_parquet(
        (normalize_diff(r) for r in stream_records(args.diff_input)),
        diff_schema(),
        out_dir / "diffs.parquet",
    )
    print(f"  {n_all:,} records", file=sys.stderr)

    # Clean diffs
    if Path(args.clean_input).exists():
        print("Writing diffs_clean.parquet...", file=sys.stderr)
        n_clean = write_parquet(
            (normalize_diff(r) for r in stream_records(args.clean_input)),
            diff_schema(),
            out_dir / "diffs_clean.parquet",
        )
        print(f"  {n_clean:,} records", file=sys.stderr)

    # Bundled
    if Path(args.bundled_input).exists():
        print("Writing bundled.parquet...", file=sys.stderr)
        n_bundled = write_parquet(
            (normalize_bundled(r) for r in stream_records(args.bundled_input)),
            bundled_schema(),
            out_dir / "bundled.parquet",
        )
        print(f"  {n_bundled:,} records", file=sys.stderr)

    # Repo provenance
    print("Writing repos.parquet...", file=sys.stderr)
    repo_rows = build_repo_provenance(args.diff_input, args.clean_input)
    tbl = pa.Table.from_pylist(repo_rows, schema=repos_schema())
    pq.write_table(tbl, out_dir / "repos.parquet", compression="zstd")
    print(f"  {len(repo_rows):,} repos", file=sys.stderr)

    # Print summary
    print("\nFiles:", file=sys.stderr)
    for p in sorted(out_dir.glob("*.parquet")):
        size_mb = p.stat().st_size / 1_000_000
        print(f"  {p.name:<24} {size_mb:>8.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
