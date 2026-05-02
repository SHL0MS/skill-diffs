#!/usr/bin/env python3
"""Merge v0.3 release parquets with v0.4 new-platform parquets.

Recovers from the consolidate_v04 bug where data/raw was missing (the v0.3
JSONL was deleted post-v0.3-ship per README instruction). Pulls v0.3 from
data/v03_backup/ (downloaded from HF), adds platform column, concats with
the new platform data already consolidated in data/v04_new_only/.

Output schema: union of v0.3 columns + platform.
v0.4 records get NULL for v0.3-only enrichment columns (skill_cluster_id,
is_canonical, pr_*); these get re-populated by enrich_v03 + pr_metadata
+ add_licenses passes after this.

Outputs (overwrites data/release/):
    diffs.parquet
    diffs_clean.parquet
    skills_initial.parquet
    repos.parquet (with platform column added)
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


V03_DIR = Path("data/v03_backup")
V04_DIR = Path("data/v04_new_only")
OUT_DIR = Path("data/release")


def add_platform_to_v03(t, platform_value="claude_skill"):
    """Add a 'platform' column with constant value to all rows."""
    n = t.num_rows
    if "platform" in t.schema.names:
        return t
    return t.append_column(
        "platform", pa.array([platform_value] * n, type=pa.string()),
    )


def align_schemas(v03, v04):
    """Make v04 schema match v03 by adding null columns where v04 is missing them."""
    v03_fields = list(v03.schema)
    v03_names = {f.name for f in v03_fields}
    v04_names = {f.name for f in v04.schema}

    # Add missing-in-v04 columns as nulls
    for f in v03_fields:
        if f.name not in v04_names:
            n = v04.num_rows
            v04 = v04.append_column(f.name, pa.array([None] * n, type=f.type))

    # Drop columns that are in v04 but not v03 (shouldn't happen but defensive)
    extra = [n for n in v04.schema.names if n not in v03_names]
    if extra:
        print(f"  (dropping v04-only columns: {extra})", file=sys.stderr)
        v04 = v04.drop(extra)

    # Reorder v04 columns to match v03 order
    v04 = v04.select(v03.schema.names)

    # Cast types where they differ (large_string vs string, etc.)
    casts_needed = []
    for f03, f04 in zip(v03.schema, v04.schema):
        if f03.type != f04.type:
            casts_needed.append((f03.name, f04.type, f03.type))

    if casts_needed:
        print(f"  type adjustments needed: {casts_needed}", file=sys.stderr)
        new_arrays = []
        for col_name in v03.schema.names:
            target_type = v03.schema.field(col_name).type
            arr = v04[col_name]
            if arr.type != target_type:
                arr = arr.cast(target_type)
            new_arrays.append(arr)
        v04 = pa.Table.from_arrays(new_arrays, names=v03.schema.names)

    return v04


def merge_diff_parquet(name):
    print(f"\n=== {name} ===", file=sys.stderr)
    v03_path = V03_DIR / name
    v04_path = V04_DIR / name

    if not v03_path.exists():
        print(f"  ERROR: {v03_path} missing", file=sys.stderr)
        return False

    v03 = pq.read_table(v03_path)
    print(f"  v03: {v03.num_rows:,} rows, {len(v03.schema.names)} cols", file=sys.stderr)
    v03 = add_platform_to_v03(v03, "claude_skill")

    if v04_path.exists():
        v04 = pq.read_table(v04_path)
        print(f"  v04 new: {v04.num_rows:,} rows, {len(v04.schema.names)} cols",
              file=sys.stderr)
        v04 = align_schemas(v03, v04)
        merged = pa.concat_tables([v03, v04])
    else:
        print(f"  (no v04 new data for {name})", file=sys.stderr)
        merged = v03

    out_path = OUT_DIR / name
    pq.write_table(merged, out_path, compression="zstd")
    print(f"  wrote {out_path}: {merged.num_rows:,} rows", file=sys.stderr)
    return True


def merge_repos():
    print(f"\n=== repos.parquet ===", file=sys.stderr)
    v03 = pq.read_table(V03_DIR / "repos.parquet")
    print(f"  v03: {v03.num_rows:,} rows", file=sys.stderr)
    v03 = add_platform_to_v03(v03, "claude_skill")

    v04_path = V04_DIR / "repos.parquet"
    if v04_path.exists():
        v04 = pq.read_table(v04_path)
        print(f"  v04 new: {v04.num_rows:,} rows", file=sys.stderr)
        v04 = align_schemas(v03, v04)
        merged = pa.concat_tables([v03, v04])
        # Dedup by repo (in case any repo appears in both — preserve first occurrence)
        repos_seen = set()
        keep_idx = []
        repos_col = merged["repo"].to_pylist()
        for i, r in enumerate(repos_col):
            if r in repos_seen:
                continue
            repos_seen.add(r)
            keep_idx.append(i)
        if len(keep_idx) < merged.num_rows:
            print(f"  deduped {merged.num_rows - len(keep_idx)} repo rows", file=sys.stderr)
            merged = merged.take(keep_idx)
    else:
        merged = v03

    out_path = OUT_DIR / "repos.parquet"
    pq.write_table(merged, out_path, compression="zstd")
    print(f"  wrote {out_path}: {merged.num_rows:,} rows", file=sys.stderr)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["diffs.parquet", "diffs_clean.parquet", "skills_initial.parquet"]:
        merge_diff_parquet(name)

    merge_repos()

    print("\n=== final state ===", file=sys.stderr)
    for p in sorted(OUT_DIR.glob("*.parquet")):
        md = pq.read_metadata(p)
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<32} rows={md.num_rows:>10,}  {size_mb:>7.1f} MB",
              file=sys.stderr)


if __name__ == "__main__":
    main()
