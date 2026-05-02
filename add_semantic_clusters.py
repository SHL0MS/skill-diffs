#!/usr/bin/env python3
"""Merge semantic_clusters.parquet into release parquets in place.

Adds two columns:
    skill_semantic_cluster_id (string)
    is_semantic_canonical     (bool)

Records whose skill_id isn't in the cluster map (shouldn't happen for valid
runs) get null values.
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TARGETS = [
    "diffs.parquet",
    "diffs_clean.parquet",
    "skills_initial.parquet",
    "cursor_diffs.parquet",
    "cursor_diffs_clean.parquet",
    "cursor_rules_initial.parquet",
]


def main():
    parser = argparse.ArgumentParser(description="Merge semantic clusters into parquets.")
    parser.add_argument("--release-dir", default="data/release")
    parser.add_argument("--clusters", default="data/semantic_clusters.parquet")
    args = parser.parse_args()

    rdir = Path(args.release_dir)
    clusters_path = Path(args.clusters)

    if not clusters_path.exists():
        print(f"ERROR: {clusters_path} not found", file=sys.stderr)
        sys.exit(1)

    cluster_t = pq.read_table(clusters_path)
    sid_to_cluster = dict(zip(
        cluster_t["skill_id"].to_pylist(),
        cluster_t["skill_semantic_cluster_id"].to_pylist(),
    ))
    sid_to_canonical = dict(zip(
        cluster_t["skill_id"].to_pylist(),
        cluster_t["is_semantic_canonical"].to_pylist(),
    ))
    print(f"Loaded clusters: {len(sid_to_cluster):,} skills",
          file=sys.stderr)

    for fname in DEFAULT_TARGETS:
        p = rdir / fname
        if not p.exists():
            continue
        print(f"\n=== {fname} ===", file=sys.stderr)
        t = pq.read_table(p)
        sids = t["skill_id"].to_pylist()
        cluster_col = [sid_to_cluster.get(sid) for sid in sids]
        canonical_col = [sid_to_canonical.get(sid) for sid in sids]
        n_matched = sum(1 for c in cluster_col if c is not None)
        print(f"  rows: {t.num_rows:,}  matched: {n_matched:,}",
              file=sys.stderr)

        # Add or replace columns
        new = t
        for name, col, dtype in [
            ("skill_semantic_cluster_id", cluster_col, pa.string()),
            ("is_semantic_canonical", canonical_col, pa.bool_()),
        ]:
            if name in new.schema.names:
                new = new.set_column(
                    new.schema.get_field_index(name),
                    pa.field(name, dtype),
                    pa.array(col, type=dtype),
                )
            else:
                new = new.append_column(
                    pa.field(name, dtype),
                    pa.array(col, type=dtype),
                )

        out_tmp = p.with_suffix(".tmp.parquet")
        pq.write_table(new, out_tmp, compression="zstd")
        out_tmp.replace(p)
        print(f"  wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
