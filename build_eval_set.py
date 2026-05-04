#!/usr/bin/env python3
"""Build a stratified held-out eval set for the curator skill-patch task.

Samples N examples per major intent_class to give meaningful per-category
metrics (vs. the original v0.4 200-random-sample set which had ~14 examples
per class — statistical noise on small differences).

Default strata: feat / fix / refactor / docs / chore (50 each = 250 total)

Quality filter applied per-record:
  - len(before) ≥ 200 AND len(after) ≥ 200
  - len(intent_text) ≥ 12
  - 0.3 ≤ len(after) / len(before) ≤ 3.0  (drop deletions and explosions)
  - len(after) ≤ 30k chars (fit in reasonable context)
  - is_canonical (already enforced by curator_training_strict)

Sources: data/release/curator_training_strict.parquet (license-clean tier)

Output: data/release/curator_eval_set_v2.parquet
"""
import argparse
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CURATOR_PARQUET = Path("data/release/curator_training_strict.parquet")
OUT_PATH = Path("data/release/curator_eval_set_v2.parquet")

DEFAULT_STRATA = {
    "feat": 50,
    "fix": 50,
    "refactor": 50,
    "docs": 50,
    "chore": 50,
}


def quality_filter(r):
    before = r.get("before_content") or ""
    after = r.get("after_content") or ""
    intent = r.get("intent_text") or ""
    if len(before) < 200 or len(after) < 200:
        return False
    if len(intent) < 12:
        return False
    ratio = len(after) / max(len(before), 1)
    if not (0.3 <= ratio <= 3.0):
        return False
    if len(after) > 30000:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(CURATOR_PARQUET))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=50,
                        help="Examples per intent class (overrides default strata)")
    parser.add_argument("--strata", nargs="+", default=list(DEFAULT_STRATA.keys()),
                        help="Intent classes to sample from")
    args = parser.parse_args()

    print(f"Loading {args.source}...", file=sys.stderr)
    t = pq.read_table(args.source)
    print(f"  {t.num_rows:,} candidates", file=sys.stderr)

    # Apply quality filter
    rows = t.to_pylist()
    rows = [r for r in rows if quality_filter(r)]
    print(f"  {len(rows):,} pass quality filter", file=sys.stderr)

    # Group by intent_class
    by_class = {}
    for r in rows:
        ic = r.get("intent_class") or "unknown"
        by_class.setdefault(ic, []).append(r)

    print("\nAvailable per intent class (after quality filter):", file=sys.stderr)
    for ic, lst in sorted(by_class.items(), key=lambda x: -len(x[1])):
        print(f"  {ic:<12} {len(lst):>6,}", file=sys.stderr)

    # Sample
    rng = random.Random(args.seed)
    sample = []
    print(f"\nSampling {args.per_class} per intent_class:", file=sys.stderr)
    for ic in args.strata:
        pool = by_class.get(ic, [])
        if not pool:
            print(f"  {ic:<12} 0 (no records)", file=sys.stderr)
            continue
        n = min(args.per_class, len(pool))
        picked = rng.sample(pool, n)
        sample.extend(picked)
        print(f"  {ic:<12} {n} (from pool of {len(pool):,})", file=sys.stderr)

    if not sample:
        print("ERROR: empty sample", file=sys.stderr)
        sys.exit(1)

    # Slim down to needed columns + add stratification metadata
    out_cols = [
        "pair_id", "skill_id", "repo", "skill_name", "platform",
        "intent_text", "commit_subject",
        "pr_title", "pr_body",
        "before_content", "after_content",
        "intent_class",
        "license_spdx", "stars",
        "skill_cluster_id", "is_canonical",
        "quality_tags",
    ]
    out_cols = [c for c in out_cols if c in t.schema.names]

    # Build aligned schema-respecting table
    out_rows = []
    for r in sample:
        out_rows.append({c: r.get(c) for c in out_cols})
    out_t = pa.Table.from_pylist(out_rows)

    pq.write_table(out_t, args.out, compression="zstd")

    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"  rows: {out_t.num_rows:,}", file=sys.stderr)
    print(f"  size: {size_mb:.1f} MB", file=sys.stderr)
    print(f"  cols: {len(out_t.schema.names)}", file=sys.stderr)


if __name__ == "__main__":
    main()
