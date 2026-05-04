#!/usr/bin/env python3
"""Add a single 0.0-1.0 `quality_score` column derived from existing signals.

Lets users do `top = diffs.filter(quality_score >= 0.7)` without writing
custom logic. Computed from:

  +0.20  has SPDX license (license_spdx is non-null)
  +0.10  has stars≥10
  +0.10  has stars≥100 (additional, so ≥100 = +0.20)
  +0.15  has pr_title (richer intent)
  +0.20  no quality_tags in disqualifying set
  +0.10  no quality_tags in strict-disqualifying set (above and beyond)
  +0.05  intent_class in {feat, fix, refactor, docs} (high-signal classes)
  +0.10  body length 500-30000 chars (not stub, not megablob)

Max score = 1.00. Designed so 0.7+ threshold gives the 'good' tier.

Idempotent: drops existing quality_score column before re-adding.

Reads + writes: diffs.parquet, diffs_clean.parquet, skills_initial.parquet,
                curator_training.parquet, curator_training_strict.parquet
"""
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


RELEASE_DIR = Path("data/release")
TARGETS = [
    "diffs.parquet",
    "diffs_clean.parquet",
    "skills_initial.parquet",
    "curator_training.parquet",
    "curator_training_strict.parquet",
]

DISQUALIFYING = {
    "bot_author", "whitespace_change", "merge_commit", "revert_subject",
    "pre_revert", "duplicate_pair", "micro_edit", "short_skill",
    "invalid_frontmatter", "same_author_dup",
}
STRICT_DISQUALIFYING = {
    "no_license", "low_engagement", "placeholder_content", "pii_email",
}
HIGH_SIGNAL_INTENTS = {"feat", "fix", "refactor", "docs"}


def compute_score(row, repo_meta):
    """row: dict with quality_tags, intent_class, after_content, repo, pr_title, etc."""
    score = 0.0
    repo = row.get("repo")
    meta = repo_meta.get(repo, {})
    license_spdx = row.get("license_spdx") or meta.get("license_spdx")
    stars = row.get("stars") if row.get("stars") is not None else meta.get("stars")

    if license_spdx:
        score += 0.20
    if stars is not None:
        if stars >= 10:
            score += 0.10
        if stars >= 100:
            score += 0.10
    if row.get("pr_title"):
        score += 0.15

    tags = set(row.get("quality_tags") or [])
    if not (tags & DISQUALIFYING):
        score += 0.20
    if not (tags & STRICT_DISQUALIFYING):
        score += 0.10

    ic = row.get("intent_class")
    if ic in HIGH_SIGNAL_INTENTS:
        score += 0.05

    after_len = len(row.get("after_content") or "")
    if 500 <= after_len <= 30000:
        score += 0.10

    return min(score, 1.0)


def process(p, repo_meta):
    print(f"\n=== {p.name} ===", file=sys.stderr)
    t = pq.read_table(p)
    n = t.num_rows
    print(f"  rows: {n:,}", file=sys.stderr)

    rows = t.to_pylist()
    started = time.time()
    scores = []
    for i, r in enumerate(rows):
        scores.append(compute_score(r, repo_meta))
        if (i + 1) % 200_000 == 0:
            print(f"  [{i+1:,}/{n:,}] ({time.time()-started:.0f}s)", file=sys.stderr)

    arr = pa.array(scores, type=pa.float32())
    if "quality_score" in t.schema.names:
        idx = t.schema.get_field_index("quality_score")
        new_t = t.set_column(idx, pa.field("quality_score", pa.float32()), arr)
    else:
        new_t = t.append_column("quality_score", arr)

    out_tmp = p.with_suffix(".tmp.parquet")
    pq.write_table(new_t, out_tmp, compression="zstd")
    out_tmp.replace(p)

    # Distribution
    bins = [0, 0.3, 0.5, 0.7, 0.85, 1.01]
    labels = ["<0.30", "0.30-0.5", "0.50-0.7", "0.70-0.85", "0.85-1.0"]
    counts = [0] * len(labels)
    for s in scores:
        for i, hi in enumerate(bins[1:]):
            if s < hi:
                counts[i] += 1
                break
    print(f"  quality_score distribution:", file=sys.stderr)
    for label, c in zip(labels, counts):
        print(f"    {label:<10} {c:>10,}  ({100*c/n:.1f}%)", file=sys.stderr)


def main():
    print("Loading repos.parquet for repo-level metadata...", file=sys.stderr)
    repos_t = pq.read_table(
        RELEASE_DIR / "repos.parquet",
        columns=["repo", "stars", "license_spdx"],
    )
    repo_meta = {}
    for r in repos_t.to_pylist():
        repo_meta[r["repo"]] = {
            "stars": r.get("stars"),
            "license_spdx": r.get("license_spdx"),
        }
    print(f"  {len(repo_meta):,} repos indexed", file=sys.stderr)

    for fname in TARGETS:
        p = RELEASE_DIR / fname
        if not p.exists():
            continue
        process(p, repo_meta)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
