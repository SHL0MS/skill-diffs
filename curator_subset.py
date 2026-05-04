#!/usr/bin/env python3
"""Derive the Curator training subset from the v0.4 release parquets.

The curator (in Hermes Agent) maintains a skill library by:
  - Patching drift / outdated content in skills
  - Consolidating overlapping skills
  - Spotting outdated content (model names, deprecated APIs)
  - Deciding keep/patch/archive per skill

This script extracts the strict-clean diff pairs that look like the kind of
edits the curator should learn to make:

  - canonical (skill_cluster_id's representative)
  - valid frontmatter (no 'invalid_frontmatter' tag)
  - no same-author duplicate
  - has a non-trivial intent label (commit_subject OR pr_title)
  - intent_class is in {refactor, fix, content_update, quality_improvement,
                        rename, expand, simplify, drift_update}
    (excludes: whitespace, merge, micro, revert)

Outputs:
  data/release/curator_training.parquet
    columns: pair_id, skill_id, repo, platform, skill_name,
             before_content, after_content,
             intent_text (commit_subject or pr_title — whichever is more substantive),
             pr_title, pr_body, commit_subject,
             intent_class, lines_added, lines_removed, char_delta,
             license_spdx, stars, source_seed
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


RELEASE_DIR = Path("data/release")

# Intent classes worth training on (curator-relevant edits)
CURATOR_INTENT_CLASSES = {
    "refactor", "fix", "content_update", "quality_improvement",
    "rename", "expand", "simplify", "drift_update", "format_change",
    "documentation", "feature_add", "polish",
    # be inclusive about classifier output — better recall for v0.4
}

# Disqualifying quality tags for the default (non-strict) curator subset
DISQUALIFYING_TAGS = {
    "bot_author", "whitespace_change", "merge_commit", "revert_subject",
    "pre_revert", "duplicate_pair", "micro_edit", "short_skill",
    "invalid_frontmatter", "same_author_dup",
}

# Additional tags excluded by --strict (added in v0.4.2)
STRICT_DISQUALIFYING_TAGS = {
    "no_license",          # repo lacks SPDX license — redistribution risk
    "low_engagement",      # 0 stars + no license + no recent push
    "placeholder_content", # <your X here>, TODO: fill, lorem ipsum, etc
    "pii_email",           # contains real-looking email addresses (not allowlist)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-non-canonical", action="store_true",
                        help="Include non-canonical skills (default: drop)")
    parser.add_argument("--strict", action="store_true",
                        help="Apply stricter quality filters: no_license, "
                             "low_engagement, placeholder_content, pii_email. "
                             "Default output path becomes curator_training_strict.parquet")
    args = parser.parse_args()

    disq = set(DISQUALIFYING_TAGS)
    if args.strict:
        disq |= STRICT_DISQUALIFYING_TAGS

    rdir = Path(args.release_dir)
    diffs_path = rdir / "diffs.parquet"
    repos_path = rdir / "repos.parquet"

    if not diffs_path.exists():
        print(f"ERROR: {diffs_path} not found. Run consolidate_v04 + enrich_v03 first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading diffs.parquet...", file=sys.stderr)
    t = pq.read_table(diffs_path)
    n_total = t.num_rows
    print(f"  {n_total:,} total records", file=sys.stderr)

    # Filter: not initial (we want before→after pairs)
    is_init = t["is_initial"].to_pylist()
    keep_init = pa.array([not x for x in is_init])
    t = t.filter(keep_init)
    print(f"  -> {t.num_rows:,} after dropping initial commits", file=sys.stderr)

    # Filter: canonical only (unless flag set)
    if not args.include_non_canonical and "is_canonical" in t.schema.names:
        t = t.filter(pc.field("is_canonical"))
        print(f"  -> {t.num_rows:,} after canonical-only", file=sys.stderr)

    # Filter: no disqualifying quality tags
    quality_tags = t["quality_tags"].to_pylist()
    keep = [
        not (set(tags or []) & disq)
        for tags in quality_tags
    ]
    t = t.filter(pa.array(keep))
    label = "strict-disqualifying" if args.strict else "disqualifying"
    print(f"  -> {t.num_rows:,} after {label}-tag filter "
          f"({len(disq)} tags)", file=sys.stderr)

    # Filter: substantive intent_class
    intent_classes = t["intent_class"].to_pylist()
    keep = [
        ic in CURATOR_INTENT_CLASSES if ic else True  # accept null for now
        for ic in intent_classes
    ]
    # Actually be more permissive — only EXCLUDE explicit anti-classes
    keep = [
        ic not in {"whitespace", "merge", "micro"} for ic in intent_classes
    ]
    t = t.filter(pa.array(keep))
    print(f"  -> {t.num_rows:,} after intent_class filter", file=sys.stderr)

    # Filter: substantive intent text (need a non-trivial label)
    commit_subjects = t["commit_subject"].to_pylist()
    pr_titles = (
        t["pr_title"].to_pylist() if "pr_title" in t.schema.names
        else [None] * t.num_rows
    )
    intent_texts = []
    for cs, pt in zip(commit_subjects, pr_titles):
        # Prefer PR title if it's longer / non-trivial
        cs_clean = (cs or "").strip()
        pt_clean = (pt or "").strip()
        # PR title trumps commit subject if available
        chosen = pt_clean if pt_clean else cs_clean
        intent_texts.append(chosen)

    # Keep records whose intent text is non-trivial
    NONTRIVIAL_MIN_LEN = 8  # "fix typo" is OK; "fix" alone is not
    GENERIC_LABELS = {
        "wip", "update", "fix", "fixes", "edit", "stuff", "changes",
        "misc", "various", "minor", "tweak", "tweaks", ".", "..", "wip.",
    }
    keep = [
        len(t) >= NONTRIVIAL_MIN_LEN and t.lower() not in GENERIC_LABELS
        for t in intent_texts
    ]
    intent_arr = pa.array(intent_texts, type=pa.string())
    t = t.append_column("intent_text", intent_arr)
    t = t.filter(pa.array(keep))
    print(f"  -> {t.num_rows:,} after substantive-intent filter", file=sys.stderr)

    # Join with repos.parquet for license, stars
    if repos_path.exists():
        rt = pq.read_table(repos_path)
        repo_to_meta = {}
        for row in rt.to_pylist():
            repo_to_meta[row["repo"]] = {
                "license_spdx": row.get("license_spdx"),
                "stars": row.get("stars"),
            }
        repos = t["repo"].to_pylist()
        license_spdx = [repo_to_meta.get(r, {}).get("license_spdx") for r in repos]
        stars = [repo_to_meta.get(r, {}).get("stars") for r in repos]
        if "license_spdx" not in t.schema.names:
            t = t.append_column("license_spdx", pa.array(license_spdx, type=pa.string()))
            t = t.append_column("stars", pa.array(stars, type=pa.int32()))

    # Select desired output columns
    out_cols = [
        "pair_id", "skill_id", "repo",
        "skill_path", "skill_name",
        "before_content", "after_content",
        "intent_text",
        "commit_subject",
    ]
    # Optional columns (present after consolidate_v04 + enrich_v03 + join_pr_metadata + add_licenses + v0.5 enrichments)
    optional_cols = [
        "platform", "pr_number", "pr_title", "pr_body", "pr_state", "pr_match_kind",
        "intent_class", "intent_confidence", "intent_source",
        "lines_added", "lines_removed", "char_delta",
        "skill_cluster_id", "is_canonical",
        "skill_semantic_cluster_id", "is_semantic_canonical",  # v0.5
        "diff_summary",                                          # v0.5
        "quality_score",                                         # v0.5
        "license_spdx", "stars", "source_seed", "quality_tags",
    ]
    for c in optional_cols:
        if c in t.schema.names:
            out_cols.append(c)

    # Filter to existing columns only
    out_cols = [c for c in out_cols if c in t.schema.names]
    out = t.select(out_cols)

    if args.out:
        out_path = Path(args.out)
    elif args.strict:
        out_path = rdir / "curator_training_strict.parquet"
    else:
        out_path = rdir / "curator_training.parquet"
    pq.write_table(out, out_path, compression="zstd")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(f"  rows: {out.num_rows:,}", file=sys.stderr)
    print(f"  size: {size_mb:.1f} MB", file=sys.stderr)
    print(f"  cols: {len(out.schema.names)}", file=sys.stderr)

    # Print a sample
    if out.num_rows > 0:
        print("\nSample row:", file=sys.stderr)
        sample = out.slice(0, 1).to_pylist()[0]
        for k in ["repo", "skill_name", "intent_text", "intent_class",
                  "lines_added", "lines_removed"]:
            v = sample.get(k)
            if v and isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
