#!/usr/bin/env python3
"""Add four new quality_tags to existing release parquets:

  - no_license           repo has no SPDX license_spdx
  - low_engagement       repo has 0 stars AND no license AND no recent push
                         (within 12 months of fetched_at)
  - placeholder_content  after_content matches placeholder/test patterns
                         (<your X here>, TODO: fill, lorem ipsum, hello world)
  - pii_email            after_content contains email addresses that don't
                         match common documentation patterns (example.com,
                         noreply@github.com, etc.)

Idempotent: if a tag is already in quality_tags, leaves it alone. If a tag
no longer applies (rare — would require external data change), removes it.

Reads + writes:
    data/release/diffs.parquet
    data/release/diffs_clean.parquet
    data/release/skills_initial.parquet
    data/release/curator_training.parquet  (regenerated separately by curator_subset.py)
"""
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


RELEASE_DIR = Path("data/release")
TARGETS = ["diffs.parquet", "diffs_clean.parquet", "skills_initial.parquet"]

NEW_TAGS = {"no_license", "low_engagement", "placeholder_content", "pii_email"}

# Placeholder patterns
PLACEHOLDER_PATTERNS = [
    re.compile(r"<your\s+\w+\s+here>", re.I),
    re.compile(r"<your[-\s_][^>]{0,30}>", re.I),
    re.compile(r"\bTODO:\s*(?:fill|add|write|complete|implement)", re.I),
    re.compile(r"\blorem\s+ipsum\b", re.I),
    re.compile(r"\bplaceholder\s+(?:text|content)\b", re.I),
    re.compile(r"\bhello,?\s*world\b", re.I),
    re.compile(r"\bfoo\s*bar\s*baz\b", re.I),
]

# Email pattern + allowlist (skip these — they're documentation, not PII)
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
EMAIL_ALLOWLIST = {
    "example.com", "example.org", "example.net",
    "users.noreply.github.com",  # GitHub privacy emails
    "noreply.github.com",
    "domain.com", "yourdomain.com",  # template
    "test.com",  # template
    "local",  # localhost addrs
}


def is_pii_email(email):
    """True if email looks like real PII (not a documentation example)."""
    domain = email.split("@", 1)[-1].lower()
    if domain in EMAIL_ALLOWLIST:
        return False
    if domain.endswith(".example") or domain.endswith(".local") or domain.endswith(".test"):
        return False
    if domain.startswith("noreply") or "noreply" in domain:
        return False
    return True


def has_pii_email(content):
    if not content:
        return False
    for m in EMAIL_RE.finditer(content):
        if is_pii_email(m.group()):
            return True
    return False


def has_placeholder(content):
    if not content:
        return False
    return any(pat.search(content) for pat in PLACEHOLDER_PATTERNS)


def main():
    rdir = Path(RELEASE_DIR)

    # Build per-repo quality signals from repos.parquet
    print("Loading repos.parquet for repo-level signals...", file=sys.stderr)
    repos_t = pq.read_table(
        rdir / "repos.parquet",
        columns=["repo", "stars", "license_spdx", "pushed_at", "fetched_at"],
    )

    repo_meta = {}
    now_iso = datetime.now(timezone.utc).isoformat()
    twelve_months_ago = datetime.now(timezone.utc).replace(
        year=datetime.now(timezone.utc).year - 1
    ).isoformat()

    for row in repos_t.to_pylist():
        stars = row.get("stars")
        license_spdx = row.get("license_spdx")
        pushed_at = row.get("pushed_at") or ""
        recent_push = pushed_at >= twelve_months_ago if pushed_at else False
        repo_meta[row["repo"]] = {
            "no_license": not license_spdx,
            "low_engagement": (
                (stars is None or stars == 0)
                and not license_spdx
                and not recent_push
            ),
        }

    # Stats
    n_no_lic = sum(1 for m in repo_meta.values() if m["no_license"])
    n_low_eng = sum(1 for m in repo_meta.values() if m["low_engagement"])
    print(f"  {len(repo_meta):,} repos: {n_no_lic:,} no_license  "
          f"{n_low_eng:,} low_engagement", file=sys.stderr)

    # Process each target parquet
    for fname in TARGETS:
        p = rdir / fname
        if not p.exists():
            continue
        print(f"\n=== {fname} ===", file=sys.stderr)
        t = pq.read_table(p)
        n = t.num_rows
        print(f"  rows: {n:,}", file=sys.stderr)

        rows = t.to_pylist()
        n_added = {tag: 0 for tag in NEW_TAGS}
        started = time.time()

        for i, r in enumerate(rows):
            existing = set(r.get("quality_tags") or [])
            # Drop the new tags first (idempotent)
            existing -= NEW_TAGS

            repo_sig = repo_meta.get(r["repo"], {})
            if repo_sig.get("no_license"):
                existing.add("no_license")
                n_added["no_license"] += 1
            if repo_sig.get("low_engagement"):
                existing.add("low_engagement")
                n_added["low_engagement"] += 1

            after = r.get("after_content") or ""
            if has_placeholder(after):
                existing.add("placeholder_content")
                n_added["placeholder_content"] += 1
            if has_pii_email(after):
                existing.add("pii_email")
                n_added["pii_email"] += 1

            r["quality_tags"] = sorted(existing)

            if (i + 1) % 200_000 == 0:
                print(f"    [{i+1:,}/{n:,}] ({time.time()-started:.0f}s)", file=sys.stderr)

        for tag, c in n_added.items():
            print(f"  {tag:<22} {c:>10,}  ({100*c/n:.1f}%)", file=sys.stderr)

        # Write back preserving schema
        out_tmp = p.with_suffix(".tmp.parquet")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=t.schema),
            out_tmp, compression="zstd",
        )
        out_tmp.replace(p)
        print(f"  wrote {p}", file=sys.stderr)

    print("\nDone. Re-run curator_subset.py (with --strict for the strict variant).",
          file=sys.stderr)


if __name__ == "__main__":
    main()
