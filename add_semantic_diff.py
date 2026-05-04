#!/usr/bin/env python3
"""Add a structured `diff_summary` column to release parquets.

For each (before_content, after_content) pair, compute:
  diff_summary = {
    frontmatter_changed: bool,
    frontmatter_name_changed: bool,
    frontmatter_description_changed: bool,
    body_added_chars: int,
    body_removed_chars: int,
    code_blocks_before: int,
    code_blocks_after: int,
    sections_added: list<str>,        (H1/H2 headings present in after but not before)
    sections_removed: list<str>,
    edit_kind: str,                   ("frontmatter_only" / "body_only" / "both" / "code_only" /
                                       "structural" / "trivial" / "addition" / "deletion")
  }

This is a flat, structural diff — not a semantic understanding. It's a quick way
to filter for "frontmatter-only fixes" or "edits that add a new section" etc.

Idempotent: drops existing diff_summary column if present before re-adding.

Reads + writes:
    data/release/diffs.parquet
    data/release/diffs_clean.parquet
    data/release/curator_training.parquet
    data/release/curator_training_strict.parquet
"""
import argparse
import re
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

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
NAME_RE = re.compile(r"^name\s*:\s*(.+?)\s*$", re.MULTILINE)
DESC_RE = re.compile(r"^description\s*:\s*(.+?)(?=\n\w|\n---|\Z)", re.MULTILINE | re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)
CODEFENCE_RE = re.compile(r"^```", re.MULTILINE)


def parse_skill(content):
    """Returns dict: {frontmatter, body, name, description, headings, n_code_blocks}."""
    if not content:
        return {
            "frontmatter": "", "body": "",
            "name": "", "description": "",
            "headings": [],
            "n_code_blocks": 0,
        }
    m = FRONTMATTER_RE.match(content)
    if m:
        fm = m.group(1)
        body = content[m.end():]
    else:
        fm = ""
        body = content

    name_m = NAME_RE.search(fm) if fm else None
    desc_m = DESC_RE.search(fm) if fm else None
    name = name_m.group(1).strip() if name_m else ""
    description = desc_m.group(1).strip() if desc_m else ""

    headings = [m.group(2).strip() for m in HEADING_RE.finditer(body)]
    # Code fences come in pairs (open + close); count as half
    n_fences = len(CODEFENCE_RE.findall(body))
    n_code_blocks = n_fences // 2

    return {
        "frontmatter": fm.strip(),
        "body": body.strip(),
        "name": name,
        "description": description,
        "headings": headings,
        "n_code_blocks": n_code_blocks,
    }


def diff_summary(before_content, after_content):
    b = parse_skill(before_content or "")
    a = parse_skill(after_content or "")

    # Frontmatter-level changes
    fm_changed = b["frontmatter"] != a["frontmatter"]
    name_changed = b["name"] != a["name"]
    desc_changed = b["description"] != a["description"]

    # Body-level deltas (char counts)
    body_b = b["body"]
    body_a = a["body"]
    body_changed = body_b != body_a
    body_added_chars = max(0, len(body_a) - len(body_b))
    body_removed_chars = max(0, len(body_b) - len(body_a))

    # Headings: which sections added/removed
    b_headings = set(b["headings"])
    a_headings = set(a["headings"])
    sections_added = sorted(a_headings - b_headings)
    sections_removed = sorted(b_headings - a_headings)

    # Code blocks
    code_blocks_before = b["n_code_blocks"]
    code_blocks_after = a["n_code_blocks"]

    # Edit kind (categorical heuristic)
    if not before_content and after_content:
        edit_kind = "addition"          # initial commit
    elif before_content and not after_content:
        edit_kind = "deletion"
    elif not (fm_changed or body_changed):
        edit_kind = "trivial"
    elif fm_changed and not body_changed:
        edit_kind = "frontmatter_only"
    elif body_changed and not fm_changed:
        if sections_added or sections_removed:
            edit_kind = "structural"    # added/removed sections
        elif code_blocks_after != code_blocks_before:
            edit_kind = "code_only"
        else:
            edit_kind = "body_only"
    else:
        edit_kind = "both"              # frontmatter and body both changed

    return {
        "frontmatter_changed": fm_changed,
        "frontmatter_name_changed": name_changed,
        "frontmatter_description_changed": desc_changed,
        "body_added_chars": body_added_chars,
        "body_removed_chars": body_removed_chars,
        "code_blocks_before": code_blocks_before,
        "code_blocks_after": code_blocks_after,
        "sections_added": sections_added,
        "sections_removed": sections_removed,
        "edit_kind": edit_kind,
    }


def diff_summary_struct():
    return pa.struct([
        ("frontmatter_changed", pa.bool_()),
        ("frontmatter_name_changed", pa.bool_()),
        ("frontmatter_description_changed", pa.bool_()),
        ("body_added_chars", pa.int32()),
        ("body_removed_chars", pa.int32()),
        ("code_blocks_before", pa.int32()),
        ("code_blocks_after", pa.int32()),
        ("sections_added", pa.list_(pa.string())),
        ("sections_removed", pa.list_(pa.string())),
        ("edit_kind", pa.string()),
    ])


def process(p):
    print(f"\n=== {p.name} ===", file=sys.stderr)
    t = pq.read_table(p)
    n = t.num_rows
    print(f"  rows: {n:,}", file=sys.stderr)

    before_col = t["before_content"].to_pylist()
    after_col = t["after_content"].to_pylist()

    started = time.time()
    summaries = []
    for i, (b, a) in enumerate(zip(before_col, after_col)):
        summaries.append(diff_summary(b, a))
        if (i + 1) % 100_000 == 0:
            print(f"  [{i+1:,}/{n:,}] ({time.time()-started:.0f}s)", file=sys.stderr)

    arr = pa.array(summaries, type=diff_summary_struct())

    # Drop existing column if present (idempotent), then add new one
    if "diff_summary" in t.schema.names:
        idx = t.schema.get_field_index("diff_summary")
        new_t = t.set_column(idx, pa.field("diff_summary", arr.type), arr)
    else:
        new_t = t.append_column("diff_summary", arr)

    out_tmp = p.with_suffix(".tmp.parquet")
    pq.write_table(new_t, out_tmp, compression="zstd")
    out_tmp.replace(p)

    # Stats
    from collections import Counter
    edit_kinds = Counter(s["edit_kind"] for s in summaries)
    print(f"  edit_kind distribution:", file=sys.stderr)
    for k, c in edit_kinds.most_common():
        print(f"    {k:<20} {c:>10,}  ({100*c/n:.1f}%)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    parser.add_argument("--targets", nargs="*", default=TARGETS)
    args = parser.parse_args()

    rdir = Path(args.release_dir)
    for fname in args.targets:
        p = rdir / fname
        if not p.exists():
            print(f"  (skip missing: {p})", file=sys.stderr)
            continue
        process(p)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
