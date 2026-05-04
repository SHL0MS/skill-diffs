#!/usr/bin/env python3
"""Flag records that contain prompt-injection-style language.

Adds `prompt_injection_pattern` to quality_tags. ADVISORY (not in strict-
disqualifying set) because most matches are defensive content (security
skills teaching about injection patterns). Downstream consumers can choose
to filter on it.

Patterns flagged (regex, case-insensitive):
  - "ignore (all )?previous (instructions|messages|prompts)"
  - "disregard (all )?prior"
  - "you are now [a-z]+"  (role hijack)
  - "developer mode"
  - "jailbroken"
  - "DAN mode" (do anything now)

Idempotent: removes existing prompt_injection_pattern tag before re-evaluating.
"""
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
]

INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all )?previous (?:instructions?|messages?|prompts?|context)", re.I),
    re.compile(r"disregard (?:all )?(?:prior|previous|earlier)", re.I),
    re.compile(r"\byou are now (?:[a-z][a-z0-9_]+|in [a-z]+ mode)", re.I),
    re.compile(r"\bdeveloper mode\b", re.I),
    re.compile(r"\bjailbroken?\b", re.I),
    re.compile(r"\bDAN(?:[ -]mode)?\b"),  # case sensitive
    re.compile(r"\boverride (?:your |the )?(?:safety|guidelines|restrictions?)", re.I),
]

TAG = "prompt_injection_pattern"


def has_injection(content):
    if not content:
        return False
    for pat in INJECTION_PATTERNS:
        if pat.search(content):
            return True
    return False


def process(p):
    print(f"\n=== {p.name} ===", file=sys.stderr)
    t = pq.read_table(p)
    n = t.num_rows
    print(f"  rows: {n:,}", file=sys.stderr)

    rows = t.to_pylist()
    started = time.time()
    n_flagged = 0
    for i, r in enumerate(rows):
        existing = set(r.get("quality_tags") or [])
        existing.discard(TAG)  # idempotent
        if has_injection(r.get("after_content") or ""):
            existing.add(TAG)
            n_flagged += 1
        r["quality_tags"] = sorted(existing)
        if (i + 1) % 200_000 == 0:
            print(f"  [{i+1:,}/{n:,}] flagged so far: {n_flagged:,} "
                  f"({time.time()-started:.0f}s)", file=sys.stderr)

    print(f"  flagged: {n_flagged:,} ({100*n_flagged/n:.2f}%)", file=sys.stderr)

    out_tmp = p.with_suffix(".tmp.parquet")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=t.schema),
        out_tmp, compression="zstd",
    )
    out_tmp.replace(p)


def main():
    for fname in TARGETS:
        p = RELEASE_DIR / fname
        if not p.exists():
            continue
        process(p)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
