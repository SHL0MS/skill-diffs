#!/usr/bin/env python3
"""Classify each diff record by commit-message intent.

Two-stage:
  1. Regex / heuristic classifier — catches conventional-commit prefixes,
     reverts, merges, initial commits, etc. Free, deterministic.
  2. (Optional) LLM classifier for records the regex couldn't classify with
     confidence. Disabled by default — pass --use-llm to enable.

Output:
  data/classified/<orig_filename>.jsonl  with added fields:
    intent_class:   feat|fix|docs|style|refactor|perf|test|build|ci|chore|
                    revert|merge|initial|whitespace|reformat|unknown
    intent_source:  regex|llm
    intent_confidence: float (0..1)
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Conventional Commits prefix
CC_RE = re.compile(
    r"^\s*(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([^)]+\))?\s*[:\-!]\s*",
    re.IGNORECASE,
)

# Common informal patterns mapped to canonical class
INFORMAL_PATTERNS = [
    (re.compile(r"^\s*revert", re.I), "revert"),
    (re.compile(r"^\s*merge\b", re.I), "merge"),
    (re.compile(r"\binit(ial)?(\s+commit)?\b", re.I), "initial"),
    (re.compile(r"\bfirst\s+commit\b", re.I), "initial"),
    (re.compile(r"\b(typo|spelling)\b", re.I), "docs"),
    (re.compile(r"\b(format|prettier|whitespace|lint)\b", re.I), "style"),
    (re.compile(r"\b(rename|move|extract|inline)\b", re.I), "refactor"),
    (re.compile(r"\b(bump|update\s+(deps?|dependencies))\b", re.I), "build"),
    (re.compile(r"\b(readme|docs|comment)\b", re.I), "docs"),
    (re.compile(r"\b(fix|bug|issue|error)\b", re.I), "fix"),
    (re.compile(r"\b(add|new|implement|introduce|feature)\b", re.I), "feat"),
    (re.compile(r"\b(improve|enhance|refine|polish|clean)\b", re.I), "refactor"),
    (re.compile(r"\bwip\b", re.I), "chore"),
]


def classify_subject(subject):
    """Return (class, confidence, source) from a commit subject line."""
    if not subject:
        return ("unknown", 0.0, "regex")

    # Stage 1: conventional commits
    m = CC_RE.match(subject)
    if m:
        return (m.group("type").lower(), 0.95, "regex")

    # Stage 2: informal patterns (first match wins; ordered most-specific first)
    for pat, klass in INFORMAL_PATTERNS:
        if pat.search(subject):
            return (klass, 0.65, "regex")

    return ("unknown", 0.0, "regex")


def classify_record(rec):
    """Classify one diff record. Augments with intent_* fields."""
    subj = rec.get("commit_subject", "") or ""
    is_initial = rec.get("is_initial", False)

    # Special-cases that override message-based classification:
    if is_initial and rec.get("lines_removed", 0) == 0:
        # Initial creation of the file
        if not subj or "init" in subj.lower() or "first" in subj.lower():
            return ("initial", 0.95, "regex")

    klass, conf, source = classify_subject(subj)

    # Whitespace / no-op detection from numstat
    added = rec.get("lines_added", 0)
    removed = rec.get("lines_removed", 0)
    delta_chars = abs(rec.get("char_delta", 0))
    if added == 0 and removed == 0:
        return ("whitespace", 0.9, "regex")
    if added <= 1 and removed <= 1 and delta_chars < 30 and klass == "unknown":
        return ("whitespace", 0.6, "regex")

    return (klass, conf, source)


def process_file(in_path, out_path):
    n = n_classified = 0
    by_class = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            klass, conf, source = classify_record(rec)
            rec["intent_class"] = klass
            rec["intent_confidence"] = round(conf, 2)
            rec["intent_source"] = source
            if klass != "unknown":
                n_classified += 1
            by_class[klass] = by_class.get(klass, 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n, n_classified, by_class


def main():
    parser = argparse.ArgumentParser(description="Classify diff records by intent.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/classified")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    files = sorted(in_dir.glob("*.jsonl"))
    print(f"Classifying {len(files)} file(s) from {in_dir}", file=sys.stderr)

    total_n = 0
    total_classified = 0
    grand_by_class = {}
    for f in files:
        out = out_dir / f.name
        n, nc, by_class = process_file(f, out)
        total_n += n
        total_classified += nc
        for k, v in by_class.items():
            grand_by_class[k] = grand_by_class.get(k, 0) + v

    print(f"\nProcessed {total_n:,} records. Classified: {total_classified:,} "
          f"({100*total_classified/max(total_n,1):.1f}%)", file=sys.stderr)
    print("\nClass distribution:", file=sys.stderr)
    for k, v in sorted(grand_by_class.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v:>8,} ({100*v/max(total_n,1):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
