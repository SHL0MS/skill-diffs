#!/usr/bin/env python3
"""Tag each record with quality flags. Does not delete — adds `quality_tags`
list so downstream consumers (e.g. training pipelines) can choose their own
filter level.

Quality tag taxonomy:
  bot_author              committer is a known bot account
  whitespace_change       intent_class == "whitespace" or numstat indicates noop
  revert_subject          commit subject starts with "Revert "
  pre_revert              this commit is immediately reverted by next commit
  merge_commit            intent_class == "merge"
  initial_commit          first commit for the file (is_initial)
  duplicate_after         after_content already seen in another record
  duplicate_pair          (before_content, after_content) pair already seen
  short_skill             after_content length < 500 chars
  micro_edit              non-initial diff with <=2 lines added/removed and <40 char delta
  large_blob              before_content or after_content > 200 KB (rare)
  non_utf8_clean          content has replacement chars from decode errors

A "clean" subset is also written, dropping records with any of:
  bot_author, whitespace_change, revert_subject, pre_revert, merge_commit,
  duplicate_pair, micro_edit, short_skill (configurable).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


BOT_EMAIL_PATTERNS = [
    re.compile(r".*\[bot\]@.*"),
    re.compile(r"^(noreply|action)@github\.com$"),
    re.compile(r".*\bdependabot\b.*", re.I),
    re.compile(r".*\brenovate\b.*", re.I),
    re.compile(r".*\bgithub-actions\b.*", re.I),
    re.compile(r".*\bsemantic-release\b.*", re.I),
]

# Tags that disqualify a record from the "clean" subset
DEFAULT_DISQUALIFYING = {
    "bot_author",
    "whitespace_change",
    "revert_subject",
    "pre_revert",
    "merge_commit",
    "duplicate_pair",
    "micro_edit",
    "short_skill",
}


def is_bot_email(email):
    if not email:
        return False
    for pat in BOT_EMAIL_PATTERNS:
        if pat.match(email):
            return True
    return False


def content_hash(s):
    if s is None:
        return None
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def pair_hash(before, after):
    h = hashlib.sha256()
    h.update((before or "").encode("utf-8", errors="replace"))
    h.update(b"\x00\x01\x02")
    h.update((after or "").encode("utf-8", errors="replace"))
    return h.hexdigest()


def load_all_records(in_dir):
    """Load all records, grouped by file (for chronological pre-revert detection)."""
    files = sorted(Path(in_dir).glob("*.jsonl"))
    for f in files:
        records = []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        yield f, records


def detect_pre_reverts(records):
    """Find indices that should be tagged pre_revert.

    Heuristic: within the same skill_path, if commit N+1's subject starts
    with 'Revert' and references commit N's subject, then commit N is a
    pre-revert state.
    """
    flags = set()
    by_skill = defaultdict(list)
    for i, r in enumerate(records):
        by_skill[(r["repo"], r["skill_path"])].append((i, r))
    for chain in by_skill.values():
        # Already in chronological order from extract.py output
        for i in range(len(chain) - 1):
            idx_curr, curr = chain[i]
            idx_next, nxt = chain[i + 1]
            nxt_subj = (nxt.get("commit_subject") or "").strip()
            curr_subj = (curr.get("commit_subject") or "").strip()
            if nxt_subj.lower().startswith("revert"):
                # Heuristic: revert refers to current commit if subject quoted
                if curr_subj and curr_subj.lower() in nxt_subj.lower():
                    flags.add(idx_curr)
                else:
                    # Conservative: still flag the commit just before any revert
                    flags.add(idx_curr)
    return flags


def classify_one(rec, after_hash_seen, pair_hash_seen):
    tags = []
    # Bot
    if is_bot_email(rec.get("commit_email")):
        tags.append("bot_author")
    # Intent-derived
    ic = rec.get("intent_class")
    if ic == "whitespace":
        tags.append("whitespace_change")
    if ic == "merge":
        tags.append("merge_commit")
    # Subject-derived (independent of intent classifier)
    subj = (rec.get("commit_subject") or "").strip()
    if subj.lower().startswith("revert"):
        tags.append("revert_subject")
    # Initial commit
    if rec.get("is_initial"):
        tags.append("initial_commit")
    # Sizes
    after = rec.get("after_content") or ""
    before = rec.get("before_content") or ""
    if len(after) < 500 and not rec.get("is_initial"):
        tags.append("short_skill")
    if len(before) > 200_000 or len(after) > 200_000:
        tags.append("large_blob")
    # UTF-8 replacement chars
    if "\ufffd" in after or "\ufffd" in before:
        tags.append("non_utf8_clean")
    # Micro-edit
    added = rec.get("lines_added", 0) or 0
    removed = rec.get("lines_removed", 0) or 0
    char_delta = abs(rec.get("char_delta", 0) or 0)
    if not rec.get("is_initial") and added <= 2 and removed <= 2 and char_delta < 40:
        tags.append("micro_edit")
    # Duplicate detection
    ah = content_hash(after)
    if ah in after_hash_seen:
        tags.append("duplicate_after")
    after_hash_seen.add(ah)
    ph = pair_hash(before, after)
    if ph in pair_hash_seen:
        tags.append("duplicate_pair")
    pair_hash_seen.add(ph)
    return tags


def main():
    parser = argparse.ArgumentParser(description="Tag records with quality flags.")
    parser.add_argument("--input-dir", default="data/classified")
    parser.add_argument("--output-dir", default="data/filtered")
    parser.add_argument("--clean-output-dir", default="data/clean")
    parser.add_argument("--disqualifying", nargs="+",
                        default=list(DEFAULT_DISQUALIFYING),
                        help="Tags that exclude record from the clean subset")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    clean_dir = Path(args.clean_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    disq = set(args.disqualifying)

    # Global dedup state (across all files)
    after_hash_seen = set()
    pair_hash_seen = set()

    n_total = n_clean = 0
    tag_counts = defaultdict(int)

    for f, records in load_all_records(args.input_dir):
        pre_revert_idx = detect_pre_reverts(records)
        out_path = out_dir / f.name
        clean_path = clean_dir / f.name
        with open(out_path, "w") as fout, open(clean_path, "w") as fclean:
            for i, rec in enumerate(records):
                tags = classify_one(rec, after_hash_seen, pair_hash_seen)
                if i in pre_revert_idx:
                    tags.append("pre_revert")
                rec["quality_tags"] = tags
                for t in tags:
                    tag_counts[t] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if not (set(tags) & disq):
                    fclean.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_clean += 1
                n_total += 1

    print(f"Processed: {n_total:,} records", file=sys.stderr)
    print(f"Clean (no disqualifying tag): {n_clean:,} ({100*n_clean/max(n_total,1):.1f}%)",
          file=sys.stderr)
    print("\nTag counts:", file=sys.stderr)
    for tag, n in sorted(tag_counts.items(), key=lambda x: -x[1]):
        marker = "X" if tag in disq else " "
        print(f"  [{marker}] {tag:<22} {n:>8,} ({100*n/max(n_total,1):.1f}%)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
