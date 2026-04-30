#!/usr/bin/env python3
"""Analyze batch output: count diff pairs, distributions, dedupe estimates."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw")


def iter_records():
    """Yield every record from every shard."""
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    n_total = 0
    n_initial = 0
    n_diff_pairs = 0

    skill_keys = set()  # (repo, skill_path)
    skill_versions = defaultdict(int)
    after_content_hashes = set()  # content-level dedup signal
    repos_seen = set()

    lines_added_dist = Counter()
    char_delta_dist = Counter()
    commits_per_skill = []

    for r in iter_records():
        n_total += 1
        repos_seen.add(r["repo"])
        skill_key = (r["repo"], r["skill_path"])
        skill_keys.add(skill_key)
        skill_versions[skill_key] += 1

        if r["is_initial"]:
            n_initial += 1
        else:
            n_diff_pairs += 1

        # Bucket lines added (10s)
        added = r["lines_added"]
        if added == 0:
            lines_added_dist["0"] += 1
        elif added <= 5:
            lines_added_dist["1-5"] += 1
        elif added <= 20:
            lines_added_dist["6-20"] += 1
        elif added <= 100:
            lines_added_dist["21-100"] += 1
        else:
            lines_added_dist["101+"] += 1

        # Content hash for dedup analysis
        h = hashlib.sha256(r["after_content"].encode("utf-8", "replace")).hexdigest()
        after_content_hashes.add(h)

    # Group commits per skill
    for n in skill_versions.values():
        commits_per_skill.append(n)

    bucket = Counter()
    for n in commits_per_skill:
        if n == 1:
            bucket["1"] += 1
        elif n <= 3:
            bucket["2-3"] += 1
        elif n <= 5:
            bucket["4-5"] += 1
        elif n <= 10:
            bucket["6-10"] += 1
        else:
            bucket["11+"] += 1

    print(f"Total records (commit_pair, skill_file): {n_total:,}")
    print(f"  Initial commits (no 'before'): {n_initial:,}")
    print(f"  True diff pairs:                {n_diff_pairs:,}")
    print()
    print(f"Unique (repo, skill_path) pairs: {len(skill_keys):,}")
    print(f"Unique repos with at least 1 skill: {len(repos_seen):,}")
    print(f"Unique after_content hashes: {len(after_content_hashes):,}")
    print(f"  (vs {n_total:,} records => "
          f"{100*(1 - len(after_content_hashes)/max(n_total,1)):.1f}% near-duplicate content)")
    print()
    print("Commits per skill distribution:")
    for k in ["1", "2-3", "4-5", "6-10", "11+"]:
        print(f"  {k:>6}: {bucket[k]:>6,} skills")
    print()
    print("Lines-added per record distribution:")
    for k in ["0", "1-5", "6-20", "21-100", "101+"]:
        print(f"  {k:>6}: {lines_added_dist[k]:>6,} records")


if __name__ == "__main__":
    main()
