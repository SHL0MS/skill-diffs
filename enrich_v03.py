#!/usr/bin/env python3
"""v0.3 enrichment passes over the parquet dataset:

  1. MinHash near-duplicate clustering on after_content (skill-level)
     -> skill_cluster_id (string), is_canonical (bool)
  2. Frontmatter validation
     -> adds 'invalid_frontmatter' to quality_tags when YAML name+description missing
  3. Same-author duplicate detection
     -> adds 'same_author_dup' to quality_tags when (email, after_content) already seen

Reads/writes data/release/{diffs,diffs_clean,skills_initial}.parquet in place
with new columns added.
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH


RELEASE_DIR = Path("data/release")
TARGET_FILES = ["diffs.parquet", "diffs_clean.parquet", "skills_initial.parquet"]

# Frontmatter detection
FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
NAME_RE = re.compile(r"^name\s*:\s*\S+", re.MULTILINE)
DESC_RE = re.compile(r"^description\s*:\s*\S", re.MULTILINE)

# MinHash params
NUM_PERM = 128
LSH_THRESHOLD = 0.7
SHINGLE_SIZE = 5  # 5-token shingles


def has_valid_frontmatter(content):
    if not content:
        return False
    m = FRONTMATTER_RE.match(content)
    if not m:
        return False
    fm = m.group(1)
    return bool(NAME_RE.search(fm)) and bool(DESC_RE.search(fm))


def shingle_tokens(content, k=SHINGLE_SIZE):
    """Yield k-shingles of word tokens from content (after frontmatter)."""
    if not content:
        return
    # Strip frontmatter
    m = FRONTMATTER_RE.match(content)
    body = content[m.end():] if m else content
    # Lowercase + tokenize on whitespace
    tokens = body.lower().split()
    if len(tokens) < k:
        # For very short content, yield single tokens
        for t in tokens:
            yield t
        return
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i:i + k])


def make_minhash(content):
    m = MinHash(num_perm=NUM_PERM)
    for shingle in shingle_tokens(content):
        m.update(shingle.encode("utf-8", errors="replace"))
    return m


def stable_short_id(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def build_clusters(unique_skills):
    """Given list of (skill_id, after_content), build LSH and return:
       cluster_map: skill_id -> cluster_id (str)
       canonical_map: cluster_id -> canonical skill_id
    """
    print(f"  Building MinHashes for {len(unique_skills):,} unique skills...",
          file=sys.stderr)
    started = time.time()
    minhashes = {}
    for i, (sid, content) in enumerate(unique_skills):
        minhashes[sid] = make_minhash(content)
        if (i + 1) % 50000 == 0:
            print(f"    [{i+1:,}/{len(unique_skills):,}] "
                  f"({time.time() - started:.0f}s elapsed)", file=sys.stderr)

    print(f"  Inserting into LSH (threshold={LSH_THRESHOLD})...", file=sys.stderr)
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for sid, mh in minhashes.items():
        lsh.insert(sid, mh)

    print(f"  Querying LSH for clusters...", file=sys.stderr)
    # Union-find via simple cluster assignment
    cluster_of = {}
    cluster_members = {}  # cluster_id -> set of skill_ids
    next_cluster_idx = 0

    for sid, mh in minhashes.items():
        if sid in cluster_of:
            continue
        candidates = lsh.query(mh)
        # All candidates form one cluster
        # Pick existing cluster_id if any candidate already assigned
        existing = None
        for c in candidates:
            if c in cluster_of:
                existing = cluster_of[c]
                break
        if existing is None:
            cluster_id = f"c{next_cluster_idx:07d}"
            next_cluster_idx += 1
            cluster_members[cluster_id] = set()
        else:
            cluster_id = existing
        for c in candidates:
            if c not in cluster_of:
                cluster_of[c] = cluster_id
                cluster_members[cluster_id].add(c)

    # Pick canonical = smallest skill_id alphabetically (deterministic)
    canonical_map = {
        cid: min(members) for cid, members in cluster_members.items()
    }

    elapsed = time.time() - started
    n_clusters = len(cluster_members)
    n_singletons = sum(1 for m in cluster_members.values() if len(m) == 1)
    n_multi = n_clusters - n_singletons
    n_clustered = sum(len(m) for m in cluster_members.values() if len(m) > 1)
    print(f"  Clustering done in {elapsed:.0f}s.", file=sys.stderr)
    print(f"    Total clusters: {n_clusters:,}", file=sys.stderr)
    print(f"    Singletons:     {n_singletons:,}", file=sys.stderr)
    print(f"    Multi-member:   {n_multi:,} (covering {n_clustered:,} skills)",
          file=sys.stderr)
    return cluster_of, canonical_map


def main():
    parser = argparse.ArgumentParser(description="v0.3 enrichment.")
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    args = parser.parse_args()

    rdir = Path(args.release_dir)

    print("Loading skills_initial.parquet to get one (skill_id, after_content) per skill...",
          file=sys.stderr)
    initial_t = pq.read_table(
        rdir / "skills_initial.parquet",
        columns=["skill_id", "after_content"],
    )
    pairs = list(zip(
        initial_t["skill_id"].to_pylist(),
        initial_t["after_content"].to_pylist(),
    ))
    # If a skill_id appears twice (shouldn't, but safety), keep first
    seen = set()
    unique_pairs = []
    for sid, content in pairs:
        if sid in seen:
            continue
        seen.add(sid)
        unique_pairs.append((sid, content))
    print(f"  {len(unique_pairs):,} unique skills", file=sys.stderr)

    print("\n=== Pass 1: MinHash clustering ===", file=sys.stderr)
    cluster_of, canonical_map = build_clusters(unique_pairs)

    # For frontmatter validation, also use the initial (latest is captured per
    # diff but the initial is always available; downstream we check the
    # PER-RECORD after_content in the loop below)

    print("\n=== Pass 2: Same-author duplicate detection (global) ===", file=sys.stderr)
    # Build set of (commit_email, sha1(after_content)) seen in any record
    # We'll detect this by streaming through diffs.parquet.
    print("  Building author+content hash set across all records...", file=sys.stderr)
    author_content_first_seen = {}  # hash -> pair_id of first record
    # Stream diffs.parquet
    diffs_path = rdir / "diffs.parquet"
    t = pq.read_table(diffs_path,
                      columns=["pair_id", "commit_email", "after_content"])
    pair_ids = t["pair_id"].to_pylist()
    emails = t["commit_email"].to_pylist()
    afters = t["after_content"].to_pylist()
    same_author_dups = set()
    for pid, em, after in zip(pair_ids, emails, afters):
        if not em or not after:
            continue
        key = hashlib.sha1(
            (em + "\x00" + after).encode("utf-8", errors="replace")
        ).hexdigest()
        if key in author_content_first_seen:
            same_author_dups.add(pid)
        else:
            author_content_first_seen[key] = pid
    print(f"  {len(same_author_dups):,} records flagged as same-author-dup", file=sys.stderr)
    del author_content_first_seen, t

    # === Pass 3 + write: process each parquet file ===
    print("\n=== Pass 3: Frontmatter validation + writing enriched parquets ===",
          file=sys.stderr)
    for fname in TARGET_FILES:
        path = rdir / fname
        if not path.exists():
            continue
        print(f"  {fname}...", file=sys.stderr)
        t = pq.read_table(path)
        records = t.to_pylist()

        n_invalid_fm = 0
        n_canonical = 0
        n_clustered_multi = 0
        for r in records:
            sid = r["skill_id"]
            after = r.get("after_content") or ""
            # Cluster info
            cluster_id = cluster_of.get(sid, "")
            canonical_sid = canonical_map.get(cluster_id, sid)
            r["skill_cluster_id"] = cluster_id
            r["is_canonical"] = (sid == canonical_sid)
            if r["is_canonical"]:
                n_canonical += 1
            # Multi-cluster check
            tags = list(r.get("quality_tags") or [])
            # Frontmatter validation (only for non-initial records, since
            # initial is the creation point)
            if not has_valid_frontmatter(after):
                if "invalid_frontmatter" not in tags:
                    tags.append("invalid_frontmatter")
                n_invalid_fm += 1
            # Same-author dup
            if r["pair_id"] in same_author_dups:
                if "same_author_dup" not in tags:
                    tags.append("same_author_dup")
            r["quality_tags"] = tags

        # Build new schema with extra columns
        new_schema = pa.schema(
            list(t.schema)
            + [
                pa.field("skill_cluster_id", pa.string()),
                pa.field("is_canonical", pa.bool_()),
            ]
        )
        out_tmp = path.with_suffix(".tmp.parquet")
        pq.write_table(
            pa.Table.from_pylist(records, schema=new_schema),
            out_tmp, compression="zstd",
        )
        out_tmp.replace(path)
        print(f"    rows: {len(records):,}", file=sys.stderr)
        print(f"    canonical: {n_canonical:,}", file=sys.stderr)
        print(f"    invalid_frontmatter tagged: {n_invalid_fm:,}", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
