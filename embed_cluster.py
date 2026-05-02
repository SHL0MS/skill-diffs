#!/usr/bin/env python3
"""Embed skill content with BAAI/bge-small-en-v1.5 and cluster via cosine.

Steps:
  1. Load unique (skill_id, after_content) pairs from skills_initial.parquet
     and cursor_rules_initial.parquet.
  2. Truncate after_content to first ~2000 chars (~512 tokens) since the
     frontmatter+intro carries most of the semantic signal.
  3. Embed in batches with sentence-transformers.
  4. Cache embeddings to data/embeddings.parquet (skill_id, vector).
  5. Build FAISS IVFFlat index for cosine similarity.
  6. For each skill, query top-K nearest neighbors. If similarity > THRESHOLD,
     union them into the same cluster.
  7. Write cluster map: skill_id -> skill_semantic_cluster_id (string), is_canonical.

Outputs:
  data/embeddings.parquet
  data/semantic_clusters.parquet  (skill_id, cluster_id, is_canonical)

Use add_semantic_clusters.py to merge into release parquets.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_TRUNCATE = 2000
DEFAULT_BATCH = 64
DEFAULT_THRESHOLD = 0.85
DEFAULT_TOPK = 10


def load_unique_skills(release_dir):
    """Return list of (skill_id, content) deduped by skill_id, drawn from
    skills_initial.parquet + cursor_rules_initial.parquet."""
    rdir = Path(release_dir)
    sources = []
    for fname in ("skills_initial.parquet", "cursor_rules_initial.parquet"):
        p = rdir / fname
        if p.exists():
            t = pq.read_table(p, columns=["skill_id", "after_content"])
            sources.append(t)

    if not sources:
        print(f"ERROR: no initial parquets in {release_dir}", file=sys.stderr)
        sys.exit(1)

    if len(sources) > 1:
        t = pa.concat_tables(sources)
    else:
        t = sources[0]

    seen = set()
    pairs = []
    for sid, content in zip(
        t["skill_id"].to_pylist(), t["after_content"].to_pylist()
    ):
        if sid in seen:
            continue
        seen.add(sid)
        pairs.append((sid, content or ""))
    print(f"  loaded {len(pairs):,} unique skills", file=sys.stderr)
    return pairs


def truncate(content, n=DEFAULT_TRUNCATE):
    if not content:
        return ""
    return content[:n]


def embed(pairs, batch_size, truncate_n, model_name=MODEL_NAME):
    """Returns (skill_ids, embeddings) — embeddings is float32 [N, D] L2-normalized."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  device: {device}", file=sys.stderr)
    print(f"  loading model {model_name}...", file=sys.stderr)
    model = SentenceTransformer(model_name, device=device)

    skill_ids = [p[0] for p in pairs]
    texts = [truncate(p[1], truncate_n) for p in pairs]

    print(f"  embedding {len(texts):,} texts (batch_size={batch_size})...",
          file=sys.stderr)
    started = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - started
    print(f"  done in {elapsed:.0f}s ({len(texts) / elapsed:.0f} texts/sec)",
          file=sys.stderr)
    return skill_ids, embs.astype(np.float32)


def write_embeddings(skill_ids, embs, out_path):
    """Write to parquet as skill_id + vector list column."""
    print(f"  writing embeddings to {out_path}...", file=sys.stderr)
    arr = pa.array([list(map(float, v)) for v in embs],
                   type=pa.list_(pa.float32(), embs.shape[1]))
    table = pa.table({
        "skill_id": pa.array(skill_ids),
        "embedding": arr,
    })
    pq.write_table(table, out_path, compression="zstd")
    print(f"    wrote {table.num_rows:,} rows", file=sys.stderr)


def cluster(skill_ids, embs, threshold, topk):
    """Greedy union-find clustering on cosine sim >= threshold via FAISS."""
    import faiss

    print(f"  building FAISS index (n={len(embs):,}, d={embs.shape[1]})...",
          file=sys.stderr)
    started = time.time()
    index = faiss.IndexFlatIP(embs.shape[1])  # inner product = cosine on L2-normalized
    index.add(embs)
    print(f"    index built in {time.time() - started:.0f}s", file=sys.stderr)

    print(f"  searching top-{topk} for each (threshold={threshold})...",
          file=sys.stderr)
    started = time.time()
    sims, idxs = index.search(embs, topk)
    print(f"    search done in {time.time() - started:.0f}s", file=sys.stderr)

    # Union-find
    parent = list(range(len(skill_ids)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_edges = 0
    for i in range(len(skill_ids)):
        for j_idx in range(topk):
            j = idxs[i, j_idx]
            s = sims[i, j_idx]
            if j == i:
                continue
            if s >= threshold:
                union(i, j)
                n_edges += 1
        if (i + 1) % 50000 == 0:
            print(f"    [{i+1:,}/{len(skill_ids):,}]", file=sys.stderr)
    print(f"  {n_edges:,} edges added", file=sys.stderr)

    # Assign cluster ids — root index gets stable cluster id
    root_to_cluster = {}
    cluster_of = []
    for i in range(len(skill_ids)):
        r = find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = f"sc{len(root_to_cluster):07d}"
        cluster_of.append(root_to_cluster[r])

    # Canonical = lex-smallest skill_id within cluster
    cluster_to_members = {}
    for sid, c in zip(skill_ids, cluster_of):
        cluster_to_members.setdefault(c, []).append(sid)
    canonical = {c: min(members) for c, members in cluster_to_members.items()}

    is_canonical = [
        sid == canonical[c] for sid, c in zip(skill_ids, cluster_of)
    ]

    n_clusters = len(cluster_to_members)
    n_singletons = sum(1 for m in cluster_to_members.values() if len(m) == 1)
    n_multi = n_clusters - n_singletons
    n_clustered = sum(len(m) for m in cluster_to_members.values() if len(m) > 1)
    print(f"  clusters: {n_clusters:,} "
          f"(singletons={n_singletons:,}, multi={n_multi:,} "
          f"covering {n_clustered:,} skills)", file=sys.stderr)

    return cluster_of, is_canonical


def main():
    parser = argparse.ArgumentParser(description="Embed + cluster skills.")
    parser.add_argument("--release-dir", default="data/release")
    parser.add_argument("--embeddings-path", default="data/embeddings.parquet")
    parser.add_argument("--clusters-path", default="data/semantic_clusters.parquet")
    parser.add_argument("--truncate", type=int, default=DEFAULT_TRUNCATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding step, load existing embeddings")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("=== Phase D: embed + cluster ===", file=sys.stderr)
    print(f"  release-dir: {args.release_dir}", file=sys.stderr)
    print(f"  model: {MODEL_NAME}", file=sys.stderr)
    print(f"  threshold: {args.threshold}", file=sys.stderr)

    if args.skip_embed and Path(args.embeddings_path).exists():
        print("  loading existing embeddings...", file=sys.stderr)
        t = pq.read_table(args.embeddings_path)
        skill_ids = t["skill_id"].to_pylist()
        # Embedding column is fixed-size list; convert to numpy
        emb_lists = t["embedding"].to_pylist()
        embs = np.array(emb_lists, dtype=np.float32)
        print(f"    loaded {len(skill_ids):,} embeddings ({embs.shape[1]}D)",
              file=sys.stderr)
    else:
        pairs = load_unique_skills(args.release_dir)
        if args.limit:
            pairs = pairs[: args.limit]
        skill_ids, embs = embed(pairs, args.batch_size, args.truncate)
        write_embeddings(skill_ids, embs, args.embeddings_path)

    cluster_of, is_canonical = cluster(skill_ids, embs, args.threshold, args.topk)

    out_t = pa.table({
        "skill_id": pa.array(skill_ids),
        "skill_semantic_cluster_id": pa.array(cluster_of),
        "is_semantic_canonical": pa.array(is_canonical),
    })
    pq.write_table(out_t, args.clusters_path, compression="zstd")
    print(f"\nWrote {args.clusters_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
