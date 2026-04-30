#!/usr/bin/env python3
"""Aggregate per-repo bundled JSONL files into a single bundled.parquet.

Run after extract_bundled.py completes:
    uv run python aggregate_bundled.py

Output: data/release/bundled.parquet
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


BUNDLED_DIR = Path("data/bundled")
RELEASE_DIR = Path("data/release")


def stable_id(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def bundled_schema():
    file_struct = pa.struct([
        ("path", pa.string()),
        ("size", pa.int64()),
        ("content", pa.large_string()),
        ("binary_or_oversize", pa.bool_()),
    ])
    return pa.schema([
        ("skill_id", pa.string()),
        ("repo", pa.string()),
        ("skill_path", pa.string()),
        ("skill_dir", pa.string()),
        ("skill_name", pa.string()),
        ("head_sha", pa.string()),
        ("bundled_count", pa.int32()),
        ("bundled_text_count", pa.int32()),
        ("bundled_files", pa.list_(file_struct)),
    ])


def normalize(rec):
    return {
        "skill_id": stable_id(rec["repo"], rec["skill_path"]),
        "repo": rec["repo"],
        "skill_path": rec["skill_path"],
        "skill_dir": rec.get("skill_dir") or "",
        "skill_name": rec.get("skill_name") or "",
        "head_sha": rec.get("head_sha") or "",
        "bundled_count": int(rec.get("bundled_count", 0)),
        "bundled_text_count": int(rec.get("bundled_text_count", 0)),
        "bundled_files": [
            {
                "path": b.get("path", ""),
                "size": int(b.get("size", 0)),
                "content": b.get("content"),
                "binary_or_oversize": bool(b.get("binary_or_oversize", False)),
            }
            for b in (rec.get("bundled_files") or [])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate bundled JSONL → parquet.")
    parser.add_argument("--input-dir", default=str(BUNDLED_DIR))
    parser.add_argument("--output", default=str(RELEASE_DIR / "bundled.parquet"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        print(f"ERROR: no JSONL files in {in_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Aggregating {len(files):,} bundled JSONL files...", file=sys.stderr)

    schema = bundled_schema()
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    batch = []
    n = 0
    started = time.time()

    for fi, f in enumerate(files):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                batch.append(normalize(rec))
                n += 1
                if len(batch) >= args.batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{len(files)}] {n:,} skill records aggregated",
                  file=sys.stderr)

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    writer.close()

    elapsed = int(time.time() - started)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed}s. Wrote {n:,} skill records to {out_path}", file=sys.stderr)
    print(f"  Size: {size_mb:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
