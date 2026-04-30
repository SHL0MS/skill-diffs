#!/usr/bin/env python3
"""Fetch the unique repo list from huzey/claude-skills as our scrape seed.

Uses DuckDB to query HF's auto-converted parquet shards directly over HTTPS
with column pushdown — no full download needed.

Writes data/huzey_repos.txt — one owner/repo per line.
"""
import sys
from pathlib import Path

import duckdb

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/huzey/claude-skills/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/huzey/claude-skills/parquet/default/train/1.parquet",
    "https://huggingface.co/api/datasets/huzey/claude-skills/parquet/default/train/2.parquet",
]


def main():
    print(f"Querying {len(PARQUET_URLS)} parquet shard(s) via DuckDB...", file=sys.stderr)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    urls_sql = "[" + ", ".join(f"'{u}'" for u in PARQUET_URLS) + "]"
    query = f"""
        SELECT DISTINCT repo
        FROM read_parquet({urls_sql})
        WHERE repo IS NOT NULL
        ORDER BY repo
    """
    rows = con.execute(query).fetchall()
    repos = [r[0] for r in rows]

    print(f"Total unique repos: {len(repos)}", file=sys.stderr)

    out = Path("data") / "huzey_repos.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(repos) + "\n")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
