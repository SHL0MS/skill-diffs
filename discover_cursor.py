#!/usr/bin/env python3
"""Discover repos that contain Cursor rules files.

Cursor rules format:
    .cursorrules                 — root, plain markdown
    .cursorrules.md / .txt       — root, variants
    .cursor/rules/*.mdc          — multi-file rules dir

GitHub code-search caps at 1000 results per query, and 10 req/min on the
code-search endpoint. We use multiple queries to cover the corpus better.

Output:
    data/cursor_repos.txt          — newline-delimited owner/repo
    data/discover_cursor_log.json  — provenance
"""
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def run_gh(args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  gh error: {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def search_repos(query, limit=200):
    print(f"  REPO search: {query}", file=sys.stderr)
    args = [
        "search", "repos", query, "--limit", str(limit),
        "--json", "fullName,stargazersCount,description,pushedAt",
    ]
    return run_gh(args) or []


def search_code(query, max_results=1000):
    print(f"  CODE search: {query}", file=sys.stderr)
    items = []
    page = 1
    per_page = 100
    while len(items) < max_results:
        args = ["api", f"search/code?q={query}&per_page={per_page}&page={page}"]
        data = run_gh(args)
        if not data or "items" not in data:
            break
        new_items = data["items"]
        for it in new_items:
            items.append({"repo": it["repository"]["full_name"], "path": it["path"]})
        if len(new_items) < per_page:
            break
        page += 1
        time.sleep(6.5)
    return items


def main():
    found = defaultdict(list)

    print("=== Repository search ===", file=sys.stderr)
    repo_queries = [
        "cursor-rules in:name,description fork:false",
        '"cursor rules" in:name,description fork:false',
        "awesome-cursor-rules in:name fork:false",
        "cursorrules in:name,description fork:false",
        ".cursorrules in:name,description fork:false",
    ]
    for q in repo_queries:
        for r in search_repos(q, limit=200):
            found[r["fullName"]].append(("repo_search", q))
        time.sleep(2)

    print("\n=== Code search ===", file=sys.stderr)
    # NOTE: GitHub code search supports `path:` qualifier inconsistently.
    # `filename:` works for direct filenames; `extension:` works for ext.
    code_queries = [
        "filename:.cursorrules",
        "filename:.cursorrules.md",
        "filename:.cursorrules.txt",
        # MDC extension is mostly Cursor-specific
        "extension:mdc",
    ]
    for q in code_queries:
        url_q = q.replace(" ", "+")
        for it in search_code(url_q, max_results=1000):
            # Only count if path actually looks like a cursor rule
            p = it["path"].lower()
            name = p.split("/")[-1]
            is_cursor = (
                name.startswith(".cursorrules")
                or (".cursor/rules/" in p and (p.endswith(".mdc") or p.endswith(".md")))
                or p.endswith(".mdc")
            )
            if is_cursor:
                found[it["repo"]].append(("code_search", q))

    print("\n=== Aggregating ===", file=sys.stderr)
    all_found = sorted(found.keys())
    print(f"Total unique repos found: {len(all_found):,}", file=sys.stderr)

    out_repos = Path("data/cursor_repos.txt")
    out_log = Path("data/discover_cursor_log.json")
    out_repos.parent.mkdir(parents=True, exist_ok=True)
    out_repos.write_text("\n".join(all_found) + ("\n" if all_found else ""))
    out_log.write_text(json.dumps({r: found[r] for r in all_found}, indent=2))

    print(f"\nWrote {out_repos}", file=sys.stderr)
    print(f"Wrote {out_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
