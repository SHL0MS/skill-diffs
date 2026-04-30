#!/usr/bin/env python3
"""Discover SKILL.md-containing repos beyond the huzey/claude-skills seed list.

Strategy:
  1. Repository search by relevant keywords (high-quality, broadly relevant).
  2. Code search by `filename:SKILL.md`, sliced by path prefix to fit GitHub's
     1000-result-per-query cap and 10 req/min rate limit on code search.
  3. Combine, dedupe against huzey's seed list.

Outputs:
  data/expansion_repos.txt  — newline-delimited owner/repo (only NEW vs huzey)
  data/expansion_log.json   — provenance: which query yielded which repo
"""
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


HUZEY_SEED = Path("data/huzey_repos.txt")


def run_gh(args):
    """Run gh CLI, return parsed JSON or None on failure."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"  gh error: {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def search_repos(query, limit=200):
    """Repository search via gh search repos."""
    print(f"  REPO search: {query}", file=sys.stderr)
    args = [
        "search", "repos", query,
        "--limit", str(limit),
        "--json", "fullName,stargazersCount,description,pushedAt",
    ]
    data = run_gh(args)
    return data or []


def search_code(query, max_results=1000):
    """Code search via gh api search/code with manual pagination.
    Returns list of {repo, path}. Bounded by GitHub's 1000-result cap.
    """
    print(f"  CODE search: {query}", file=sys.stderr)
    items = []
    page = 1
    per_page = 100
    while len(items) < max_results:
        args = [
            "api",
            f"search/code?q={query}&per_page={per_page}&page={page}",
        ]
        data = run_gh(args)
        if not data or "items" not in data:
            break
        new_items = data["items"]
        for it in new_items:
            items.append({
                "repo": it["repository"]["full_name"],
                "path": it["path"],
            })
        if len(new_items) < per_page:
            break
        page += 1
        # Code search is 10 req/min; sleep 6.5s between pages
        time.sleep(6.5)
    return items


def main():
    # Load existing seed list to dedupe against
    if not HUZEY_SEED.exists():
        print(f"ERROR: {HUZEY_SEED} not found. Run fetch_huzey_repos.py first.", file=sys.stderr)
        sys.exit(1)

    huzey_set = set(HUZEY_SEED.read_text().splitlines())
    print(f"Huzey seed list: {len(huzey_set)} repos\n", file=sys.stderr)

    found = defaultdict(list)  # repo -> list of (source, detail)

    # === Phase A: Repository search ===
    print("=== Repository search ===", file=sys.stderr)
    repo_queries = [
        "claude skills in:name,description fork:false",
        "agent skills in:name,description fork:false",
        "claude-skills in:name,description fork:false",
        "agent-skills in:name,description fork:false",
        "claude code skills in:name,description fork:false",
        "skill.md in:readme fork:false",
        "awesome claude skills fork:false",
    ]
    for q in repo_queries:
        results = search_repos(q, limit=200)
        for r in results:
            full_name = r["fullName"]
            found[full_name].append(("repo_search", q))
        time.sleep(2)  # Stay below 30/min

    # === Phase B: Code search by path slice ===
    print("\n=== Code search ===", file=sys.stderr)
    code_queries = [
        "filename:SKILL.md path:.claude/skills",
        "filename:SKILL.md path:skills",
        # Some skills live at repo root or in non-standard paths:
        "filename:SKILL.md",
    ]
    for q in code_queries:
        # URL-encode minimal: spaces -> +
        url_q = q.replace(" ", "+")
        results = search_code(url_q, max_results=1000)
        for it in results:
            found[it["repo"]].append(("code_search", q))

    # === Phase C: Aggregate, dedupe, filter ===
    print("\n=== Aggregating ===", file=sys.stderr)
    all_found = sorted(found.keys())
    new_repos = sorted(r for r in all_found if r not in huzey_set)

    print(f"Total unique repos found: {len(all_found)}", file=sys.stderr)
    print(f"  Already in huzey seed:    {len(all_found) - len(new_repos)}", file=sys.stderr)
    print(f"  NEW (to add):             {len(new_repos)}", file=sys.stderr)

    # Write outputs
    out_repos = Path("data/expansion_repos.txt")
    out_log = Path("data/expansion_log.json")
    out_repos.parent.mkdir(parents=True, exist_ok=True)
    out_repos.write_text("\n".join(new_repos) + "\n")
    out_log.write_text(json.dumps({r: found[r] for r in all_found}, indent=2))

    print(f"\nWrote {out_repos}", file=sys.stderr)
    print(f"Wrote {out_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
