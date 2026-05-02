#!/usr/bin/env python3
"""Discover SKILL.md-bearing repos for OpenCode, Hermes Agent, and OpenClaw.

All three platforms use the SAME `<skill>/SKILL.md` file format as Claude/
Anthropic, only the repo conventions differ (top-level dirs vs `.agents/skills/`
vs `<category>/<skill>/`). So we reuse extract.py downstream — only discovery
needs to expand.

Outputs (newline-delimited owner/repo, deduped against existing v0.3 corpus):
    data/opencode_repos.txt
    data/hermes_repos.txt
    data/openclaw_repos.txt
    data/discover_v04_log.json   — provenance: which query yielded which repo
"""
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


HUZEY_SEED = Path("data/huzey_repos.txt")
EXPANSION = Path("data/expansion_repos.txt")


def run_gh(args):
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False,
    )
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
    data = run_gh(args)
    return data or []


def search_code(query, max_results=1000):
    """Code search via gh api search/code with manual pagination."""
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
        time.sleep(6.5)  # 10 req/min code-search rate limit
    return items


PLATFORMS = {
    "opencode": {
        "repo_queries": [
            "opencode skills in:name,description fork:false",
            "opencode-skills in:name,description fork:false",
            '"opencode skill" in:name,description fork:false',
        ],
        "code_queries": [
            "filename:SKILL.md+opencode+in:file",
            "filename:SKILL.md+path:.opencode",
        ],
    },
    "hermes": {
        "repo_queries": [
            "hermes-agent skills in:name,description fork:false",
            "hermes skills in:name,description fork:false",
            '"hermes agent skill" in:name,description fork:false',
            "awesome-hermes-skills in:name fork:false",
        ],
        "code_queries": [
            "filename:SKILL.md+hermes-agent+in:file",
        ],
    },
    "openclaw": {
        "repo_queries": [
            "openclaw skills in:name,description fork:false",
            "clawhub in:name,description fork:false",
            "awesome-openclaw in:name fork:false",
            '"openclaw skill" in:name,description fork:false',
        ],
        "code_queries": [
            "filename:SKILL.md+openclaw+in:file",
            "filename:SKILL.md+path:.agents/skills",
        ],
    },
}


def main():
    # Load known repos to dedupe against
    known = set()
    if HUZEY_SEED.exists():
        known.update(HUZEY_SEED.read_text().splitlines())
    if EXPANSION.exists():
        known.update(EXPANSION.read_text().splitlines())
    print(f"Known repos (to exclude): {len(known):,}\n", file=sys.stderr)

    full_log = {}

    for platform, queries in PLATFORMS.items():
        print(f"=== Platform: {platform} ===", file=sys.stderr)
        found = defaultdict(list)

        for q in queries["repo_queries"]:
            results = search_repos(q, limit=200)
            for r in results:
                found[r["fullName"]].append(("repo_search", q))
            time.sleep(2)

        for q in queries["code_queries"]:
            url_q = q.replace(" ", "+")
            results = search_code(url_q, max_results=1000)
            for it in results:
                found[it["repo"]].append(("code_search", q))

        # Filter dupes against known
        new_repos = sorted(r for r in found.keys() if r not in known)
        all_found = sorted(found.keys())

        print(f"  Total found: {len(all_found):,}", file=sys.stderr)
        print(f"  Already known: {len(all_found) - len(new_repos):,}", file=sys.stderr)
        print(f"  NEW: {len(new_repos):,}\n", file=sys.stderr)

        out = Path(f"data/{platform}_repos.txt")
        out.write_text("\n".join(new_repos) + ("\n" if new_repos else ""))
        print(f"  Wrote {out}\n", file=sys.stderr)

        # Add to global known so cross-platform repos go to first list only
        known.update(new_repos)

        full_log[platform] = {r: found[r] for r in all_found}

    Path("data/discover_v04_log.json").write_text(
        json.dumps(full_log, indent=2)
    )
    print("Wrote data/discover_v04_log.json", file=sys.stderr)


if __name__ == "__main__":
    main()
