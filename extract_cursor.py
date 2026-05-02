#!/usr/bin/env python3
"""Extract Cursor rules commit histories from a GitHub repo as diff pairs.

Cursor rules format:
    .cursorrules                 — single file at repo root (legacy)
    .cursor/rules/*.mdc          — multi-file rules dir (current)
    .cursorrules.md              — variant
    .cursorrules-*.md            — locale variants

Same JSONL output schema as extract.py with format='cursor_rule' added.

Usage:
    uv run python extract_cursor.py <repo_url> [--output PATH]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def is_cursor_rule_path(path):
    """True if path looks like a Cursor rule file."""
    p = Path(path)
    name = p.name.lower()
    parent_parts = [s.lower() for s in p.parts[:-1]]

    # Direct root-level cursor rules
    if name == ".cursorrules":
        return True
    if name.startswith(".cursorrules") and name.endswith((".md", ".txt", ".mdx")):
        # .cursorrules.md, .cursorrules-zh.md, .cursorrules.txt
        return True

    # .cursor/rules/*.mdc (or .md)
    if ".cursor" in parent_parts and "rules" in parent_parts:
        if name.endswith((".mdc", ".md")):
            return True

    return False


def run_git(args, cwd, check=True):
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=check,
    )
    return result.stdout


def parse_repo_slug(url):
    if "/" in url and not url.startswith(("http", "git@")):
        parts = url.strip("/").split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
    parsed = urlparse(url)
    path = parsed.path.strip("/").removesuffix(".git")
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse repo from URL: {url}")
    return parts[0], parts[1]


def clone_partial(url, dest):
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet",
         url, str(dest)], check=True,
    )


def find_rule_files_in_head(repo_dir):
    stdout = run_git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=repo_dir)
    rules = []
    for line in stdout.splitlines():
        if is_cursor_rule_path(line):
            rules.append(line)
    return sorted(rules)


def get_file_history(repo_dir, path):
    sep = "\x1f"
    eol = "\x1e"
    fmt = sep.join(["%H", "%an", "%ae", "%aI", "%s"]) + eol
    stdout = run_git(
        ["log", "--follow", "--reverse", f"--format={fmt}", "--", path],
        cwd=repo_dir,
    )
    commits = []
    for record in stdout.split(eol):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(sep, 4)
        if len(parts) != 5:
            continue
        sha, author, email, date, subject = parts
        commits.append({
            "sha": sha, "author": author, "email": email,
            "date": date, "subject": subject,
        })
    return commits


def get_blob_at_commit(repo_dir, sha, path):
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_dir, capture_output=True, check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def diff_numstat(repo_dir, before_sha, after_sha, path):
    before_ref = before_sha or GIT_EMPTY_TREE
    stdout = run_git(
        ["diff", "--numstat", before_ref, after_sha, "--", path],
        cwd=repo_dir,
    )
    line = stdout.strip().split("\n", 1)[0]
    if not line:
        return (0, 0)
    parts = line.split("\t")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (0, 0)


def rule_name_for_path(path):
    """Best-effort skill_name analog for cursor rules."""
    p = Path(path)
    if p.name.lower() == ".cursorrules":
        return "cursorrules-root"
    if p.name.lower().startswith(".cursorrules"):
        return p.stem.lstrip(".") or "cursorrules-root"
    # .cursor/rules/foo.mdc -> foo
    return p.stem


def extract_repo(url, output_path, quiet=False):
    def log(msg):
        if not quiet:
            print(msg, file=sys.stderr)

    owner, repo = parse_repo_slug(url)
    repo_full = f"{owner}/{repo}"

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "repo"
        log(f"Cloning {url} (partial)...")
        clone_partial(url, repo_dir)

        rule_files = find_rule_files_in_head(repo_dir)
        log(f"Found {len(rule_files)} cursor-rule file(s) in HEAD")

        records = 0
        with open(output_path, "w") as out:
            for rule_path in rule_files:
                history = get_file_history(repo_dir, rule_path)
                log(f"  {rule_path}: {len(history)} commit(s)")

                prev_sha = None
                prev_content = None
                for commit in history:
                    sha = commit["sha"]
                    content = get_blob_at_commit(repo_dir, sha, rule_path)
                    if content is None:
                        continue
                    added, removed = diff_numstat(repo_dir, prev_sha, sha, rule_path)
                    char_delta = len(content) - (len(prev_content) if prev_content else 0)

                    record = {
                        "repo": repo_full,
                        "format": "cursor_rule",
                        "skill_path": rule_path,         # named skill_* for schema parity
                        "skill_name": rule_name_for_path(rule_path),
                        "before_sha": prev_sha,
                        "after_sha": sha,
                        "before_content": prev_content,
                        "after_content": content,
                        "commit_subject": commit["subject"],
                        "commit_author": commit["author"],
                        "commit_email": commit["email"],
                        "commit_date": commit["date"],
                        "lines_added": added,
                        "lines_removed": removed,
                        "char_delta": char_delta,
                        "is_initial": prev_sha is None,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
                    prev_sha = sha
                    prev_content = content

        log(f"\nWrote {records} record(s) to {output_path}")
        return records


def main():
    parser = argparse.ArgumentParser(description="Extract Cursor rule commit history.")
    parser.add_argument("repo_url", help="GitHub repo URL or owner/repo shorthand")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    owner, repo = parse_repo_slug(args.repo_url)
    if args.output is None:
        out_dir = Path("data")
        out_dir.mkdir(exist_ok=True)
        args.output = out_dir / f"{owner}__{repo}.cursor.jsonl"

    repo_url = args.repo_url
    if "://" not in repo_url and not repo_url.startswith("git@"):
        repo_url = f"https://github.com/{owner}/{repo}.git"

    extract_repo(repo_url, args.output)


if __name__ == "__main__":
    main()
