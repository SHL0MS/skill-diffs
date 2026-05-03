#!/usr/bin/env python3
"""Extract SKILL.md commit histories from a GitHub repo as diff pairs.

Usage:
    uv run python extract.py <repo_url> [--output PATH]

Output: JSONL, one record per (commit_pair, skill_file).
"""
import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse


SKILL_FILENAME_LOWER = "skill.md"
GIT_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
DEFAULT_REPO_TIMEOUT_S = 1800  # 30 min — covers all but absolute worst monorepos


class RepoTimeoutError(Exception):
    """Raised when extracting a repo exceeds its time budget."""


def run_git(args, cwd, check=True):
    """Run a git command, return stdout text."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )
    return result.stdout


def parse_repo_slug(url):
    """Extract owner/repo from a GitHub URL or owner/repo shorthand."""
    if "/" in url and not url.startswith(("http", "git@")):
        # owner/repo shorthand
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
    """Partial clone (no blobs, no checkout) — keeps disk usage minimal."""
    subprocess.run(
        [
            "git", "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--quiet",
            url,
            str(dest),
        ],
        check=True,
    )


def find_skill_files_in_head(repo_dir):
    """List paths in HEAD whose basename is SKILL.md (case-insensitive)."""
    stdout = run_git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=repo_dir)
    skills = []
    for line in stdout.splitlines():
        if Path(line).name.lower() == SKILL_FILENAME_LOWER:
            skills.append(line)
    return sorted(skills)


def get_file_history(repo_dir, path):
    """Commits that touched this path, oldest first, with rename following."""
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
            "sha": sha,
            "author": author,
            "email": email,
            "date": date,
            "subject": subject,
        })
    return commits


def get_blob_at_commit(repo_dir, sha, path):
    """Get UTF-8 file contents at a given commit. None if missing or binary."""
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_dir,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def diff_numstat(repo_dir, before_sha, after_sha, path):
    """Return (lines_added, lines_removed) from git diff --numstat."""
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
    # Binary diffs show "-\t-\t..."; treat as 0/0
    return (0, 0)


def extract_repo(url, output_path, quiet=False, timeout=DEFAULT_REPO_TIMEOUT_S):
    """Extract diff pairs from a repo, writing JSONL to output_path.

    timeout: max wall-clock seconds for this whole repo. After timeout
    expires, raises RepoTimeoutError. Partial output is unlinked. Set to
    None to disable.

    Implementation: signal.SIGALRM in the worker subprocess. Works because
    batch.py / batch_v04.py spawn each repo in its own ProcessPoolExecutor
    worker, where signals are isolated. Does not work if extract_repo is
    called from a non-main thread.
    """
    def log(msg):
        if not quiet:
            print(msg, file=sys.stderr)

    owner, repo = parse_repo_slug(url)
    repo_full = f"{owner}/{repo}"

    # Set up timeout via SIGALRM
    started = time.time()
    timed_out = {"flag": False}

    def _alarm_handler(signum, frame):
        timed_out["flag"] = True
        raise RepoTimeoutError(
            f"Repo {repo_full} exceeded {timeout}s timeout"
        )

    prev_handler = None
    if timeout:
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            log(f"Cloning {url} (partial)...")
            clone_partial(url, repo_dir)

            skill_files = find_skill_files_in_head(repo_dir)
            log(f"Found {len(skill_files)} SKILL.md file(s) in HEAD")

            records = 0
            with open(output_path, "w") as out:
                for skill_path in skill_files:
                    # Cheap check: if we've gone over budget, bail before
                    # starting a potentially-expensive history walk
                    if timeout and (time.time() - started) > timeout:
                        raise RepoTimeoutError(
                            f"Repo {repo_full} timeout before processing {skill_path}"
                        )
                    history = get_file_history(repo_dir, skill_path)
                    log(f"  {skill_path}: {len(history)} commit(s)")

                    prev_sha = None
                    prev_content = None
                    for commit in history:
                        sha = commit["sha"]
                        content = get_blob_at_commit(repo_dir, sha, skill_path)
                        if content is None:
                            continue

                        added, removed = diff_numstat(repo_dir, prev_sha, sha, skill_path)
                        char_delta = len(content) - (len(prev_content) if prev_content else 0)

                        record = {
                            "repo": repo_full,
                            "skill_path": skill_path,
                            "skill_name": Path(skill_path).parent.name or Path(skill_path).stem,
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
    except RepoTimeoutError:
        # Clean up partial output on timeout
        try:
            Path(output_path).unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if timeout:
            signal.alarm(0)  # cancel
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Extract SKILL.md commit history as diff pairs."
    )
    parser.add_argument("repo_url", help="GitHub repo URL or owner/repo shorthand")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL path")
    args = parser.parse_args()

    owner, repo = parse_repo_slug(args.repo_url)
    if args.output is None:
        out_dir = Path("data")
        out_dir.mkdir(exist_ok=True)
        args.output = out_dir / f"{owner}__{repo}.jsonl"

    # Normalize owner/repo shorthand to a clone URL
    repo_url = args.repo_url
    if "://" not in repo_url and not repo_url.startswith("git@"):
        repo_url = f"https://github.com/{owner}/{repo}.git"

    extract_repo(repo_url, args.output)


if __name__ == "__main__":
    main()
