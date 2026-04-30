#!/usr/bin/env python3
"""Extract bundled resource files from each skill folder at HEAD.

For each (repo, skill_path) we already have diff records for, capture the
non-SKILL.md files in the skill's folder (scripts/, references/, assets/, etc.)
at the repo's HEAD. This gives us the "complete skill" snapshot adjacent to
the SKILL.md commit history.

Usage:
    uv run python extract_bundled.py [--workers N] [--max-new N]

Output:
    data/bundled/<owner>__<repo>.jsonl  (one record per skill folder)
    data/bundled_manifest.jsonl         (per-repo run metadata)
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract import clone_partial, parse_repo_slug, run_git


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BUNDLED_DIR = DATA_DIR / "bundled"
BUNDLED_MANIFEST = DATA_DIR / "bundled_manifest.jsonl"

MAX_FILE_BYTES = 1_000_000  # 1 MB per file cap
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


def output_path(repo_full):
    safe = repo_full.replace("/", "__")
    return BUNDLED_DIR / f"{safe}.jsonl"


def list_skill_paths_for_repo(repo_full):
    """Read skill paths from the existing diff JSONL for this repo."""
    diff_file = RAW_DIR / f"{repo_full.replace('/', '__')}.jsonl"
    if not diff_file.exists():
        return []
    skills = set()
    with open(diff_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                skills.add(rec["skill_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(skills)


def get_head_sha(repo_dir):
    return run_git(["rev-parse", "HEAD"], cwd=repo_dir).strip()


def list_tree_files(repo_dir, sha, prefix):
    """List all files at given commit under the given path prefix."""
    if not prefix.endswith("/"):
        prefix_filter = prefix + "/"
    else:
        prefix_filter = prefix

    stdout = run_git(
        ["ls-tree", "-r", "--name-only", sha, prefix_filter or "."],
        cwd=repo_dir,
        check=False,
    )
    files = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip files in unwanted directories
        parts = line.split("/")
        if any(p in SKIP_DIRS for p in parts):
            continue
        files.append(line)
    return files


def get_blob_text(repo_dir, sha, path):
    """Get UTF-8 file content. Returns (content, size_bytes) or (None, size)."""
    result = subprocess.run(
        ["git", "cat-file", "-s", f"{sha}:{path}"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        size = int(result.stdout.strip())
    except (ValueError, AttributeError):
        return (None, 0)

    if size > MAX_FILE_BYTES:
        return (None, size)

    blob = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_dir,
        capture_output=True,
        check=False,
    )
    if blob.returncode != 0:
        return (None, size)
    try:
        text = blob.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return (None, size)
    return (text, size)


def extract_bundled_for_repo(repo_full):
    """Clone repo, extract bundled resources for every skill_path we know."""
    started = time.time()
    out = output_path(repo_full)
    skill_paths = list_skill_paths_for_repo(repo_full)
    if not skill_paths:
        return {
            "repo": repo_full,
            "status": "skip",
            "reason": "no_skills_in_diff_data",
            "skills": 0,
            "elapsed_s": 0,
        }

    repo_url = f"https://github.com/{repo_full}.git"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            clone_partial(repo_url, repo_dir)
            head_sha = get_head_sha(repo_dir)

            n_skills = 0
            n_files = 0
            with open(out, "w") as fp:
                for skill_path in skill_paths:
                    skill_dir = str(Path(skill_path).parent)
                    if skill_dir in (".", ""):
                        skill_dir = ""
                    files = list_tree_files(repo_dir, head_sha, skill_dir)
                    bundled = []
                    for f in files:
                        if f == skill_path:
                            continue  # SKILL.md captured separately
                        # Only include files actually under the skill folder
                        if skill_dir and not f.startswith(skill_dir + "/"):
                            continue
                        text, size = get_blob_text(repo_dir, head_sha, f)
                        rel_path = f[len(skill_dir) + 1:] if skill_dir else f
                        if text is None:
                            bundled.append({
                                "path": rel_path,
                                "size": size,
                                "content": None,
                                "binary_or_oversize": True,
                            })
                        else:
                            bundled.append({
                                "path": rel_path,
                                "size": size,
                                "content": text,
                                "binary_or_oversize": False,
                            })
                            n_files += 1

                    rec = {
                        "repo": repo_full,
                        "skill_path": skill_path,
                        "skill_dir": skill_dir,
                        "skill_name": Path(skill_dir).name if skill_dir else Path(skill_path).stem,
                        "head_sha": head_sha,
                        "bundled_files": bundled,
                        "bundled_count": len(bundled),
                        "bundled_text_count": sum(1 for b in bundled if not b["binary_or_oversize"]),
                    }
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_skills += 1

            return {
                "repo": repo_full,
                "status": "ok",
                "skills": n_skills,
                "text_files": n_files,
                "elapsed_s": round(time.time() - started, 2),
            }
    except Exception as e:
        if out.exists():
            out.unlink()
        return {
            "repo": repo_full,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
            "elapsed_s": round(time.time() - started, 2),
        }


def load_manifest_index(path):
    if not path.exists():
        return {}
    index = {}
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
                index[e["repo"]] = e
            except json.JSONDecodeError:
                continue
    return index


def append_manifest(entry, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Bundled resource extractor.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)

    # Discover repos that have diff data
    diff_repos = set()
    for f in RAW_DIR.glob("*.jsonl"):
        # filename is "<owner>__<repo>.jsonl" (single underscore)
        repo_full = f.stem.replace("__", "/", 1)
        diff_repos.add(repo_full)

    bundled_index = load_manifest_index(BUNDLED_MANIFEST)

    pending = []
    skipped_ok = 0
    skipped_err = 0
    for r in sorted(diff_repos):
        entry = bundled_index.get(r)
        if entry and entry.get("status") in ("ok", "skip"):
            skipped_ok += 1
            continue
        if entry and entry.get("status") == "error" and not args.retry_errors:
            skipped_err += 1
            continue
        pending.append(r)

    if args.max_new is not None:
        pending = pending[: args.max_new]

    print(f"Diff repos available: {len(diff_repos)}", file=sys.stderr)
    print(f"  Bundled already done:    {skipped_ok}", file=sys.stderr)
    print(f"  Bundled prev failed:     {skipped_err}", file=sys.stderr)
    print(f"  To process now:          {len(pending)}", file=sys.stderr)
    print(f"Workers: {args.workers}\n", file=sys.stderr)

    started = time.time()
    n_ok = n_err = n_skip = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_bundled_for_repo, r): r for r in pending}
        for i, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            append_manifest(entry, BUNDLED_MANIFEST)

            if entry["status"] == "ok":
                n_ok += 1
                detail = f"{entry['skills']} skills, {entry['text_files']} text files"
            elif entry["status"] == "skip":
                n_skip += 1
                detail = f"skip ({entry['reason']})"
            else:
                n_err += 1
                detail = entry["error"][:80]

            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate if rate > 0 else 0
            print(
                f"[{i}/{len(pending)}] {entry['status'].upper()} {entry['repo']}: "
                f"{detail} ({entry['elapsed_s']}s) | "
                f"ok={n_ok} skip={n_skip} err={n_err} | eta={int(eta)}s",
                file=sys.stderr,
            )

    print(f"\nDone in {int(time.time() - started)}s. "
          f"ok={n_ok} skip={n_skip} err={n_err}", file=sys.stderr)


if __name__ == "__main__":
    main()
