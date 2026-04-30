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
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract import clone_partial, parse_repo_slug, run_git


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"  # legacy; only used as fallback
RELEASE_DIR = DATA_DIR / "release"
BUNDLED_DIR = DATA_DIR / "bundled"
BUNDLED_MANIFEST = DATA_DIR / "bundled_manifest.jsonl"

# Cache for skill paths grouped by repo (loaded once from parquet)
_SKILL_PATHS_BY_REPO = None

MAX_FILE_BYTES = 1_000_000  # 1 MB per file cap
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


def output_path(repo_full):
    safe = repo_full.replace("/", "__")
    return BUNDLED_DIR / f"{safe}.jsonl"


def _load_skill_paths_by_repo():
    """Load (repo -> sorted list of skill_paths) once, from parquet if available."""
    global _SKILL_PATHS_BY_REPO
    if _SKILL_PATHS_BY_REPO is not None:
        return _SKILL_PATHS_BY_REPO

    diffs_parquet = RELEASE_DIR / "diffs.parquet"
    by_repo = {}
    if diffs_parquet.exists():
        import pyarrow.parquet as pq
        t = pq.read_table(diffs_parquet, columns=["repo", "skill_path"])
        for r, p in zip(t["repo"].to_pylist(), t["skill_path"].to_pylist()):
            by_repo.setdefault(r, set()).add(p)
    elif RAW_DIR.exists():
        # Legacy fallback: read from per-repo JSONL
        for f in RAW_DIR.glob("*.jsonl"):
            with open(f) as fp:
                for line in fp:
                    try:
                        rec = json.loads(line)
                        by_repo.setdefault(rec["repo"], set()).add(rec["skill_path"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    _SKILL_PATHS_BY_REPO = {k: sorted(v) for k, v in by_repo.items()}
    return _SKILL_PATHS_BY_REPO


def list_skill_paths_for_repo(repo_full):
    """Return sorted list of skill_paths for the repo, from the global cache."""
    return _load_skill_paths_by_repo().get(repo_full, [])


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


def _download_tarball(repo_full, max_bytes=500 * 1024 * 1024):
    """Download GitHub repo tarball for HEAD. Returns (tar_bytes, sha_short)."""
    owner, name = repo_full.split("/", 1)
    url = f"https://codeload.github.com/{owner}/{name}/tar.gz/HEAD"
    req = urllib.request.Request(url, headers={"User-Agent": "skill-diffs/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        # Stream-read with cap
        chunks = []
        total = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"tarball exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def extract_bundled_for_repo(repo_full, skill_paths=None):
    """Download repo tarball (1 HTTP request) and extract sibling files
    for every known skill folder. Much faster than git clone for repos
    with many bundled files (one network round trip vs. one per file).
    """
    started = time.time()
    out = output_path(repo_full)
    if skill_paths is None:
        skill_paths = list_skill_paths_for_repo(repo_full)
    if not skill_paths:
        return {
            "repo": repo_full,
            "status": "skip",
            "reason": "no_skills_in_diff_data",
            "skills": 0,
            "elapsed_s": 0,
        }

    skill_path_set = set(skill_paths)
    skill_dirs = {}  # skill_path -> skill_dir
    for sp in skill_paths:
        sd = str(Path(sp).parent)
        if sd in (".", ""):
            sd = ""
        skill_dirs[sp] = sd

    try:
        tar_bytes = _download_tarball(repo_full)

        # Collect bundled files indexed by (skill_path, rel_path)
        bundled_by_skill = defaultdict(list)
        head_sha_short = ""

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # Tarball entries are "<owner>-<repo>-<sha>/path/to/file"
                parts = member.name.split("/", 1)
                if len(parts) < 2:
                    continue
                if not head_sha_short:
                    # Extract short SHA from prefix
                    m = re.match(r".+-([0-9a-f]{7,40})$", parts[0])
                    if m:
                        head_sha_short = m.group(1)
                rel_path = parts[1]

                # Find which skill (if any) this file belongs to
                for skill_path, skill_dir in skill_dirs.items():
                    if rel_path == skill_path:
                        continue  # SKILL.md captured separately
                    in_dir = (
                        (skill_dir == "" and "/" not in rel_path)
                        or (skill_dir and rel_path.startswith(skill_dir + "/"))
                    )
                    if not in_dir:
                        continue
                    # Read the file
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content_bytes = f.read()
                    size = len(content_bytes)
                    rel_in_skill = (
                        rel_path[len(skill_dir) + 1:] if skill_dir else rel_path
                    )
                    if size > MAX_FILE_BYTES:
                        bundled_by_skill[skill_path].append({
                            "path": rel_in_skill,
                            "size": size,
                            "content": None,
                            "binary_or_oversize": True,
                        })
                    else:
                        try:
                            text = content_bytes.decode("utf-8")
                            bundled_by_skill[skill_path].append({
                                "path": rel_in_skill,
                                "size": size,
                                "content": text,
                                "binary_or_oversize": False,
                            })
                        except UnicodeDecodeError:
                            bundled_by_skill[skill_path].append({
                                "path": rel_in_skill,
                                "size": size,
                                "content": None,
                                "binary_or_oversize": True,
                            })
                    break  # File only belongs to one skill

        # Write per-skill records
        n_skills = 0
        n_text_files = 0
        with open(out, "w") as fp:
            for skill_path in skill_paths:
                bundled = bundled_by_skill.get(skill_path, [])
                rec = {
                    "repo": repo_full,
                    "skill_path": skill_path,
                    "skill_dir": skill_dirs[skill_path],
                    "skill_name": (
                        Path(skill_dirs[skill_path]).name
                        if skill_dirs[skill_path]
                        else Path(skill_path).stem
                    ),
                    "head_sha": head_sha_short,
                    "bundled_files": bundled,
                    "bundled_count": len(bundled),
                    "bundled_text_count": sum(
                        1 for b in bundled if not b["binary_or_oversize"]
                    ),
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_skills += 1
                n_text_files += rec["bundled_text_count"]

        return {
            "repo": repo_full,
            "status": "ok",
            "skills": n_skills,
            "text_files": n_text_files,
            "elapsed_s": round(time.time() - started, 2),
        }
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        if out.exists():
            out.unlink()
        return {
            "repo": repo_full,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
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

    # Load skill_paths grouped by repo (from parquet — single load, memory-efficient)
    print("Loading repo→skills index from data/release/diffs.parquet...", file=sys.stderr)
    skill_paths_by_repo = _load_skill_paths_by_repo()
    diff_repos = set(skill_paths_by_repo.keys())
    print(f"  Found {len(diff_repos):,} repos with at least one skill", file=sys.stderr)

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
        futures = {
            pool.submit(extract_bundled_for_repo, r, skill_paths_by_repo[r]): r
            for r in pending
        }
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
