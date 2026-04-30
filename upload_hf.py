#!/usr/bin/env python3
"""Upload the parquet dataset to HuggingFace.

Usage:
    HF_TOKEN=hf_...  uv run python upload_hf.py --repo-id <namespace>/skill-diffs
    # or:
    uv run python upload_hf.py --repo-id <namespace>/skill-diffs --token-from-keychain skill-diffs-hf

Pre-flight:
    1. Create the dataset repo on HF (or use --create flag here)
    2. Have a token with write scope
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


RELEASE_DIR = Path("data/release")


def get_token(args):
    if args.token:
        return args.token
    if args.token_from_keychain:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", args.token_from_keychain, "-w"],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            print(f"ERROR: keychain entry '{args.token_from_keychain}' not found", file=sys.stderr)
            sys.exit(1)
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    if "HUGGINGFACE_TOKEN" in os.environ:
        return os.environ["HUGGINGFACE_TOKEN"]
    print("ERROR: provide --token, --token-from-keychain, or set HF_TOKEN", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload skill-diffs dataset to HF.")
    parser.add_argument("--repo-id", required=True,
                        help="HF dataset repo id, e.g. 'username/skill-diffs'")
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-from-keychain", default=None,
                        help="macOS Keychain service name to read token from")
    parser.add_argument("--create", action="store_true",
                        help="Create the repo on HF if it doesn't exist")
    parser.add_argument("--private", action="store_true",
                        help="Create as private (only with --create)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token(args)

    if not RELEASE_DIR.exists():
        print(f"ERROR: {RELEASE_DIR} does not exist. Run consolidate.py first.", file=sys.stderr)
        sys.exit(1)

    files = sorted(RELEASE_DIR.glob("*.parquet")) + sorted(RELEASE_DIR.glob("*.md"))
    if not files:
        print(f"ERROR: no files in {RELEASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo: {args.repo_id}", file=sys.stderr)
    print(f"Files to upload ({len(files)}):", file=sys.stderr)
    total_size = 0
    for f in files:
        size_mb = f.stat().st_size / 1e6
        total_size += size_mb
        print(f"  {f.name:<28} {size_mb:>8.1f} MB", file=sys.stderr)
    print(f"  {'TOTAL':<28} {total_size:>8.1f} MB", file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] not uploading", file=sys.stderr)
        return

    api = HfApi(token=token)

    if args.create:
        print(f"\nCreating dataset repo {args.repo_id} (private={args.private})...", file=sys.stderr)
        create_repo(
            args.repo_id, token=token, repo_type="dataset",
            private=args.private, exist_ok=True,
        )

    print(f"\nUploading folder {RELEASE_DIR} to {args.repo_id}...", file=sys.stderr)
    api.upload_folder(
        folder_path=str(RELEASE_DIR),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload skill-diffs v0.1",
    )
    print(f"\nDone. View at: https://huggingface.co/datasets/{args.repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
