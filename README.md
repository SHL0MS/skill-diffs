# skill-diffs

Pipeline that scrapes commit histories of agent skills (`SKILL.md` files) from public GitHub repos and packages them as a dataset of (before, after, intent) diff pairs for training and evaluation.

## What's in the dataset

A snapshot built April 2026 covers **2,774 repos / 91,355 clean diff pairs / 662,885 total records** — roughly 60x larger than the existing public diff corpus (`huzey/claude-skills-diff`). See `data/release/README.md` for the full data card.

## Why

Agent skills are a rare kind of training data: structured procedural specs that have been iteratively refined through human feedback. Existing public datasets capture *snapshots*; this one captures the **evolution** — every commit-by-commit revision with diffs, commit messages (intent labels), and authorship metadata.

Use cases:

- DPO / preference-pair training — `(before, after)` where `after` is the human-corrected version
- Instruction-tuned skill editors — condition on commit subject (intent) → produce the edit
- Pattern mining — what kinds of edits are most common in skill iteration
- Initial-state generation — `skills_initial.parquet` for "create a skill from scratch" training

## Pipeline

```
fetch_huzey_repos.py   →  data/huzey_repos.txt        (522 seed repos via DuckDB on HF parquet)
discover.py            →  data/expansion_repos.txt    (2,454 more via GH search)
batch.py               →  data/raw/<repo>.jsonl       (per-repo diff records)
consolidate.py         →  data/release/*.parquet      (streaming classify + filter + parquet)
upload_hf.py           →  HuggingFace dataset         (publish)
```

`extract.py` is the per-repo extractor used by `batch.py`. `classify.py` and `filter_quality.py` provide the regex intent classifier and quality-tagging logic that `consolidate.py` reuses in its streaming pass.

## Reproducing the dataset

```bash
uv sync
uv run python fetch_huzey_repos.py        # ~5 sec
uv run python discover.py                 # ~5 min (rate-limited)
uv run python batch.py --workers 16       # ~5-6 hours wall time at 16 workers
uv run python consolidate.py              # ~1 min, JSONL → parquet
uv run python upload_hf.py                # push to HuggingFace
```

Each phase is resumable (manifest-based for `batch.py`).

## Files

| File | Purpose |
|---|---|
| `extract.py` | Single-repo SKILL.md commit-history extractor |
| `batch.py` | Parallel extraction across a repo list with per-repo manifest |
| `fetch_huzey_repos.py` | Pull seed repo list from `huzey/claude-skills` via DuckDB |
| `discover.py` | Find non-huzey skill repos via GitHub repo + code search |
| `classify.py` | Regex intent classifier for commit messages |
| `filter_quality.py` | Tag records with quality flags; produce clean subset |
| `consolidate.py` | Streaming JSONL → parquet with classify + filter applied inline |
| `extract_bundled.py` | (v0.2) Capture sibling files (scripts/, references/) from skill folders |
| `analyze.py` | Quick stats over the JSONL output |

## Status

- **v0.1 (current)** — diff dataset with regex intent classification and heuristic filtering
- **v0.2 (planned)** — LLM classification of `unknown` intent records (~$5-10 budget); bundled resources capture; PR description metadata
- **v0.3** — broader corpus (Sourcegraph for current GitHub coverage; non-`SKILL.md` formats like Cursor rules and OpenCode skills)
