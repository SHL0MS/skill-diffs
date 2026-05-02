# skill-diffs

Pipeline that scrapes commit histories of agent skills (`SKILL.md` files) from public GitHub repos and packages them as a dataset of (before, after, intent) diff pairs for training and evaluation.

## What's in the dataset

Published at **[`shl0ms/skill-diffs`](https://huggingface.co/datasets/shl0ms/skill-diffs)** on HuggingFace.

The May 2026 snapshot covers **3 platforms** (Anthropic Claude, OpenCode, Hermes Agent):

- **4,523 source repos** with SPDX license metadata
- **577,794 unique skills** total / **MinHash-clustered** for near-duplicate dedup
- **864,877 total records** (every commit-by-commit revision across all platforms)
- **112,482 clean diff pairs** (default tier — ~75x larger than `huzey/claude-skills-diff`)
- **66,171 records in `curator_training.parquet`** — the recommended training subset for skill-edit / curator models (strict-clean + canonical + non-trivial PR/commit intent + license-known)
- **53,340 records with PR title + body** as intent labels (richer than commit subjects alone)
- **415,506 bundled-resource snapshots** (v0.3 only — does not yet cover OpenCode/Hermes)

See `data/release/README.md` for the full data card.

```python
from datasets import load_dataset

# Default clean tier (all 3 platforms)
diffs = load_dataset("shl0ms/skill-diffs", "diffs_clean", split="train")

# Filter to one platform
hermes = diffs.filter(lambda r: r["platform"] == "hermes_skill")

# The curator-recommended training subset (pre-filtered)
curator = load_dataset("shl0ms/skill-diffs", "curator_training", split="train")

# Bundled skill-folder context (sibling files)
bundled = load_dataset("shl0ms/skill-diffs", "bundled", split="train")
```

## Why

Agent skills are a rare kind of training data: structured procedural specs iteratively refined through human feedback. Existing public datasets capture *snapshots*; this one captures the **evolution** — every commit-by-commit revision with diffs, commit messages, **PR titles + bodies (where available)**, and authorship metadata.

Use cases:

- **Skill-editor / Curator model training** — see `curator_training.parquet`. Designed for fine-tuning a model that takes `(before_skill, intent_text)` and produces the patched skill. Drops in as the LLM review pass for [Hermes Agent's Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator) or any equivalent maintenance loop.
- **DPO / preference-pair training** — `(before, after)` where `after` is the human-corrected version
- **Pattern mining** — what kinds of edits are most common in skill iteration (frontmatter fixes, model name updates, code-block language tags)
- **Initial-state generation** — `skills_initial.parquet` for "create a skill from scratch" training
- **Cross-platform analysis** — `platform` column lets you compare conventions between Anthropic, Hermes Agent, and OpenCode skill formats

## Companion tools

Two complementary scripts ship alongside the dataset:

- **`skill_linter.py`** — rule-based linter (no LLM, no clone, no network) covering 13 patterns derived from observed defects: missing or incomplete frontmatter, missing code-block languages, deprecated model references (e.g. `gpt-3.5-turbo`, `claude-2.x`), legacy API calls (`openai.ChatCompletion`), weak/long descriptions. Validated against 577k skills (61% have at least one finding).
- **`eval_curator.py`** — held-out eval scaffold for benchmarking models on the skill-patch task: given `(before, intent_text)`, produce the patched skill. Built-in baselines (`identity`, `intent_only`) plus adapters for OpenAI, Anthropic, OpenRouter. Metrics: exact_match, edit_distance_ratio, ROUGE-L, BAAI/bge-small-en-v1.5 cosine similarity.

## Pipeline

```
fetch_huzey_repos.py      →  data/huzey_repos.txt
discover.py               →  data/expansion_repos.txt
discover_v04.py           →  data/{opencode,hermes,openclaw}_repos.txt
batch.py                  →  data/raw/<repo>.jsonl                  (Anthropic corpus)
batch_v04.py              →  data/raw_<platform>_skill/<repo>.jsonl (new platforms)
consolidate.py            →  data/release/{diffs,diffs_clean,skills_initial,repos}.parquet
pr_metadata.py            →  data/pr_cache/<repo>.json              (cached PR fetch)
join_pr_metadata.py       →  adds pr_* columns to release parquets
add_licenses.py           →  adds license/stars to repos.parquet
enrich_v03.py             →  MinHash clustering + frontmatter validation
curator_subset.py         →  data/release/curator_training.parquet
skill_linter.py           →  rule-based defect detector (also a CLI tool)
eval_curator.py           →  held-out benchmark for skill-patch models
upload_hf.py              →  push to HuggingFace
```

## Reproducing the dataset

```bash
uv sync
uv run python fetch_huzey_repos.py
uv run python discover.py
uv run python discover_v04.py             # new platforms (Hermes, OpenCode, OpenClaw)
uv run python batch.py --workers 16       # ~5-6 hours wall time
uv run python batch_v04.py --repos data/hermes_repos.txt   --platform hermes_skill   --extractor skill_md --workers 16
uv run python batch_v04.py --repos data/opencode_repos.txt --platform opencode_skill --extractor skill_md --workers 16
uv run python consolidate_v04.py
uv run python pr_metadata.py --workers 4
uv run python join_pr_metadata.py
uv run python add_licenses.py --workers 8
uv run python enrich_v03.py
uv run python curator_subset.py
uv run python upload_hf.py
```

Each phase is resumable (manifest-based for batch jobs, per-repo cache for API fetches).

## Files

| File | Purpose |
|---|---|
| `extract.py` | Single-repo SKILL.md commit-history extractor |
| `extract_cursor.py` | Per-repo extractor for `.cursorrules` / `.cursor/rules/*.mdc` (deferred to v0.5) |
| `batch.py` | Parallel extraction across a repo list with per-repo manifest |
| `batch_v04.py` | Generalized batch runner — accepts `--platform` / `--extractor` for multi-format scraping |
| `discover.py` | Find Claude/Anthropic skill repos via GitHub repo + code search |
| `discover_v04.py` | Discovery for OpenCode / Hermes Agent / OpenClaw repos |
| `discover_cursor.py` | Discovery for Cursor rules repos (deferred to v0.5) |
| `classify.py` | Regex intent classifier for commit messages |
| `filter_quality.py` | Tag records with quality flags; produce clean subset |
| `consolidate.py` | Streaming JSONL → parquet with classify + filter applied inline |
| `consolidate_v04.py` | Multi-platform consolidate emitting per-format parquets with `platform` column |
| `pr_metadata.py` | Per-repo PR list fetch + cache; matches `head_sha` and `merge_commit_sha` |
| `join_pr_metadata.py` | Add PR columns to release parquets |
| `add_licenses.py` | SPDX license + stars + pushed_at metadata via gh API |
| `enrich_v03.py` | MinHash near-duplicate clustering + frontmatter validation + same-author dedup |
| `extract_bundled.py` | (v0.2) Capture sibling files (scripts/, references/) from skill folders |
| `merge_v04.py` | Recovery script: combine v0.3 release parquets + new platform data |
| `curator_subset.py` | Derive `curator_training.parquet` from the full corpus |
| `skill_linter.py` | Rule-based linter for SKILL.md (13 rules; CLI tool + report mode) |
| `eval_curator.py` | Held-out skill-patch eval harness; identity / intent_only / API-model adapters |

## Status

- **v0.4 (current)** — PR title+body metadata; multi-platform expansion (+ Hermes Agent + OpenCode); `curator_training.parquet` + skill linter + eval scaffold
- **v0.3** — MinHash skill clustering, frontmatter validation, same-author dedup, SPDX license metadata
- **v0.2** — bundled resources (skill folder sibling files) captured via tarball API
- **v0.1** — diff dataset with full LLM-augmented intent classification
- **v0.5 (planned)** — OpenClaw + Cursor corpus expansion (discovery completed, extraction deferred); embedding-based semantic clustering; bundled.parquet refresh for new platforms; PR-commit-list deep matching (currently only catches squash + head SHAs, ~10-20% record coverage)
