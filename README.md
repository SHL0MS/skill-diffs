# skill-diffs

Pipeline that scrapes commit histories of agent skills (`SKILL.md` files) from public GitHub repos and packages them as a dataset of (before, after, intent) diff pairs for training and evaluation.

> **Just want to fine-tune a curator/skill-edit model?** Read [TRAINING.md](TRAINING.md). You don't need this pipeline — just `load_dataset("shl0ms/skill-diffs", "curator_training")`.

## What's in the dataset

Published at **[`shl0ms/skill-diffs`](https://huggingface.co/datasets/shl0ms/skill-diffs)** on HuggingFace.

The May 2026 snapshot covers **4 platforms** (Anthropic Claude, OpenClaw, OpenCode, Hermes Agent):

- **5,891 source repos** with SPDX license metadata
- **664,872 unique skills** total — MinHash-clustered for near-duplicate dedup
- **986,515 total records** (every commit-by-commit revision across all platforms)
- **130,631 clean diff pairs** (default tier — ~85x larger than `huzey/claude-skills-diff`)
- **75,310 records in `curator_training.parquet`** — recommended default training subset for skill-edit / curator models (strict-clean + canonical + non-trivial intent)
- **38,010 records in `curator_training_strict.parquet`** — stricter tier that also requires SPDX license + no PII + no placeholder content + engaged-repo signal. Use this if you plan to publish a model trained on the data.
- **76,142 records with PR title + body** as intent labels (7.7% full / 18.8% of the clean tier — richer than commit subjects alone)
- **630,119 bundled-resource snapshots** with sibling files (scripts/, references/, assets/) — v0.5 covers all 4 platforms (415k Anthropic + 215k new platforms)

See `data/release/README.md` for the full data card.

```python
from datasets import load_dataset

# Default clean tier (all 3 platforms)
diffs = load_dataset("shl0ms/skill-diffs", "diffs_clean", split="train")

# Filter to one platform (claude_skill / opencode_skill / hermes_skill / openclaw_skill)
hermes = diffs.filter(lambda r: r["platform"] == "hermes_skill")

# The curator-recommended training subset (pre-filtered)
curator = load_dataset("shl0ms/skill-diffs", "curator_training", split="train")

# Stricter variant — license-known, PII-filtered, no placeholders, engaged repos
strict = load_dataset("shl0ms/skill-diffs", "curator_training_strict", split="train")

# Bundled skill-folder context (sibling files)
bundled = load_dataset("shl0ms/skill-diffs", "bundled", split="train")
```

## Why

Agent skills are a rare kind of training data: structured procedural specs that get iteratively refined through merged commits in public repos. Authorship is heterogeneous (humans, agents like Claude Code / Cursor / Copilot, and human-AI collaborations) — we don't distinguish. Existing public datasets capture *snapshots*; this one captures the **evolution** — every commit-by-commit revision with diffs, commit messages, **PR titles + bodies (where available)**, and authorship metadata.

Use cases:

- **Skill-editor / Curator model training** — see `curator_training.parquet`. Designed for fine-tuning a model that takes `(before_skill, intent_text)` and produces the patched skill. Drops in as the LLM review pass for [Hermes Agent's Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator) or any equivalent maintenance loop.
- **DPO / preference-pair training** — `(before, after)` where `after` is the merged version (authorship varies)
- **Pattern mining** — what kinds of edits are most common in skill iteration (frontmatter fixes, model name updates, code-block language tags)
- **Initial-state generation** — `skills_initial.parquet` for "create a skill from scratch" training
- **Cross-platform analysis** — `platform` column lets you compare conventions between Anthropic, OpenClaw, OpenCode, and Hermes Agent skill formats

## Companion tools

Two complementary scripts ship alongside the dataset:

- **`skill_linter.py`** — rule-based linter (no LLM, no clone, no network) covering 13 patterns derived from observed defects: missing or incomplete frontmatter, missing code-block languages, deprecated model references (e.g. `gpt-3.5-turbo`, `claude-2.x`), legacy API calls (`openai.ChatCompletion`), weak/long descriptions. Validated against 665k skills (61% have at least one finding).
- **`eval_curator.py`** — held-out eval scaffold for benchmarking models on the skill-patch task: given `(before, intent_text)`, produce the patched skill. Built-in baselines (`identity`, `intent_only`) plus adapters for OpenAI, Anthropic, OpenRouter. Metrics: exact_match, edit_distance_ratio, ROUGE-L, BAAI/bge-small-en-v1.5 cosine similarity.

## Pipeline

```
fetch_huzey_repos.py      →  data/huzey_repos.txt
discover.py               →  data/expansion_repos.txt
discover_v04.py           →  data/{opencode,hermes,openclaw}_repos.txt
batch.py                  →  data/raw/<repo>.jsonl                  (Anthropic corpus, legacy)
batch_v04.py              →  data/raw_<platform>_skill/<repo>.jsonl (new platforms, with --timeout)
consolidate_v04.py        →  data/release/{diffs,diffs_clean,skills_initial,repos}.parquet
add_platform.py           →  incrementally append a new platform's raw JSONL into existing release
pr_metadata.py            →  data/pr_cache/<repo>.json              (cached PR fetch)
join_pr_metadata.py       →  adds pr_* columns to release parquets
add_licenses.py           →  adds license/stars to repos.parquet (idempotent)
enrich_v03.py             →  MinHash clustering + frontmatter validation (idempotent)
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
uv run python batch_v04.py --repos data/openclaw_repos.txt --platform openclaw_skill --extractor skill_md --workers 16
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

### v0.5 current (use these)

| File | Purpose |
|---|---|
| `extract.py` | Single-repo SKILL.md commit-history extractor with per-repo `timeout` (default 30 min) |
| `batch_v04.py` | Generalized batch runner — accepts `--platform` / `--extractor` / `--timeout` for multi-format scraping |
| `add_platform.py` | Incrementally append a new platform's raw JSONL into existing release parquets (used to add OpenClaw to v0.4 → v0.4.1) |
| `discover.py` | Find Claude/Anthropic skill repos via GitHub repo + code search |
| `discover_v04.py` | Discovery for OpenCode / Hermes Agent / OpenClaw repos |
| `consolidate_v04.py` | Multi-platform consolidate emitting per-format parquets with `platform` column |
| `pr_metadata.py` | Per-repo PR list fetch + cache; matches `head_sha` and `merge_commit_sha` |
| `join_pr_metadata.py` | Add PR columns to release parquets |
| `add_licenses.py` | SPDX license + stars + pushed_at metadata via gh API |
| `enrich_v03.py` | MinHash near-duplicate clustering + frontmatter validation + same-author dedup (kept name from when it was new in v0.3) |
| `extract_bundled.py` | Capture sibling files (scripts/, references/) from skill folders (v0.3 only — needs v0.5 refresh) |
| `curator_subset.py` | Derive `curator_training.parquet` from the full corpus (default + `--strict` mode for the strict variant) |
| `add_quality_v041.py` | Apply 4 v0.4.2 quality tags (no_license, low_engagement, placeholder_content, pii_email) to release parquets. Idempotent. |
| `add_semantic_diff.py` | (v0.5) Add structured `diff_summary` column with edit_kind taxonomy. Idempotent. |
| `add_injection_tag.py` | (v0.5) Flag prompt-injection-style content with `prompt_injection_pattern` quality tag. Idempotent. |
| `add_quality_score.py` | (v0.5) Add aggregate `quality_score` (0.0-1.0) derived from license + stars + tags + intent + length. Must run last. Idempotent. |
| `embed_cluster.py` | (v0.5) BAAI/bge-small-en-v1.5 embeddings + FAISS cosine clustering at 0.85 threshold. Outputs `data/embeddings.parquet` + `data/semantic_clusters.parquet` |
| `add_semantic_clusters.py` | (v0.5) Merge `skill_semantic_cluster_id` + `is_semantic_canonical` into release parquets |
| `build_eval_set.py` | (v0.5) Build stratified `curator_eval_set_v2.parquet` (50 per intent × 5 classes = 250) |
| `skill_linter.py` | Rule-based linter for SKILL.md (13 rules; CLI tool + report mode) |
| `eval_curator.py` | Held-out skill-patch eval harness; identity / intent_only / API-model adapters |
| `finish_v04.sh` | Orchestrates the full enrichment chain after batch jobs finish |
| `monitor.sh` | Live progress monitor for long-running pipeline runs |
| `upload_hf.py` | Push release parquets to HuggingFace |

### Legacy / superseded (kept for reproducing earlier versions)

| File | Notes |
|---|---|
| `batch.py` | v0.3-era batch runner; superseded by `batch_v04.py` (which has a `--platform` arg + per-platform manifests) |
| `consolidate.py` | v0.3-era consolidate; superseded by `consolidate_v04.py` |
| `classify.py`, `filter_quality.py` | Imported by both `consolidate.py` and `consolidate_v04.py` — still used |
| `llm_classify.py` | Stand-alone Claude Haiku 4.5 batch classifier; only run once, results in `data/llm_classifications.json` |
| `aggregate_bundled.py`, `analyze.py`, `build_dataset.py` | Earlier exploratory scripts; not on the v0.4 path |
| `fetch_huzey_repos.py` | One-off seed list fetch from `huzey/claude-skills` |

### Deferred to v0.5 (written but not run)

| File | Notes |
|---|---|
| `extract_cursor.py` | Cursor rules extractor (`.cursorrules`, `.cursor/rules/*.mdc`) |
| `discover_cursor.py` | Cursor rules discovery |
| `embed_cluster.py` | Embedding-based semantic clustering (BAAI/bge-small-en-v1.5) |
| `add_semantic_clusters.py` | Merge semantic clusters into release parquets |

### One-off recovery

| File | Notes |
|---|---|
| `merge_v04.py` | Recovery script used once to reconstruct v0.4 from `data/v03_backup/` (downloaded from HF) + new platform data after `data/raw/` was missing. Don't run unless reproducing the same recovery. |
| `finish_openclaw.sh` | Orchestrator used once to integrate the OpenClaw scrape into v0.4.1. Pattern is reusable for future incremental platform additions. |

## Status

- **v0.5 (current)** — semantic clustering (BAAI/bge-small-en-v1.5; 47k unique clusters catching cross-author dups MinHash misses); `diff_summary` structural edit-type column (`frontmatter_only` / `structural` / `body_only` / etc.); aggregate `quality_score` column; `prompt_injection_pattern` advisory tag; bundled.parquet refresh covering all 4 platforms; stratified eval set (`curator_eval_set_v2.parquet`); LLM-as-judge + linter_delta correctness metric; honest reframing that the gold AFTER is the merged-edit (not "human-written")
- **v0.4.2** — 4 new `quality_tags` (`no_license`, `low_engagement`, `placeholder_content`, `pii_email`); `curator_training_strict.parquet` (38k records) for redistribution-safe / publishable training
- **v0.4.1** — adds OpenClaw platform (1,631 repos, +18k clean diff pairs); per-repo timeout in `extract.py` to prevent monorepo straggler hangs
- **v0.4** — PR title+body metadata; multi-platform expansion (Anthropic + Hermes Agent + OpenCode); `curator_training.parquet` + skill linter + eval scaffold
- **v0.3** — MinHash skill clustering, frontmatter validation, same-author dedup, SPDX license metadata
- **v0.2** — bundled resources (skill folder sibling files) captured via tarball API
- **v0.1** — diff dataset with full LLM-augmented intent classification
- **v0.6 (planned)** — Cursor corpus extraction (discovery completed in v0.4.1); PR-commit-list deep matching (currently only catches squash + head SHAs, achieving 7.7% full / 18.8% clean-tier coverage); refined task-specific intent labels via LLM relabeling (e.g. `frontmatter_fix`, `outdated_model_update`, `add_section`); consolidations.parquet derived from cluster history; analysis notebook
