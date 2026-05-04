# Training a skill-edit / curator model

This is the handoff doc. You want to take this dataset and produce a small model that's better than generic aux models (Claude Haiku 4.5, Gemini Flash) at the skill-patch task. That model can then drop into [Hermes Agent's Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator) `auxiliary.curator` slot, or any equivalent skill-maintenance loop.

## TL;DR

```python
# 1. Load
from datasets import load_dataset
train = load_dataset("shl0ms/skill-diffs", "curator_training", split="train")
eval_ = load_dataset("shl0ms/skill-diffs", "curator_eval_set", split="train")

# 2. Format as instruction-tuning examples
def to_chat(ex):
    return {"messages": [
        {"role": "user", "content": (
            "You are editing a SKILL.md file for an AI agent. The maintainer "
            f'requested: "{ex["intent_text"]}"\n\n'
            "Produce the updated SKILL.md, applying the requested change. "
            "Output ONLY the file contents — no explanation, no code fences.\n\n"
            f"CURRENT SKILL.md:\n{ex['before_content']}"
        )},
        {"role": "assistant", "content": ex["after_content"]},
    ]}

train_chat = train.map(to_chat, remove_columns=train.column_names)
```

3. Fine-tune any 4B-7B base (Hermes-3, Llama-3, Mistral, etc.) on `train_chat` with axolotl/torchtune/whatever your stack is.
4. Eval with `eval_curator.py --model <provider:your-model>` (add an adapter to the script if your serving stack isn't OpenAI/Anthropic/OpenRouter API-compatible).

## The task

Given:
- `before_content` — the current SKILL.md file content (typically 1-5k chars)
- `intent_text` — the maintainer-stated description of what to change (PR title preferred, falls back to commit subject)

Produce:
- `after_content` — the patched SKILL.md

Stays SKILL.md format. Preserves YAML frontmatter, structure, and unrelated content. Only applies the intent.

> **Note on the "ground truth" AFTER:** The `after_content` in this corpus represents *edits that got merged into a public skill repo* — a mix of human-authored edits, LLM-authored edits (Claude, Copilot, Cursor, etc.), and human-AI-collaborative edits. We don't reliably distinguish authorship. ~49% of records that have a PR body show explicit AI-coauthor signatures, but this undercounts because most agent-assisted edits don't carry signatures. **The eval below measures relative imitation quality, not absolute correctness.** A strong score means "matches the merged-edit distribution"; it does *not* mean "matches a human expert's correct answer." For an objective correctness signal, see the `linter_delta` metric described below.

## Why a fine-tune is worth doing

Two reasons:

**1. Cost / latency.** A 7B fine-tune at ~1s + ~$0.001 per call replaces a generic aux model at ~30s + ~$0.04 per call. Even if the fine-tune merely *matches* a frontier model's score, the unit economics make it shippable for production Curator where the aux is invoked frequently in the background.

**2. Distribution match.** From the v0.5 baselines on the stratified 250-example eval (50 per intent class):

| Model | edit_dist_ratio | rouge_l | judge_overall (0-5) |
|---|---|---|---|
| `identity` (return BEFORE unchanged) | **0.8169** | **0.8596** | 1.00 |
| `intent_only` (return only intent) | 0.0047 | 0.0086 | 0.38 |
| `claude-haiku-4-5` | 0.7771 | 0.8311 | 2.08 |
| `claude-sonnet-4-5` | 0.7520 | 0.8187 | **2.30** |

**Bigger model → higher judge score, lower lexical match.** Sonnet wins judge_overall (judge correctly recognizes real edits as more valuable than no-op) but loses on edit_dist (frontier models over-rewrite). Haiku and Sonnet both *underperform identity on lexical metrics* — the corpus has a particular edit style that generic prompting doesn't reproduce.

A small fine-tune trained on the corpus's edit distribution can plausibly do both at once: hit identity-level edit_dist (>0.82) AND Sonnet-level judge_overall (>2.30). Neither generic aux currently does both.

> **What the eval does NOT claim.** It's not "humans wrote the gold AFTER, so your model should match human quality." It's *"this is what got merged into public skill repos (in practice often LLM-assisted), and your job is to match that distribution at lower inference cost."* Whether the merged edits are *correct* in an absolute sense is a separate question — see `linter_delta` for an objective correctness signal that doesn't depend on the gold being optimal.

## Recommended training setup

### Data

| Source | Records | Notes |
|---|---|---|
| `curator_training` config | 75,310 | Strict-clean + canonical + non-trivial intent. **Default training set.** |
| `curator_training_strict` config | 38,010 | Same plus SPDX license + no PII + no placeholders + engaged-repo. **Use this if you plan to publish a trained model** (license-clean for redistribution) |
| `curator_eval_set` config | 200 | Held-out (seed=42), already filtered for quality. **Don't train on this.** |
| `diffs_clean` config | 130,631 | Looser tier — includes records the curator subset filtered out. Use for ablations |

The 75k training records have ~3-5k chars of `before` + ~3-5k chars of `after` + ~30-200 chars of `intent_text`. Avg ~2-4k tokens per training example.

For DPO instead of SFT: the same `(before, after)` pairs work as `(rejected, chosen)`. The before is the pre-merge state, after is the merged version (authorship varies — see framing note above). Skip the intent for vanilla DPO; use it as the prompt for instruction-conditioned DPO.

### Filtering for higher quality (optional)

**Quick recommendation:** if you plan to publish a model trained on this data, start with `curator_training_strict` (38k records, license-clean) instead of `curator_training`.

The default 75k subset is permissive. Tighter filters worth considering:

```python
# Only records with PR title (richer intent labels)
with_pr = train.filter(lambda r: r["pr_title"] is not None)  # ~20k records

# Only Hermes-format skills (in-domain for Curator)
hermes_only = train.filter(lambda r: r["platform"] == "hermes_skill")  # ~5k

# Only edits that aren't massive rewrites (more focused patches)
small_edits = train.filter(
    lambda r: abs(len(r["after_content"]) - len(r["before_content"])) < 5000
)

# Only specific intent classes (drop docs/chore/refactor for fix+feat focus)
focused = train.filter(lambda r: r["intent_class"] in {"fix", "feat", "refactor"})

# v0.5: top quality records only (uses aggregate quality_score)
top = train.filter(lambda r: r["quality_score"] >= 0.7)  # ~9k highest-quality

# v0.5: dedupe across authors (semantic clustering catches forks + re-implementations)
unique = train.filter(lambda r: r["is_semantic_canonical"])

# v0.5: filter by edit type — e.g. only frontmatter fixes for narrow training
frontmatter_only = train.filter(
    lambda r: r["diff_summary"]["edit_kind"] == "frontmatter_only"
)
```

### v0.5 columns worth knowing about

| Column | Use |
|---|---|
| `skill_semantic_cluster_id` | Embedding-based cluster id. 47k unique clusters (vs MinHash's 175k). Catches **independent re-implementations** that MinHash misses. |
| `is_semantic_canonical` | True for the canonical skill in each semantic cluster. ~7.5% of records. Use this for the strictest cross-author dedup. |
| `diff_summary` | Struct with `edit_kind` (frontmatter_only / body_only / structural / code_only / both / trivial / addition / deletion) plus char counts and section deltas. Filter to specific edit types or compute structural metrics. |
| `quality_score` | Aggregate 0.0-1.0 score from license + stars + tags + intent + length signals. Lets you do `df.filter(quality_score >= 0.7)` for top-quartile (~9.6%) without writing custom logic. |
| `prompt_injection_pattern` (in `quality_tags`) | Advisory flag for content matching prompt-injection patterns (~0.27% of records, mostly defensive content in security skills). |

### Hyperparameters (starting points)

These are reasonable defaults — your team's standard SFT recipe is probably fine:

- **Base model:** Hermes-3-7B / Llama-3.1-8B / Mistral-7B-v0.3
- **Sequence length:** 8192 (most examples fit; truncate longer ones from the END of `before_content`, not the start — frontmatter is the highest-signal region)
- **Epochs:** 1-3 (start with 2)
- **Learning rate:** 2e-5 (SFT default)
- **Batch size:** 64 (or whatever your gradient-accumulation-aware total is)
- **Loss:** LM loss on assistant turn only (mask the user turn)

### Cost estimate

7B SFT on 75k examples × ~3k tokens × 2 epochs ≈ ~450M training tokens.

| Hardware | Approx wall time | Approx cost |
|---|---|---|
| 1× H100 80GB | 4-8 hrs | $15-50 rented |
| 1× A100 80GB | 8-15 hrs | $15-50 rented |
| 8× H100 cluster | 1-2 hrs | depends on infra |

## Eval

The eval harness lives in `eval_curator.py`. Built-in adapters for OpenAI / Anthropic / OpenRouter. Add an adapter for your serving stack:

```python
# in eval_curator.py around line 130, add:
def model_yourstack(model_id):
    def call(before, intent, **_):
        from your_client import Client
        client = Client(...)
        prompt = PROMPT_TEMPLATE.format(intent=intent, before=before)
        resp = client.complete(model=model_id, prompt=prompt, max_tokens=8000)
        return _strip_codefence(resp.text)
    return call

# and in resolve_model():
}[provider](model_id) if provider != "yourstack" else model_yourstack(model_id)
```

Then:

```bash
uv run python eval_curator.py --model yourstack:hermes-3-7b-curator-sft-v1
```

Output: per-metric scores + `data/eval_results/<timestamp>__yourstack_hermes-3-7b-curator-sft-v1.json` with predictions saved if `--output-predictions`.

### Bar to clear

Baselines on the v0.5 stratified eval set (250 examples, 50 per intent class):

**Imitation metrics** — "does the output match the merged-edit distribution?"

| Metric | identity | haiku | sonnet | What to aim for |
|---|---|---|---|---|
| `edit_dist_ratio` | **0.8169** | 0.7771 | 0.7520 | **>0.82 = beats identity** |
| `rouge_l` | **0.8596** | 0.8311 | 0.8187 | **>0.86 = beats identity** |
| `judge_overall` (Sonnet judge, 0-5) | 1.00 | 2.08 | **2.30** | **>2.3 = matches Sonnet ceiling** |
| `intent_only` floor | edit_dist 0.005, judge 0.38 — sanity floor only |

**Correctness metric** — independent of who authored the gold. Pure rule-based:

| Metric | identity | sonnet | What it means |
|---|---|---|---|
| `linter_delta` | +0.024 | -0.036 | `(# findings on gold) - (# findings on pred)`. **Positive = pred objectively cleaner than gold**. Identity is slightly cleaner than gold; Sonnet introduces a tiny number of new defects. |

**The fine-tune target:** simultaneously **edit_dist > 0.82** (matches identity, makes targeted edits) AND **judge_overall > 2.3** (matches Sonnet ceiling on faithfulness). Neither Haiku nor Sonnet does both — Haiku/Sonnet trade lexical match for judge quality. A small fine-tune trained on the corpus's edit distribution can plausibly do both at once.

The linter rules are deterministic (frontmatter validity, code-block language tags, deprecated model references like `gpt-3.5-turbo`, legacy API calls, placeholder content, etc.). A model that achieves `linter_delta > 0` produces output with FEWER defects than the merged-edit baseline — that's a real win regardless of how the gold was authored.

**Suggested optimization target:**
- Lexical metrics + judge: aim to *match* identity baseline while making real changes (keeps style consistent)
- `linter_delta`: aim to be slightly positive (improve on the gold's defect rate)

The lexical metrics reward *targeted* edits over *expansive* ones. Semantic cosine is saturated for most pairs (everything's close in embedding space because most edits are small) so don't read too much into single-decimal-point differences there.

## Common failure modes to watch for in eval

When you run your trained model and look at predictions:

1. **Over-rewriting** — model rephrases unchanged sections. → train more, lower temperature, prompt more strictly
2. **Frontmatter mangling** — model breaks YAML structure. → Add `quality_tags` filter to drop records with `invalid_frontmatter` (already in default subset)
3. **Refusal / safety overrides** — model wraps output in apologies. → Strip code fences (already handled in `_strip_codefence`); maybe filter base model
4. **Length explosion** — model produces 2x the gold length. → Inspect `pred_len / gold_len` ratio in eval results; cap `max_tokens`

The `output-predictions` flag saves predictions alongside scores so you can sample bad predictions and see what's going wrong.

## Where to look in the dataset

| Question | Where |
|---|---|
| What edit patterns appear in skills? | `intent_class` distribution + `pr_title` text in `curator_training.parquet` |
| What does "broken" frontmatter look like? | filter `quality_tags` containing `invalid_frontmatter` in `diffs.parquet` (these are excluded from curator subset but useful as negative examples) |
| What does Hermes-format skill look like? | filter `platform = hermes_skill` |
| What's a typical patch like? | `pq.read_table("curator_training.parquet").slice(0, 10).to_pylist()` |

## Pipeline scripts (for reference, not needed for training)

These rebuild the dataset from scratch. **You don't need to run them to train** — just `load_dataset("shl0ms/skill-diffs", "curator_training")`. They're documented in the root `README.md`.

## Questions, sanity checks

- Want to compare against more aux models (Sonnet, GPT-4o, Gemini Flash) before fine-tuning? `eval_curator.py --model anthropic:claude-sonnet-4-5` etc. ~$2-10 per 200-example run.
- Want to see specific predictions? `eval_curator.py --model anthropic:claude-haiku-4-5 --output-predictions --limit 10`
- Want to spot-check the dataset? `pq.read_table("curator_training.parquet").slice(0, 5).to_pandas()`
- Want to get the dataset as JSONL for non-pyarrow tooling? Use `datasets.load_dataset("shl0ms/skill-diffs", "curator_training", split="train").to_json("curator_training.jsonl")`

If a coworker hits something the docs don't cover, ping the original author (see `git log`).
