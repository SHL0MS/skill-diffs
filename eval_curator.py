#!/usr/bin/env python3
"""Held-out eval for the curator skill-patch task.

Task definition:
    Given a skill (BEFORE) and a human-written intent label (commit_subject
    or PR title), produce the patched skill (AFTER).

Input:  curator_training.parquet -> sample N held-out triples
Output: per-model scores on
          - exact_match
          - char-level edit distance ratio (1 - distance/max_len)
          - rouge-L f-measure
          - cosine sim of bge-small-en-v1.5 embeddings (semantic)

Built-in baselines you can run with no API key:
    --model identity     return BEFORE unchanged (lower bound)
    --model intent_only  return only the intent text (sanity floor)

Real models — supplied via callable in --model-spec or via these adapters:
    --model openrouter:google/gemini-3-flash-preview
    --model openai:gpt-4o-mini
    --model anthropic:claude-3-5-haiku-latest

Usage:
    # 1. Sample held-out eval set (one-time, deterministic)
    uv run python eval_curator.py --sample-eval-set --n 200

    # 2. Run a baseline
    uv run python eval_curator.py --model identity

    # 3. Run a real model (requires API key in env or keychain)
    OPENAI_API_KEY=sk-... uv run python eval_curator.py --model openai:gpt-4o-mini
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


CURATOR_PARQUET = Path("data/release/curator_training.parquet")
EVAL_SET_PATH = Path("data/release/curator_eval_set.parquet")
RESULTS_DIR = Path("data/eval_results")


# === Sampling ===

def sample_eval_set(curator_path, n, seed):
    """Deterministic sample of N triples for eval. Runs once, then read-only."""
    import random
    print(f"Sampling {n} eval examples from {curator_path}...", file=sys.stderr)
    t = pq.read_table(curator_path)
    print(f"  {t.num_rows:,} candidates", file=sys.stderr)

    # Filter for higher quality:
    #   - has substantial before/after (not initial)
    #   - char_delta is not tiny (real edit) and not huge (not a rewrite)
    #   - intent_text long enough to be informative
    rows = t.to_pylist()

    def quality_filter(r):
        before = r.get("before_content") or ""
        after = r.get("after_content") or ""
        intent = r.get("intent_text") or ""
        if len(before) < 200 or len(after) < 200:
            return False
        if abs(len(after) - len(before)) > 20000:  # massive rewrites are noisy
            return False
        if len(intent) < 12:
            return False
        # Don't sample skills that became trivially short after edit
        if len(after) < 0.3 * len(before):
            return False
        return True

    rows = [r for r in rows if quality_filter(r)]
    print(f"  {len(rows):,} pass quality filter", file=sys.stderr)

    if len(rows) < n:
        print(f"  WARNING: only {len(rows)} eligible, returning all", file=sys.stderr)
        sample = rows
    else:
        rng = random.Random(seed)
        sample = rng.sample(rows, n)

    # Strip down to needed columns
    out = []
    for r in sample:
        out.append({
            "pair_id": r.get("pair_id"),
            "skill_id": r.get("skill_id"),
            "repo": r.get("repo"),
            "skill_name": r.get("skill_name"),
            "intent_text": r.get("intent_text"),
            "commit_subject": r.get("commit_subject"),
            "pr_title": r.get("pr_title"),
            "pr_body": r.get("pr_body"),
            "before_content": r.get("before_content"),
            "after_content": r.get("after_content"),
            "intent_class": r.get("intent_class"),
            "platform": r.get("platform"),
            "license_spdx": r.get("license_spdx"),
        })
    return out


# === Model adapters ===

def model_identity(before: str, intent: str, **_) -> str:
    """Lower bound: return BEFORE unchanged."""
    return before


def model_intent_only(before: str, intent: str, **_) -> str:
    """Sanity floor: return only the intent text."""
    return intent


PROMPT_TEMPLATE = """You are editing a SKILL.md file for an AI agent.

Below is the current content of the SKILL.md file. The maintainer requested:
"{intent}"

Produce the updated SKILL.md file in full, applying the requested change.
Output ONLY the file contents, with no explanation, no surrounding markdown
code fences, and no commentary. Preserve YAML frontmatter, file structure,
and any unrelated content exactly.

CURRENT SKILL.md:
{before}
"""


def _strip_codefence(s: str) -> str:
    """Some models wrap output in ```...``` even when told not to."""
    s = s.strip()
    if s.startswith("```"):
        # Drop first line (```lang) and last fence
        lines = s.split("\n")
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        s = "\n".join(lines)
    return s


def model_openai(model_id):
    def call(before, intent, **_):
        from openai import OpenAI
        api_key = _keychain_get("openai", "API_KEY") \
                  or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        prompt = PROMPT_TEMPLATE.format(intent=intent, before=before)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8000,
        )
        return _strip_codefence(resp.choices[0].message.content)
    return call


def _keychain_get(service, account):
    try:
        return subprocess.run(
            ["security", "find-generic-password",
             "-a", account, "-s", service, "-w"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def model_anthropic(model_id):
    def call(before, intent, **_):
        from anthropic import Anthropic
        # Prefer keychain over env var so callers don't need to inline the key
        api_key = _keychain_get("ANTHROPIC_WORK", "opencode") \
                  or _keychain_get("anthropic", "API_KEY") \
                  or os.environ.get("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        prompt = PROMPT_TEMPLATE.format(intent=intent, before=before)
        resp = client.messages.create(
            model=model_id,
            max_tokens=8000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return _strip_codefence(resp.content[0].text)
    return call


def model_openrouter(model_id):
    def call(before, intent, **_):
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            try:
                api_key = subprocess.run(
                    ["security", "find-generic-password", "-a", "openrouter",
                     "-s", "API_KEY", "-w"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
            except Exception:
                pass
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or "missing",
        )
        prompt = PROMPT_TEMPLATE.format(intent=intent, before=before)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8000,
        )
        return _strip_codefence(resp.choices[0].message.content)
    return call


def resolve_model(spec):
    """Return a callable (before, intent) -> predicted_after."""
    if spec == "identity":
        return model_identity
    if spec == "intent_only":
        return model_intent_only
    if ":" not in spec:
        raise ValueError(f"unknown model spec: {spec}")
    provider, model_id = spec.split(":", 1)
    return {
        "openai": model_openai,
        "anthropic": model_anthropic,
        "openrouter": model_openrouter,
    }[provider](model_id)


# === Metrics ===

def metric_exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def metric_edit_distance_ratio(pred: str, gold: str) -> float:
    """1 - levenshtein / max_len. Clamped to [0, 1]. Higher = better."""
    # Use python-Levenshtein-style fast distance via difflib for portability
    import difflib
    matcher = difflib.SequenceMatcher(None, pred, gold)
    return matcher.ratio()


def metric_rouge_l(pred: str, gold: str) -> float:
    """Token-level ROUGE-L F1. Unigram tokens on whitespace+punct."""
    def tokens(s):
        return re.findall(r"\w+", s.lower())
    p_tok = tokens(pred)
    g_tok = tokens(gold)
    if not p_tok or not g_tok:
        return 0.0
    # Longest common subsequence length
    n, m = len(p_tok), len(g_tok)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if p_tok[i] == g_tok[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[n][m]
    p_recall = lcs / m
    p_prec = lcs / n
    if p_recall + p_prec == 0:
        return 0.0
    return 2 * p_recall * p_prec / (p_recall + p_prec)


_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu"))
        _ENCODER = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    return _ENCODER


def metric_semantic_cosine(pred: str, gold: str) -> float:
    """Cosine similarity of bge-small-en-v1.5 embeddings."""
    enc = _get_encoder()
    embs = enc.encode([pred[:4000], gold[:4000]], normalize_embeddings=True)
    return float((embs[0] * embs[1]).sum())


METRICS = [
    ("exact_match", metric_exact_match),
    ("edit_dist_ratio", metric_edit_distance_ratio),
    ("rouge_l", metric_rouge_l),
    ("semantic_cosine", metric_semantic_cosine),
]


# === Eval loop ===

@dataclass
class Result:
    pair_id: str
    repo: str
    skill_name: str
    intent_text: str
    metrics: dict
    pred_len: int
    gold_len: int
    error: Optional[str] = None


def run_eval(eval_set, model_fn, model_label, output_predictions=False, limit=None):
    if limit:
        eval_set = eval_set[:limit]
    results = []
    started = time.time()
    for i, ex in enumerate(eval_set, 1):
        try:
            pred = model_fn(ex["before_content"], ex["intent_text"])
        except Exception as e:
            pred = ""
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  [{i}/{len(eval_set)}] ERROR {ex['repo']}: {err}",
                  file=sys.stderr)
            results.append(Result(
                pair_id=ex["pair_id"], repo=ex["repo"],
                skill_name=ex.get("skill_name") or "",
                intent_text=ex.get("intent_text") or "",
                metrics={m: 0.0 for m, _ in METRICS},
                pred_len=0, gold_len=len(ex["after_content"]),
                error=err,
            ))
            continue
        gold = ex["after_content"]
        scores = {}
        for name, fn in METRICS:
            try:
                scores[name] = fn(pred, gold)
            except Exception as e:
                scores[name] = float("nan")
                print(f"    metric {name} crashed: {e}", file=sys.stderr)
        results.append(Result(
            pair_id=ex["pair_id"], repo=ex["repo"],
            skill_name=ex.get("skill_name") or "",
            intent_text=ex.get("intent_text") or "",
            metrics=scores,
            pred_len=len(pred), gold_len=len(gold),
        ))

        if output_predictions:
            # Store predicted text alongside
            results[-1].__dict__["prediction"] = pred

        if i % 10 == 0 or i == len(eval_set):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(eval_set) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(eval_set)}] {rate:.1f}/s eta={int(eta)}s",
                  file=sys.stderr)

    # Aggregate
    n = len([r for r in results if r.error is None])
    summary = {"n_eval": len(results), "n_ok": n, "model": model_label}
    for name, _ in METRICS:
        vals = [r.metrics[name] for r in results if r.error is None]
        if vals:
            avg = sum(vals) / len(vals)
            summary[f"{name}_mean"] = round(avg, 4)
        else:
            summary[f"{name}_mean"] = None

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Curator skill-patch eval.")
    parser.add_argument("--model", default="identity",
                        help="Model spec: identity | intent_only | "
                             "openai:<id> | anthropic:<id> | openrouter:<id>")
    parser.add_argument("--sample-eval-set", action="store_true",
                        help="Sample a fresh eval set from curator_training.parquet")
    parser.add_argument("--n", type=int, default=200,
                        help="Eval set size when --sample-eval-set is used")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curator-parquet", default=str(CURATOR_PARQUET))
    parser.add_argument("--eval-set", default=str(EVAL_SET_PATH))
    parser.add_argument("--output-predictions", action="store_true",
                        help="Save predictions alongside scores")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run on first N eval items only (for smoke tests)")
    args = parser.parse_args()

    if args.sample_eval_set:
        sample = sample_eval_set(args.curator_parquet, args.n, args.seed)
        if not sample:
            print("ERROR: empty sample", file=sys.stderr)
            sys.exit(1)
        # Write as parquet
        # Need a unified schema — use first row as template
        t = pa.Table.from_pylist(sample)
        pq.write_table(t, args.eval_set, compression="zstd")
        print(f"Wrote eval set: {args.eval_set}  ({t.num_rows:,} examples)",
              file=sys.stderr)
        return

    if not Path(args.eval_set).exists():
        print(f"ERROR: {args.eval_set} not found. Run with --sample-eval-set first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading eval set: {args.eval_set}", file=sys.stderr)
    eval_set = pq.read_table(args.eval_set).to_pylist()
    print(f"  {len(eval_set):,} examples", file=sys.stderr)

    print(f"Resolving model: {args.model}", file=sys.stderr)
    model_fn = resolve_model(args.model)

    print(f"\nRunning eval...", file=sys.stderr)
    summary, results = run_eval(
        eval_set, model_fn, args.model,
        output_predictions=args.output_predictions,
        limit=args.limit,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Model: {summary['model']}")
    print(f"  N: {summary['n_eval']} (ok: {summary['n_ok']})")
    print()
    for name, _ in METRICS:
        v = summary.get(f"{name}_mean")
        if v is None:
            print(f"  {name:<20} —")
        else:
            print(f"  {name:<20} {v:.4f}")
    print("=" * 60)

    # Save full results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace(":", "_").replace("/", "_")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{ts}__{safe_name}.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "results": [asdict(r) for r in results],
        "model": args.model,
        "timestamp": ts,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
