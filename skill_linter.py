#!/usr/bin/env python3
"""Rule-based linter for SKILL.md files.

Derived from the most common defect patterns observed across the skill-diffs
corpus (PR titles like "fix YAML frontmatter", "update outdated model names",
"add language tags to code blocks" reveal what skill maintainers actually fix in merged commits).

Designed to be:
  - Fast (no LLM, no clone, no network)
  - Easy to integrate into Hermes Agent's Curator pre-pass or skill_manage
    write-action validation
  - Honest about confidence: each rule has a `severity` (error/warning/info)
    and an explanation suitable for surfacing to the agent or user

Usage:
    # Lint a single skill file
    uv run python skill_linter.py path/to/SKILL.md

    # Lint a directory of skills
    uv run python skill_linter.py ~/.hermes/skills/

    # Generate a report over the v0.4 corpus to validate rule coverage
    uv run python skill_linter.py --report
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq


# === Frontmatter validation ===

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
NAME_RE = re.compile(r"^name\s*:\s*(.+?)\s*$", re.MULTILINE)
DESC_RE = re.compile(r"^description\s*:\s*(\S(?:.*\S)?)\s*$", re.MULTILINE)


# === Rule definitions ===

@dataclass
class Finding:
    rule: str
    severity: str   # error | warning | info
    message: str
    line: Optional[int] = None
    fix_hint: Optional[str] = None


def lint_no_frontmatter(content: str) -> List[Finding]:
    """E001: SKILL.md must start with YAML frontmatter."""
    if FRONTMATTER_RE.match(content):
        return []
    return [Finding(
        rule="E001-no-frontmatter",
        severity="error",
        message="Missing YAML frontmatter (must start with --- ... ---)",
        line=1,
        fix_hint="Add a frontmatter block with `name:` and `description:` fields",
    )]


def lint_frontmatter_fields(content: str) -> List[Finding]:
    """E002: Frontmatter must include name + description."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return []  # E001 already covers this
    fm = m.group(1)
    findings = []
    if not NAME_RE.search(fm):
        findings.append(Finding(
            rule="E002-missing-name",
            severity="error",
            message="Frontmatter missing `name:` field",
            line=1,
            fix_hint="Add `name: <skill-slug>` (kebab-case, matching the directory name)",
        ))
    if not DESC_RE.search(fm):
        findings.append(Finding(
            rule="E003-missing-description",
            severity="error",
            message="Frontmatter missing `description:` field",
            line=1,
            fix_hint="Add a `description:` line — when to invoke this skill",
        ))
    return findings


def lint_description_quality(content: str) -> List[Finding]:
    """W001-W003: description should be informative and clearly scoped."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return []
    fm = m.group(1)
    dm = DESC_RE.search(fm)
    if not dm:
        return []
    desc = dm.group(1).strip()
    findings = []
    # Strip surrounding quotes for length check
    bare = desc.strip("\"'")
    if bare.startswith("|"):  # YAML block scalar — read line below
        # Crude: just take next non-empty line
        lines = fm.split("\n")
        for i, ln in enumerate(lines):
            if ln.strip().startswith("description:"):
                # Get continuation lines
                cont = []
                for next_ln in lines[i+1:]:
                    if next_ln.startswith("  ") or next_ln.startswith("\t"):
                        cont.append(next_ln.strip())
                    else:
                        break
                bare = " ".join(cont)
                break
    if len(bare) < 20:
        findings.append(Finding(
            rule="W001-short-description",
            severity="warning",
            message=f"Description is very short ({len(bare)} chars). "
                    "Curator/skill discovery relies on this; consider expanding.",
            fix_hint="Aim for 1-3 sentences describing when to use this skill",
        ))
    if len(bare) > 1000:
        findings.append(Finding(
            rule="W002-long-description",
            severity="warning",
            message=f"Description is very long ({len(bare)} chars). "
                    "Frontmatter description should be a tagline, not the whole skill.",
            fix_hint="Move details into the body; keep description as a 1-3 sentence summary",
        ))
    # Heuristic: description should ideally include 'use this skill when' / 'for' phrasing
    triggers = ["use this", "use when", "when ", "for ", "to "]
    if not any(t in bare.lower() for t in triggers):
        findings.append(Finding(
            rule="W003-no-trigger-language",
            severity="warning",
            message="Description doesn't say WHEN to use this skill. "
                    "Convention: include 'use this skill when ...' or 'for ...'.",
            fix_hint="Rephrase: 'Use this skill when X' or 'For Y'",
        ))
    return findings


def lint_codeblock_languages(content: str) -> List[Finding]:
    """W004: code fences without a language tag."""
    findings = []
    # Match lines that are exactly ``` (no language)
    lines = content.split("\n")
    in_fence = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            tag = stripped[3:].strip()
            if not in_fence and not tag:
                findings.append(Finding(
                    rule="W004-codeblock-no-lang",
                    severity="warning",
                    message="Code fence missing language tag",
                    line=i,
                    fix_hint="Add a language: ```python, ```bash, ```yaml, ```text, etc.",
                ))
            in_fence = not in_fence
    return findings


# === Outdated content detection (curator-specific) ===

OUTDATED_PATTERNS = [
    # (pattern, replacement_hint, rule_name)
    (r"\bdeepseek[- ]?v3\.1\b", "DeepSeek V3.1 → live model name (deepseek/deepseek-chat-v3-0324)",
     "I001-outdated-model"),
    (r"\bgpt-3\.5-turbo\b", "gpt-3.5-turbo is largely deprecated; consider gpt-4o-mini or current",
     "I002-deprecated-model"),
    (r"\bclaude-2(?:\.[01])?\b", "Claude 2.x is deprecated; use claude-3.x or claude-4.x",
     "I003-deprecated-model"),
    (r"\bclaude-instant\b", "claude-instant is deprecated", "I004-deprecated-model"),
    (r"\btext-davinci-(?:002|003)\b", "text-davinci-* is fully deprecated", "I005-deprecated-model"),
    (r"\bcode-davinci-002\b", "code-davinci-002 is fully deprecated", "I006-deprecated-model"),
    (r"\bopenai\.Completion\.create\b", "openai.Completion is legacy; use chat.completions.create",
     "I007-legacy-api"),
    (r"\bopenai\.ChatCompletion\.create\b", "Pre-1.0 OpenAI SDK syntax; modern is openai.chat.completions.create",
     "I008-legacy-api"),
]


def lint_outdated_content(content: str) -> List[Finding]:
    findings = []
    lines = content.split("\n")
    for pattern_str, hint, rule in OUTDATED_PATTERNS:
        pat = re.compile(pattern_str, re.IGNORECASE)
        for i, line in enumerate(lines, 1):
            if pat.search(line):
                findings.append(Finding(
                    rule=rule,
                    severity="info",
                    message=f"Possibly outdated reference: {pat.pattern.strip(chr(92)+'b')}",
                    line=i,
                    fix_hint=hint,
                ))
                break  # only report first occurrence per pattern
    return findings


# === Heading + structure checks ===

def lint_no_h1_after_frontmatter(content: str) -> List[Finding]:
    """W005: should have an H1 (or H2) section near the top of the body."""
    m = FRONTMATTER_RE.match(content)
    body = content[m.end():] if m else content
    # Look at first 50 lines of body
    head = "\n".join(body.split("\n")[:50])
    if not re.search(r"^#{1,2}\s", head, re.MULTILINE):
        return [Finding(
            rule="W005-no-heading",
            severity="warning",
            message="Skill body has no top-level heading in first 50 lines",
            fix_hint="Add a `# Skill Name` heading at the top of the body",
        )]
    return []


def lint_skill_too_short(content: str) -> List[Finding]:
    """W006: skill content is suspiciously short."""
    m = FRONTMATTER_RE.match(content)
    body = content[m.end():] if m else content
    body_chars = len(body.strip())
    if body_chars < 200:
        return [Finding(
            rule="W006-short-body",
            severity="warning",
            message=f"Skill body is only {body_chars} chars. "
                    "Consider whether this needs more concrete instructions or examples.",
        )]
    return []


# === All rules ===

ALL_RULES = [
    lint_no_frontmatter,
    lint_frontmatter_fields,
    lint_description_quality,
    lint_codeblock_languages,
    lint_outdated_content,
    lint_no_h1_after_frontmatter,
    lint_skill_too_short,
]


def lint_content(content: str) -> List[Finding]:
    findings = []
    for rule in ALL_RULES:
        try:
            findings.extend(rule(content))
        except Exception as e:
            print(f"  (rule {rule.__name__} crashed: {e})", file=sys.stderr)
    return findings


def lint_path(path: Path) -> List[tuple]:
    """Return [(skill_path, [findings])] for one path or directory."""
    results = []
    if path.is_file():
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return [(str(path), [Finding(
                rule="E000-read-error", severity="error",
                message=f"Could not read file: {e}",
            )])]
        results.append((str(path), lint_content(content)))
    elif path.is_dir():
        # Find all SKILL.md
        for f in sorted(path.rglob("SKILL.md")):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                results.append((str(f), lint_content(content)))
            except Exception:
                continue
    return results


def print_findings(skill_path: str, findings: List[Finding]):
    if not findings:
        print(f"  [OK]  {skill_path}")
        return
    severity_marks = {"error": "[ERR]", "warning": "[WARN]", "info": "[INFO]"}
    print(f"\n{skill_path}:")
    for f in findings:
        mark = severity_marks.get(f.severity, "[?]")
        loc = f":{f.line}" if f.line else ""
        print(f"  {mark} {f.rule}{loc}: {f.message}")
        if f.fix_hint:
            print(f"        fix: {f.fix_hint}")


def report_over_corpus(release_dir: Path):
    """Run linter over skills_initial.parquet to validate rule coverage."""
    p = release_dir / "skills_initial.parquet"
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        return

    print(f"Reading {p}...", file=sys.stderr)
    t = pq.read_table(p, columns=["skill_id", "skill_name", "after_content"])
    print(f"  {t.num_rows:,} skills", file=sys.stderr)

    rule_counts = Counter()
    severity_counts = Counter()
    skills_with_issues = 0
    n = 0
    for sid, name, content in zip(
        t["skill_id"].to_pylist(),
        t["skill_name"].to_pylist(),
        t["after_content"].to_pylist(),
    ):
        n += 1
        if not content:
            continue
        findings = lint_content(content)
        if findings:
            skills_with_issues += 1
            for f in findings:
                rule_counts[f.rule] += 1
                severity_counts[f.severity] += 1
        if n % 10000 == 0:
            print(f"  [{n:,}/{t.num_rows:,}]", file=sys.stderr)

    print(f"\nLinted {n:,} skills, {skills_with_issues:,} with issues "
          f"({100*skills_with_issues/max(n,1):.1f}%)")
    print("\nFindings by severity:")
    for sev, c in severity_counts.most_common():
        print(f"  {sev:<10} {c:>10,}")
    print("\nTop rules triggered:")
    for rule, c in rule_counts.most_common():
        pct = 100 * c / max(n, 1)
        print(f"  {rule:<28} {c:>8,} ({pct:.1f}% of skills)")


def main():
    parser = argparse.ArgumentParser(description="Skill linter (rule-based, no LLM).")
    parser.add_argument("path", nargs="?", help="SKILL.md file or directory of skills")
    parser.add_argument("--report", action="store_true",
                        help="Run over the v0.4 corpus and print rule coverage stats")
    parser.add_argument("--release-dir", default="data/release")
    parser.add_argument("--json", action="store_true",
                        help="Output findings as JSON (machine-readable)")
    args = parser.parse_args()

    if args.report:
        report_over_corpus(Path(args.release_dir))
        return

    if not args.path:
        parser.print_help()
        sys.exit(1)

    p = Path(args.path).expanduser()
    if not p.exists():
        print(f"ERROR: {p} does not exist", file=sys.stderr)
        sys.exit(1)

    results = lint_path(p)
    if args.json:
        out = [
            {"path": sp, "findings": [asdict(f) for f in fs]}
            for sp, fs in results
        ]
        print(json.dumps(out, indent=2))
        return

    n_clean = sum(1 for _, fs in results if not fs)
    n_issues = len(results) - n_clean
    for sp, fs in results:
        print_findings(sp, fs)
    print()
    print(f"Linted {len(results)} skill(s): {n_clean} clean, {n_issues} with findings")


if __name__ == "__main__":
    main()
