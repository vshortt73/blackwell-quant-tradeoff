"""
Validate data/tasks.jsonl before spending GPU time on it.

Two modes:

  static  -- schema, formatting, duplicates, and the constraints the grader and
             the completions endpoint impose.
  --probe -- additionally runs the set against a live server and reports the
             DIFFICULTY CALIBRATION, which is the design decision that most
             often ruins an accuracy anchor.

On calibration: a task set the baseline answers PERFECTLY cannot detect
degradation -- every arm scores 1.0. A set it answers at chance is equally
useless, since misses then reflect prompt format rather than capability.

But do not chase a low baseline blindly. Measured on Qwen3-8B/BF16, this model
answers essentially all short-answer factual and computational items
correctly -- obscure entities (Thimphu, Borrelia burgdorferi, 1729), precise
constants (299792458, 5730, 44.1), and arithmetic (53x47, gcd(462,1071)) alike.
Its only reliable failures are facts postdating its training cutoff. Forcing a
0.70-0.85 baseline on such a model means loading the set with post-cutoff tech
trivia, which re-clusters every discriminating item into one domain and breaks
the independence McNemar assumes.

So: 0.70-0.85 is ideal when reachable with a BROAD set. When it is not, prefer
breadth at a higher baseline and add items -- McNemar needs discordant pairs,
and those come from items near the decision boundary, which exist at 0.92 too.
There are simply fewer per hundred, so raise n.

Usage:
    python harness/check_tasks.py data/tasks.jsonl
    python harness/check_tasks.py data/tasks.jsonl --probe --config configs/bf16_baseline.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader import grade  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_file")
    ap.add_argument("--probe", action="store_true", help="run against a live server")
    ap.add_argument("--config", default="configs/bf16_baseline.yaml")
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N items")
    args = ap.parse_args()

    path = Path(args.task_file)
    if not path.exists():
        print(f"[FAIL] {path} does not exist")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    items = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            it = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: not valid JSON ({e.msg})")
            continue
        if not isinstance(it, dict):
            errors.append(f"line {lineno}: must be a JSON object")
            continue
        for k in ("prompt", "answer"):
            if k not in it:
                errors.append(f"line {lineno}: missing required key '{k}'")
        if "aliases" in it and not isinstance(it["aliases"], list):
            errors.append(f"line {lineno}: 'aliases' must be a list")
        items.append((lineno, it))

    if not items:
        print("[FAIL] no items parsed")
        return 1

    # --- constraints the endpoint and grader impose ------------------------ #
    for lineno, it in items:
        p = str(it.get("prompt", ""))
        a = str(it.get("answer", ""))
        if p.rstrip().endswith("?"):
            warnings.append(
                f"line {lineno}: prompt ends with '?'. This hits /v1/completions "
                "with no chat template, so a question invites the model to "
                "continue with MORE questions. Prefer a stem: "
                "'The capital of France is'")
        if p != p.rstrip():
            warnings.append(
                f"line {lineno}: prompt has trailing whitespace; the model "
                "continues directly from it, so this changes tokenization")
        if not a.strip():
            errors.append(f"line {lineno}: empty answer")
        if len(a.split()) > 8 and not it.get("whole_response"):
            warnings.append(
                f"line {lineno}: answer is {len(a.split())} words. The grader "
                "scopes to the first sentence and max_tokens is 32 -- long "
                "answers may not fit. Consider 'whole_response': true")
        # a task whose answer already appears in the prompt is not a test
        if a.strip().lower() in p.lower():
            warnings.append(f"line {lineno}: answer appears in the prompt itself")

    prompts = [str(it.get("prompt", "")) for _, it in items]
    dupes = [p for p, n in collections.Counter(prompts).items() if n > 1]
    if dupes:
        warnings.append(f"{len(dupes)} duplicate prompt(s); paired analysis treats them as separate items")

    print(f"=== {path} ===")
    print(f"  items parsed : {len(items)}")
    print(f"  errors       : {len(errors)}")
    print(f"  warnings     : {len(warnings)}")
    for e in errors[:15]:
        print(f"    [ERROR] {e}")
    for w in warnings[:15]:
        print(f"    [warn]  {w}")
    if len(warnings) > 15:
        print(f"    ... and {len(warnings) - 15} more warnings")

    n = len(items)
    if n < 300:
        print(f"\n  [SIZE] {n} items. McNemar works on DISCORDANT pairs only; if a "
              f"scheme flips ~5% you would see ~{max(1, int(n * 0.05))} of them. "
              "300-1000 items is the comfortable range.")

    if errors:
        print("\n[FAIL] fix errors before running")
        return 1

    if not args.probe:
        print("\n[OK] static checks passed. Re-run with --probe against a live "
              "server to check difficulty calibration.")
        return 0

    # --- difficulty probe -------------------------------------------------- #
    import requests

    from common import load_config

    merged = load_config(args.config)
    base_url = merged["base_url"]
    model = merged["served_model_name"]
    max_tokens = merged.get("quality", {}).get("max_tokens", 32)
    few_shot = merged.get("quality", {}).get("few_shot", "")

    probe = items[: args.limit] if args.limit else items
    correct = 0
    rules: collections.Counter = collections.Counter()
    misses: list[tuple[int, str]] = []
    for lineno, it in probe:
        r = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json={"model": model, "prompt": few_shot + it["prompt"],
                  "max_tokens": max_tokens, "temperature": 0.0},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["text"]
        g = grade(text, it["answer"], it.get("aliases", ()),
                  whole_response=bool(it.get("whole_response", False)))
        correct += int(g.correct)
        rules[g.rule] += 1
        if not g.correct and len(misses) < 10:
            misses.append((lineno, g.rule))

    acc = correct / len(probe)
    print(f"\n=== difficulty probe ({len(probe)} items, {args.config}) ===")
    print(f"  baseline accuracy : {acc:.3f}  ({correct}/{len(probe)})")
    print(f"  deciding rules    : {dict(rules)}")
    if acc >= 0.98:
        print("\n  [CEILING] the baseline answers essentially everything. Add harder "
              "items, or if the model has no reachable competence edge (common for "
              "strong models on short-answer recall), raise n instead so enough "
              "boundary items exist to flip.")
    elif acc >= 0.90:
        print(f"\n  [HIGH] baseline {acc:.2f}. Usable, but only ~{int(len(probe)*(1-acc))} "
              "items per " f"{len(probe)} sit near the boundary where flips happen. "
              "Scale n so discordant pairs are not vanishingly rare.")
    elif acc <= 0.40:
        print("\n  [FLOOR] the baseline fails most items, so degradation has "
              "little room to show and misses may reflect prompt format rather "
              "than capability. Check the misses below, then make items easier.")
    elif 0.70 <= acc <= 0.85:
        print("\n  [GOOD] in the discriminating band -- the model is at the edge "
              "of its competence, where perturbations flip answers.")
    else:
        print("\n  [OK] usable, though 0.70-0.85 discriminates best.")
    if misses:
        print("  first misses (line, deciding rule):")
        for ln, rule in misses:
            print(f"    line {ln}: {rule}")
        print("  inspect full text in results/audit/ after a real run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
