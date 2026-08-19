"""
External-anchor quality eval -- the grounded counterpart to APEX.

Purpose: give a reviewer who does not yet trust APEX two accepted signals, so
APEX becomes the mechanistic explanation of WHY these moved and WHICH dimension
drove it, rather than the sole quality claim.

Two anchors:
  1. task_accuracy  -- exact-match / choice accuracy on a controlled, held-out
                       set. Prefer a domain set YOU own over public GSM8K/MMLU
                       subsets (less contamination risk, more defensible).
  2. perplexity     -- held-out corpus perplexity via prompt_logprobs from the
                       vLLM OpenAI endpoint. Cheap continuous proxy; good for
                       catching degradation the discrete accuracy metric misses.

CRITICAL CONTROL: everything here is greedy-decoded (temperature = 0). Sampling
noise would otherwise contaminate a quantization comparison and you could not
attribute a delta to the quant scheme vs the RNG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from grader import GRADER_VERSION, grade

import yaml

try:
    import requests  # pip install requests
except ImportError as e:  # pragma: no cover
    raise SystemExit("requests required: pip install requests") from e

from common import RunResult


def _completion(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    r = requests.post(
        f"{base_url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy -- non-negotiable for this comparison
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def task_accuracy(base_url: str, model: str, task_file: str, max_tokens: int) -> dict[str, Any]:
    """task_file: JSONL of {"prompt":..., "answer":..., "aliases":[...] (opt),
    "whole_response": bool (opt)}.

    Graded on GOAL SATISFACTION, not string equality -- a response that answers
    the question and then elaborates has given the user what they asked for.
    See harness/grader.py for the rules and why they are deterministic.

    Returns the per-rule breakdown alongside the score. Which rule decided each
    item is what makes an accuracy number auditable rather than a bare float,
    and a shift in the rule mix across schemes is itself a degradation signal
    (e.g. answers drifting from first-segment to buried, or into negation)."""
    items = [json.loads(l) for l in Path(task_file).read_text().splitlines() if l.strip()]
    if not items:
        raise SystemExit(f"no task items in {task_file}")
    correct = 0
    rule_counts: dict[str, int] = {}
    per_item: list[dict[str, Any]] = []
    for it in items:
        out = _completion(base_url, model, it["prompt"], max_tokens)
        text = out["choices"][0]["text"]
        g = grade(
            text,
            it["answer"],
            it.get("aliases", ()),
            whole_response=bool(it.get("whole_response", False)),
        )
        correct += int(g.correct)
        rule_counts[g.rule] = rule_counts.get(g.rule, 0) + 1
        per_item.append({"prompt": it["prompt"], "answer": it["answer"], "response": text, **g.as_dict()})
    return {
        "n": len(items),
        "correct": correct,
        "accuracy": correct / len(items),
        "grader_version": GRADER_VERSION,
        "rule_counts": rule_counts,
        "per_item": per_item,
    }


def perplexity(base_url: str, model: str, corpus_file: str) -> dict[str, Any]:
    """Held-out perplexity via prompt_logprobs. Requires the server to expose
    prompt logprobs (vLLM OpenAI endpoint supports `prompt_logprobs`). One
    passage per line."""
    passages = [l.strip() for l in Path(corpus_file).read_text().splitlines() if l.strip()]
    if not passages:
        raise SystemExit(f"no passages in {corpus_file}")

    total_nll = 0.0
    total_tokens = 0
    for passage in passages:
        r = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": passage,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 1,  # ask server to score the prompt tokens
                "echo": True,
                # REQUIRED for correctness: without the realized token ids we
                # cannot tell which candidate in each entry is the token that
                # actually appears in the corpus. See _selected_logprob().
                "return_token_ids": True,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        # vLLM returns prompt_logprobs as a list aligned to prompt tokens;
        # first token has no preceding context -> None. Sum the rest.
        choice = data["choices"][0]
        plps = choice.get("prompt_logprobs") or []
        token_ids = choice.get("prompt_token_ids")
        if token_ids is None:
            raise SystemExit(
                "server did not return prompt_token_ids -- perplexity cannot be "
                "computed correctly without them (see _selected_logprob). "
                "Confirm this vLLM version supports `return_token_ids`."
            )
        for i, entry in enumerate(plps):
            if not entry:
                continue
            lp = _selected_logprob(entry, token_ids[i])
            if lp is not None:
                total_nll += -lp
                total_tokens += 1

    if total_tokens == 0:
        raise SystemExit(
            "no prompt logprobs returned -- confirm the server exposes "
            "prompt_logprobs and echo=True for this vLLM version."
        )
    mean_nll = total_nll / total_tokens
    return {"tokens_scored": total_tokens, "mean_nll": mean_nll, "perplexity": math.exp(mean_nll)}


def _selected_logprob(entry: dict[str, Any], realized_token_id: int) -> float | None:
    """Return the logprob of the token that ACTUALLY appears in the corpus at
    this position.

    `entry` is {token_id_str: {"logprob": x, "rank": r, ...}}. With
    prompt_logprobs=1 vLLM returns the top-1 token AND, when it differs, the
    realized token as an extra candidate carrying its true (worse) rank.

    Taking max(logprob) here is WRONG: it always selects the rank-1 token, so
    the result is the perplexity of the model's own greedy path rather than of
    the held-out corpus. Measured on this stack that underestimated perplexity
    by ~3.2x (1.66 vs 5.37). It is worse than a constant offset for this study:
    the gap between greedy and realized is exactly what quantization perturbs,
    so the error would contaminate the degradation signal being measured.

    We therefore look the realized token up by id. Verified against
    vLLM 0.27.1 (see env/requirements.lock)."""
    v = entry.get(str(realized_token_id))
    if v is None:
        # The realized token was absent from the returned candidates. Skipping
        # would silently bias the mean toward easy tokens, so fail loudly.
        raise SystemExit(
            f"realized token id {realized_token_id} missing from prompt_logprobs "
            f"entry (keys: {sorted(entry)[:5]}...). Raise prompt_logprobs or "
            "check the response schema for this vLLM version."
        )
    return v.get("logprob")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a configs/*.yaml file")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    metrics: dict[str, Any] = {}
    q = cfg.get("quality", {})
    if q.get("task_file"):
        metrics["task_accuracy"] = task_accuracy(
            cfg["base_url"], cfg["served_model_name"], q["task_file"], q.get("max_tokens", 32)
        )
    if q.get("corpus_file"):
        metrics["perplexity"] = perplexity(
            cfg["base_url"], cfg["served_model_name"], q["corpus_file"]
        )

    if not metrics:
        raise SystemExit("no quality sources configured (task_file / corpus_file).")

    RunResult(
        run_kind="quality",
        config_name=cfg["config_name"],
        quant_scheme=cfg["quant_scheme"],
        model=cfg["model"],
        metrics=metrics,
        notes=cfg.get("notes", ""),
    ).write()
    print("[quality] wrote result.")


if __name__ == "__main__":
    main()
