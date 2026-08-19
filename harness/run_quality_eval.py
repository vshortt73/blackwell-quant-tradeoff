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
    """task_file: JSONL of {"prompt": ..., "answer": ...}. Exact-match after a
    normalization you control. Swap in your grader for structured answers."""
    items = [json.loads(l) for l in Path(task_file).read_text().splitlines() if l.strip()]
    if not items:
        raise SystemExit(f"no task items in {task_file}")
    correct = 0
    for it in items:
        out = _completion(base_url, model, it["prompt"], max_tokens)
        text = out["choices"][0]["text"].strip()
        # Replace with your real grader; exact-match normalization shown.
        if _normalize(text) == _normalize(str(it["answer"])):
            correct += 1
    return {"n": len(items), "correct": correct, "accuracy": correct / len(items)}


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


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
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        # vLLM returns prompt_logprobs as a list aligned to prompt tokens;
        # first token has no preceding context -> None. Sum the rest.
        plps = data["choices"][0].get("prompt_logprobs") or []
        for entry in plps:
            if not entry:
                continue
            # entry maps token_id -> {logprob, ...}; take the realized token's lp
            lp = _selected_logprob(entry)
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


def _selected_logprob(entry: dict[str, Any]) -> float | None:
    """entry is {token_id_str: {"logprob": x, "rank": r, ...}}. The realized
    token is the one with rank == 1's counterpart -- but the schema varies by
    vLLM version, so pull the max logprob defensively and adjust to your
    version if you have the exact realized-token id."""
    try:
        return max(v["logprob"] for v in entry.values())
    except Exception:
        return None


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
