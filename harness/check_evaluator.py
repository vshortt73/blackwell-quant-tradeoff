"""
Preflight and fingerprint the APEX evaluator model.

WHY THIS EXISTS: rubric-scored probes (all of salience, 15 of 20 application)
are graded by an LLM evaluator, which puts it inside the measurement chain. For
a degradation study the evaluator must be IDENTICAL across every arm -- a stable
judge bias cancels in the BF16 -> FP8 -> AWQ delta, but a judge that changes
mid-sweep injects a difference that looks exactly like quantization damage.

APEX records `evaluator_model_id`, but that value comes from the `name` field a
human typed into the config. It is a label, not evidence: swap the model served
on the evaluator host and APEX will keep recording the old label.

This script turns that label into evidence. It asks the endpoint what it is
actually serving and probes it with fixed prompts at temperature 0, then prints
a fingerprint hash. Run it BEFORE the first arm and AFTER the last. If the hash
differs, the evaluator changed underneath the sweep and the rubric-scored
dimensions are not comparable across arms.

It also verifies determinism: the same prompt twice must give byte-identical
output. A non-deterministic judge adds variance to every rubric-scored point.

Usage:
    python harness/check_evaluator.py --base-url http://node2:8080
    python harness/check_evaluator.py --base-url http://node2:8080 --json > eval_pre.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import requests

# Fixed, boring prompts. Content does not matter; stability does. Changing them
# invalidates comparison against previously recorded fingerprints.
PROBES = [
    "Reply with exactly one word: OK",
    "Rate the following on a scale of 1 to 5 and reply with only the number. Text: The sky is blue.",
    "Summarise in exactly three words: The cat sat on the mat.",
]


def _identity(base_url: str, timeout: float) -> dict:
    """Ask the server what it is serving. llama-server exposes /props; anything
    OpenAI-compatible exposes /v1/models."""
    out: dict = {}
    for path, key in (("/v1/models", "v1_models"), ("/props", "props")):
        try:
            r = requests.get(base_url.rstrip("/") + path, timeout=timeout)
            if r.ok:
                out[key] = r.json()
        except Exception as e:
            out[key + "_error"] = f"{type(e).__name__}: {e}"
    return out


def _complete(base_url: str, model: str | None, prompt: str, timeout: float) -> str:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 24,
        "stream": False,
    }
    if model:
        body["model"] = model
    r = requests.post(base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://node2:8080")
    ap.add_argument("--model", default=None, help="model name, if the server needs one")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--json", action="store_true", help="emit the fingerprint as JSON")
    args = ap.parse_args()

    report: dict = {"base_url": args.base_url}

    try:
        requests.get(args.base_url.rstrip("/") + "/v1/models", timeout=args.timeout)
    except Exception as e:
        print(f"[FAIL] evaluator unreachable at {args.base_url}: {type(e).__name__}: {e}",
              file=sys.stderr)
        print("       Start the evaluator before running APEX -- rubric-scored probes "
              "will otherwise run and store NULL scores.", file=sys.stderr)
        return 1

    ident = _identity(args.base_url, args.timeout)
    report["identity"] = ident
    served = None
    if isinstance(ident.get("v1_models"), dict):
        data = ident["v1_models"].get("data") or []
        if data:
            served = data[0].get("id")
    if not served and isinstance(ident.get("props"), dict):
        served = ident["props"].get("model_path") or ident["props"].get("default_generation_settings", {}).get("model")
    report["served_model"] = served

    responses = []
    for p in PROBES:
        try:
            a = _complete(args.base_url, args.model or served, p, args.timeout)
            b = _complete(args.base_url, args.model or served, p, args.timeout)
        except Exception as e:
            print(f"[FAIL] evaluator did not answer a probe: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        responses.append({"prompt": p, "response": a, "deterministic": a == b})

    report["probes"] = responses
    nondet = [r for r in responses if not r["deterministic"]]
    # Fingerprint = identity + canonical responses. If the served model, its
    # quantization, or its sampling settings change, this changes.
    canon = json.dumps({"served": served, "responses": [r["response"] for r in responses]},
                       sort_keys=True)
    report["fingerprint"] = hashlib.sha256(canon.encode()).hexdigest()[:16]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"  endpoint     : {args.base_url}")
        print(f"  served model : {served}")
        for r in responses:
            mark = "ok" if r["deterministic"] else "NON-DETERMINISTIC"
            print(f"  probe [{mark}] -> {r['response'].strip()[:60]!r}")
        print(f"  FINGERPRINT  : {report['fingerprint']}")
        print("\n  Record this. Re-run after the last arm; a different fingerprint means")
        print("  the evaluator changed mid-sweep and rubric-scored dimensions are not")
        print("  comparable across arms.")

    if nondet:
        print(f"\n[FAIL] {len(nondet)} probe(s) non-deterministic at temperature 0. "
              "A varying judge adds noise to every rubric-scored point.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
