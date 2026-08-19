"""
Serving benchmark against a running vLLM OpenAI-compatible server.

Measures, per concurrency level:
  - TTFT  (time to first token)      -- from streamed first chunk
  - TPOT  (time per output token)    -- mean inter-token gap after first
  - E2E   (end-to-end latency)
  - output-token throughput (tok/s)  -- OUTPUT tokens only, stated explicitly
  - request throughput (req/s)

All latency stats reported at p50/p90/p99, not means. Warmup requests are
discarded before stats are computed (CUDA-graph capture and cold KV cache
distort the first iterations badly on Blackwell).

This is a self-contained client so the measurement path is fully in-repo and
auditable. vLLM also ships benchmarks/benchmark_serving.py; running that as a
cross-check and committing its JSON alongside these results is encouraged, but
the headline numbers should come from one tool used consistently.

Assumes the server is already up, e.g.:
    vllm serve <model> --quantization <scheme> \
        --gpu-memory-utilization 0.85 --max-model-len 8192 \
        --port 8000 --served-model-name model-under-test
Flags are version-sensitive; pin the exact launch command per config in
configs/*.yaml and record the vLLM version via the fingerprint.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

import yaml  # pip install pyyaml

from common import RunResult, declared_controls, load_config
from gpu_sampler import GpuSampler

try:
    import aiohttp  # pip install aiohttp
except ImportError as e:  # pragma: no cover
    raise SystemExit("aiohttp required: pip install aiohttp") from e


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    """Fire one streamed completion, return per-request timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,  # 0.0 for quality runs; per-config for throughput
        "stream": True,
        # CONTROL: force every request to decode exactly max_tokens. Without
        # this each scheme stops at its OWN EOS, and quantization changes what
        # the model emits -- so the arms would decode different token counts,
        # with different KV pressure and batch occupancy, and the difference
        # would be reported as throughput. Equal decode work per arm is the
        # only way the comparison isolates the quant scheme.
        "ignore_eos": ignore_eos,
    }
    t0 = time.perf_counter()
    ttft = None
    token_times: list[float] = []
    n_out = 0
    async with session.post(url, json=payload) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            token_times.append(now)
            n_out += 1
    t_end = time.perf_counter()
    tpot = None
    if len(token_times) > 1:
        gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        tpot = statistics.mean(gaps)
    return {
        "ttft": ttft,
        "tpot": tpot,
        "e2e": t_end - t0,
        "n_out": n_out,
    }


async def run_concurrency_level(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    warmup: int,
    sample_interval: float = 0.25,
    gpu_index: int = 0,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:

        async def guarded(p: str):
            async with sem:
                return await one_request(
                    session, url, model, p, max_tokens, temperature, ignore_eos
                )

        # warmup (discarded) -- deliberately OUTSIDE the power sampler, so
        # CUDA-graph capture and cache-warm transients don't skew load power.
        if warmup:
            await asyncio.gather(*[guarded(prompts[i % len(prompts)]) for i in range(warmup)])

        # Telemetry is sampled only across the measured interval. A single
        # post-hoc nvidia-smi read is a wind-down snapshot, not load power.
        with GpuSampler(interval=sample_interval, gpu_index=gpu_index) as sampler:
            wall_start = time.perf_counter()
            tasks = [guarded(prompts[i % len(prompts)]) for i in range(len(prompts))]
            results = await asyncio.gather(*tasks)
            wall = time.perf_counter() - wall_start
        telemetry = sampler.stats()

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if r["tpot"] is not None]
    e2es = [r["e2e"] for r in results]
    total_out = sum(r["n_out"] for r in results)

    out_tps = total_out / wall if wall > 0 else None
    # Energy efficiency: tokens generated per joule. Only meaningful when
    # telemetry is available; this is the metric that actually compares cards
    # of different power classes on equal footing.
    tokens_per_joule = None
    mean_w = (telemetry.get("power_w") or {}).get("mean") if telemetry.get("available") else None
    if out_tps and mean_w:
        tokens_per_joule = out_tps / mean_w

    return {
        "concurrency": concurrency,
        "n_requests": len(results),
        "wall_seconds": wall,
        "request_throughput_rps": len(results) / wall if wall > 0 else None,
        "output_token_throughput_tps": out_tps,
        "tokens_per_joule": tokens_per_joule,
        "ttft_s": {"p50": pct(ttfts, 0.50), "p90": pct(ttfts, 0.90), "p99": pct(ttfts, 0.99)},
        "tpot_s": {"p50": pct(tpots, 0.50), "p90": pct(tpots, 0.90), "p99": pct(tpots, 0.99)},
        "e2e_s": {"p50": pct(e2es, 0.50), "p90": pct(e2es, 0.90), "p99": pct(e2es, 0.99)},
        "telemetry": telemetry,
    }


def load_prompts(path: str, n: int) -> list[str]:
    """One prompt per line. Replace with your controlled prompt set.
    Keep the input-length distribution FIXED and DECLARED across all schemes."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit(f"no prompts found in {path}")
    return [lines[i % len(lines)] for i in range(n)]


async def main_async(cfg: dict[str, Any]) -> None:
    prompts = load_prompts(cfg["prompt_file"], cfg["n_requests"])
    per_level = []
    for c in cfg["concurrency_levels"]:
        print(f"[serving] concurrency={c} ...")
        res = await run_concurrency_level(
            base_url=cfg["base_url"],
            model=cfg["served_model_name"],
            prompts=prompts,
            concurrency=c,
            max_tokens=cfg["max_tokens"],
            temperature=cfg.get("throughput_temperature", 0.0),
            warmup=cfg.get("warmup_requests", 8),
            ignore_eos=cfg.get("ignore_eos", True),
            sample_interval=cfg.get("power_sample_interval_s", 0.25),
            gpu_index=cfg.get("gpu_index", 0),
        )
        tel = res.get("telemetry", {})
        if tel.get("available"):
            pw = tel["power_w"]
            print(
                f"    -> {res['output_token_throughput_tps']:.1f} tok/s | "
                f"power mean {pw['mean']:.0f}W (max {pw['max']:.0f}W, n={tel['n_samples']}) | "
                f"{res['tokens_per_joule']:.2f} tok/J"
            )
        else:
            print(f"    -> {res['output_token_throughput_tps']:.1f} tok/s | telemetry UNAVAILABLE")
        per_level.append(res)

    RunResult(
        run_kind="serving",
        config_name=cfg["config_name"],
        quant_scheme=cfg["quant_scheme"],
        model=cfg["model"],
        metrics={"by_concurrency": per_level, "input_profile": cfg.get("input_profile")},
        notes=cfg.get("notes", ""),
        declared_controls=declared_controls(cfg),
    ).write()
    print("[serving] wrote result.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a configs/*.yaml file")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
