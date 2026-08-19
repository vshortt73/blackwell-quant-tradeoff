#!/usr/bin/env bash
# Full three-arm sweep, fully autonomous.
#
# Runs to completion without a live terminal: launch it with setsid so it is
# reparented to init and survives the SSH session that started it. Every output
# goes to the repo (results/), never to a session-scoped scratchpad -- losing
# hours of GPU time to a /tmp cleanup is the failure this guards against.
#
#   setsid nohup env/run_sweep.sh > /dev/null 2>&1 < /dev/null &
#
# Per arm: serve -> settle -> serving bench -> quality eval -> APEX -> stop.
# An arm that fails is logged and the sweep continues, so one bad arm does not
# cost the other two.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT="$REPO/results/apex"
mkdir -p "$OUT" "$REPO/results/raw"
LOG="$OUT/sweep.log"
DB="$OUT/study.db"
EVAL_URL="${EVAL_URL:-http://node2:8080}"
PROBES='["I-001","I-002","I-005","I-006","I-007","I-008","I-009","I-010","I-012","I-013","I-014","I-015","I-016","I-017","I-018","I-020","F-008","F-009","F-013","F-014","F-015","F-019","S-005","S-008","S-010","S-014","S-017","S-018"]'

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== sweep start ==="
say "repo=$REPO db=$DB evaluator=$EVAL_URL"

# Preflight: a missing input or a dead evaluator wastes the whole sweep.
for f in data/prompts_512in.txt data/tasks.jsonl data/heldout_corpus.txt; do
  [ -f "$f" ] || { say "FATAL missing input $f"; exit 1; }
done
curl -s --max-time 10 "$EVAL_URL/v1/models" >/dev/null 2>&1 \
  || { say "FATAL evaluator unreachable at $EVAL_URL"; exit 1; }
.venv/bin/python harness/check_evaluator.py --base-url "$EVAL_URL" --json \
  > "$OUT/evaluator_before.json" 2>/dev/null \
  && say "evaluator fingerprint (before): $(.venv/bin/python -c "import json;print(json.load(open('$OUT/evaluator_before.json'))['fingerprint'])")"

nvidia-smi --query-gpu=clocks.sm,power.limit --format=csv,noheader | tee -a "$LOG"

run_arm() {
  local cfg="$1" ckpt="$2" mid="$3" quant="$4"
  say "--- ARM $mid ---"
  VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_DEEP_GEMM=0 \
    .venv/bin/vllm serve "$ckpt" \
    --gpu-memory-utilization 0.85 --max-model-len 12288 --port 8000 \
    --served-model-name model-under-test --reasoning-parser qwen3 \
    > "$OUT/vllm_$mid.log" 2>&1 &
  local vp=$!
  local up=0
  for i in $(seq 1 90); do
    curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $vp 2>/dev/null || break
    sleep 10
  done
  [ $up -eq 1 ] || { say "ARM $mid: vLLM failed to start, skipping"; kill -9 $vp 2>/dev/null; return 1; }
  say "ARM $mid: vLLM up; settling 60s"
  sleep 60

  say "ARM $mid: serving benchmark"
  .venv/bin/python harness/run_serving_bench.py "$cfg" >> "$LOG" 2>&1 \
    || say "ARM $mid: serving bench FAILED"
  say "ARM $mid: quality eval"
  .venv/bin/python harness/run_quality_eval.py "$cfg" >> "$LOG" 2>&1 \
    || say "ARM $mid: quality eval FAILED"

  cat > "$OUT/apex_$mid.yaml" <<YAML
run: {seed: 42, temperature: 0.0, repetitions: 2, filler_type: neutral, workers: 3}
data: {directory: /opt/apex/data, output_db: $DB}
probes:
  select: $PROBES
positions: [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
context_lengths: [8192]
models:
  - name: $mid
    backend: openai
    model_name: model-under-test
    base_url: http://127.0.0.1:8000/v1
    api_key: vllm-local
    tokenizer: approximate
    max_context_window: 12288
    max_tokens: 512
    architecture: transformer-dense
    parameters: 8B
    quantization: $quant
    no_think: true
evaluator_models:
  - name: qwen2.5-14b-eval
    backend: llamacpp
    model_name: qwen2.5-14b-eval
    base_url: $EVAL_URL
    max_context_window: 8192
YAML
  say "ARM $mid: APEX (28 probes x 13 positions x 2 reps)"
  ( cd /opt/apex && env -u APEX_DATABASE_URL /opt/apex/.venv/bin/apex run "$OUT/apex_$mid.yaml" ) \
    >> "$LOG" 2>&1 || say "ARM $mid: APEX FAILED"

  say "ARM $mid: stopping vLLM"
  kill $vp 2>/dev/null
  for i in $(seq 1 30); do kill -0 $vp 2>/dev/null || break; sleep 2; done
  kill -9 $vp 2>/dev/null; sleep 8
  say "ARM $mid: done"
}

run_arm configs/bf16_baseline.yaml /mnt/nvme4tb/llm_models/qwen/Qwen3-8B       qwen3-8b-bf16 BF16
run_arm configs/fp8_native.yaml    /mnt/nvme4tb/llm_models/qwen/Qwen3-8B-FP8   qwen3-8b-fp8  FP8-native
run_arm configs/awq_4bit.yaml      /mnt/nvme4tb/llm_models/qwen/Qwen3-8B-AWQ   qwen3-8b-awq  AWQ-4bit

.venv/bin/python harness/check_evaluator.py --base-url "$EVAL_URL" --json \
  > "$OUT/evaluator_after.json" 2>/dev/null \
  && say "evaluator fingerprint (after): $(.venv/bin/python -c "import json;print(json.load(open('$OUT/evaluator_after.json'))['fingerprint'])")"

say "=== sweep complete ==="
say "results/raw: $(ls "$REPO/results/raw" 2>/dev/null | wc -l) files"
[ -f "$DB" ] && say "apex rows: $(.venv/bin/python -c "import sqlite3;print(sqlite3.connect('$DB').execute('select count(*) from probe_results').fetchone()[0])" 2>/dev/null)"
