#!/usr/bin/env bash
# Evaluate Nemotron-3-Nano-30B-A3B on test_500 with vLLM (optionally + LoRA).
#
# Run from this directory:
#     cd training/sft_v1
#     ./run_eval.sh                       # base model only
#     LORA=output/weights ./run_eval.sh   # with a trained LoRA adapter
#     DP=8 ./run_eval.sh                  # data-parallel across 8 GPUs (8 copies)
#     TP=4 ./run_eval.sh                  # tensor-parallel across 4 GPUs (1 copy)
#
# Data:
#   - test csv     : <repo>/data/test_500.csv  (columns id,prompt,answer)
#   - corpus index : <repo>/data/corpus.jsonl  (category labels only)
#
# Extra args are forwarded to eval_vllm.py, e.g.:
#     ./run_eval.sh --limit 20 --max-tokens 4096 --no-enable-thinking
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
LORA="${LORA:-}"
TEST_CSV="${TEST_CSV:-${REPO_ROOT}/data/test_500.csv}"
CORPUS_INDEX="${CORPUS_INDEX:-${REPO_ROOT}/data/corpus.jsonl}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/eval_results.jsonl}"

# ── Engine / sampling (override via env) ─────────────────────────────
DP="${DP:-1}"                                  # data-parallel size (model copies)
TP="${TP:-1}"                                  # tensor-parallel size (per copy)
MAX_TOKENS="${MAX_TOKENS:-7680}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
SEED="${SEED:-0}"

# ── Sanity checks ────────────────────────────────────────────────────
[[ -f "${TEST_CSV}"     ]] || { echo "Missing test csv:     ${TEST_CSV}"     >&2; exit 1; }
[[ -f "${CORPUS_INDEX}" ]] || { echo "Missing corpus index: ${CORPUS_INDEX}" >&2; exit 1; }
if [[ -n "${LORA}" ]]; then
  [[ -d "${LORA}" ]] || { echo "LoRA adapter dir not found: ${LORA}" >&2; exit 1; }
fi

ARGS=(
  --model "${MODEL}"
  --test-csv "${TEST_CSV}"
  --corpus-index "${CORPUS_INDEX}"
  --output "${OUTPUT}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --tensor-parallel-size "${TP}"
  --data-parallel-size "${DP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-lora-rank "${MAX_LORA_RANK}"
  --seed "${SEED}"
)
[[ -n "${LORA}" ]] && ARGS+=(--lora "${LORA}")

cd "${SCRIPT_DIR}"

echo "Repo root   : ${REPO_ROOT}"
echo "Model       : ${MODEL}"
echo "LoRA        : ${LORA:-<none, base model>}"
echo "Test csv    : ${TEST_CSV}"
echo "Output      : ${OUTPUT}"
echo "DP x TP     : ${DP} x ${TP}"
echo

python eval_vllm.py "${ARGS[@]}" "$@"
