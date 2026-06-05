#!/usr/bin/env bash
# Launch SFT LoRA training for Nemotron-3-Nano-30B-A3B.
#
# Run from this directory:
#     cd training/sft_v1
#     ./run_train.sh            # single GPU
#     NUM_GPUS=4 ./run_train.sh # multi-GPU (single node)
#
# Data:
#   - index : data/corpus.jsonl   (test_500 already marked included=False)
#   - tokens: <repo>/corpus/<id>/synthetic.jsonl
#   - csv   : data/train_9000.csv  (used only when --original_problems_only)
#
# Extra args are forwarded to train_sft.py, e.g.:
#     ./run_train.sh --learning_rate 1e-4 --num_steps 500 --original_problems_only
set -euo pipefail

# Reduce allocator fragmentation. On 80GB H100s the 30B model sits near the
# memory ceiling; multi-GPU adds NCCL comm buffers that can tip it into OOM.
# expandable_segments reclaims the "reserved but unallocated" fragmentation that
# the OOM message flags. Override by exporting before calling this script.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CORPUS_INDEX="${REPO_ROOT}/data/corpus.jsonl"
CORPUS_DIR="${REPO_ROOT}/corpus"
TRAIN_CSV="${REPO_ROOT}/data/train_9000.csv"
OUTPUT_DIR="${SCRIPT_DIR}/output/weights"

# ── Hyperparameters (override via env) ───────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_STEPS="${NUM_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

# ── Sanity checks ────────────────────────────────────────────────────
[[ -f "${CORPUS_INDEX}" ]] || { echo "Missing corpus index: ${CORPUS_INDEX}" >&2; exit 1; }
[[ -d "${CORPUS_DIR}"   ]] || { echo "Missing corpus dir:   ${CORPUS_DIR}"   >&2; exit 1; }
[[ -f "${TRAIN_CSV}"    ]] || { echo "Missing train csv:    ${TRAIN_CSV}"    >&2; exit 1; }

COMMON_ARGS=(
  --corpus_index "${CORPUS_INDEX}"
  --corpus_dir "${CORPUS_DIR}"
  --train_csv "${TRAIN_CSV}"
  --output_dir "${OUTPUT_DIR}"
  --num_epochs "${NUM_EPOCHS}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
)

cd "${SCRIPT_DIR}"

echo "Repo root   : ${REPO_ROOT}"
echo "Corpus index: ${CORPUS_INDEX}"
echo "Corpus dir  : ${CORPUS_DIR}"
echo "Train csv   : ${TRAIN_CSV}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "GPUs        : ${NUM_GPUS}"
echo

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    train_sft.py "${COMMON_ARGS[@]}" "$@"
else
  python train_sft.py "${COMMON_ARGS[@]}" "$@"
fi