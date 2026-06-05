#!/usr/bin/env bash
# Installation for the Nemotron SFT training environment (local cluster, conda).
#
# This is DECOUPLED from training: run it once to build the environment, then
# run train_sft.py separately. See README.md for details.
#
# Usage:
#   bash install.sh
#
# Environment variables you may want to set first:
#   CONDA_ENV_NAME         Name of the conda env to create/use. default: nemotron-sft
#   PYTHON_VERSION         Python version for the env. default: 3.12
#   TORCH_CUDA_ARCH_LIST   GPU compute capability for building the CUDA kernels.
#                          e.g. "12.0" (RTX PRO 6000), "9.0" (H100), "8.0" (A100).
#   TORCH_INDEX_URL        PyTorch wheel index matching your CUDA toolkit.
#                          default: https://download.pytorch.org/whl/cu128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-nemotron-sft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
WHEEL_DIR="${WHEEL_DIR:-$SCRIPT_DIR/wheels}"

# 0) Create + activate the conda environment.
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
}
# Make `conda activate` usable inside this non-interactive script.
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "==> Creating conda env '${CONDA_ENV_NAME}' (python ${PYTHON_VERSION})"
    conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
else
    echo "==> Reusing existing conda env '${CONDA_ENV_NAME}'"
fi
conda activate "${CONDA_ENV_NAME}"

echo "==> Python: $(python --version) ($(which python))"
echo "==> Building for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "==> Torch index: ${TORCH_INDEX_URL}"

# 1) PyTorch (CUDA build matching your toolkit).
pip install "torch==2.10.0" --extra-index-url "${TORCH_INDEX_URL}"

# 2) Core Python dependencies.
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 3) Build mamba_ssm + causal_conv1d from source for your GPU arch.
#    Patch torch's strict CUDA-version check that otherwise blocks the build.
python -c "import torch.utils.cpp_extension as e; p=e.__file__; \
t=open(p).read().replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass  # '); \
open(p,'w').write(t)"

mkdir -p "${WHEEL_DIR}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip wheel --no-build-isolation --wheel-dir "${WHEEL_DIR}" \
    mamba_ssm==2.3.1 causal_conv1d==1.6.1
pip install --no-deps "${WHEEL_DIR}"/mamba_ssm-*.whl "${WHEEL_DIR}"/causal_conv1d-*.whl

# 4) Unsloth (from git, no deps so it doesn't override the pinned stack).
pip install --no-deps "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
pip install --no-deps "unsloth[base] @ git+https://github.com/unslothai/unsloth"

echo "==> Install complete in conda env '${CONDA_ENV_NAME}'."
echo "==> Activate it before training with:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo "==> Verify with:"
echo "    python -c 'import torch, unsloth, mamba_ssm, causal_conv1d, cut_cross_entropy; print(\"ok\")'"
