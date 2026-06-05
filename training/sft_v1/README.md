# Nemotron SFT fine-tuning (local cluster)

LoRA supervised fine-tuning of **Nemotron-3-Nano-30B-A3B** on the synthetic
reasoning corpus, refactored from the original Kaggle/Modal notebook
(`end-to-end-finetuning-for-lb-0-85.ipynb`) to run on your own GPU cluster.

Installation and training are **decoupled**:

| File | Purpose |
|------|---------|
| `install.sh` + `requirements.txt` | One-time environment setup (this README) |
| `train_sft.py` | Pure training script (single- or multi-GPU) |

The original Modal/Kaggle orchestration has been removed.

---

## 1. Requirements

- Linux with NVIDIA GPU(s) and a matching CUDA toolkit (CUDA 12.x).
- A C/C++ build toolchain (`build-essential`, `clang`, `git`) — needed to
  compile `mamba_ssm` and `causal_conv1d`.
- [conda](https://docs.conda.io/) (Miniconda or Anaconda) on your `PATH`.
- Enough VRAM to hold the 30B (A3B) base model in bf16 plus LoRA + activations.
  With data-parallel multi-GPU, **each** GPU holds a full copy of the model.

Install the system build tools first (Debian/Ubuntu example):

```sh
sudo apt-get update && sudo apt-get install -y git build-essential clang
```

## 2. Install the Python environment

`install.sh` creates (or reuses) a conda environment and installs everything
into it. Just set your GPU compute capability and CUDA wheel index, then run it:

```sh
# Set your GPU compute capability and CUDA wheel index, then install.
#   RTX PRO 6000 -> 12.0     H100 -> 9.0     A100 -> 8.0     L40S -> 8.9
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# Optional: customize the env name / Python version (defaults shown).
export CONDA_ENV_NAME=nemotron-sft
export PYTHON_VERSION=3.12

bash install.sh
```

`install.sh` performs, in order:

0. Creates conda env `$CONDA_ENV_NAME` (Python `$PYTHON_VERSION`) if missing,
   then activates it.
1. Installs `torch==2.10.0` from `TORCH_INDEX_URL`.
2. Installs the pinned stack from `requirements.txt`.
3. Builds `mamba_ssm==2.3.1` and `causal_conv1d==1.6.1` from source for your
   `TORCH_CUDA_ARCH_LIST`.
4. Installs `unsloth` + `unsloth_zoo` from git (`--no-deps`).

Activate the env and verify the install:

```sh
conda activate nemotron-sft
python -c "import torch, unsloth, mamba_ssm, causal_conv1d, cut_cross_entropy; print('ok')"
```

## 3. Data

The script reads the corpus produced by the repo's `corpus.py`:

- `corpus.jsonl` — entry index (at the repo root)
- `corpus/<problem_id>/synthetic.jsonl` — per-problem token/mask segments

Defaults point at the repo root automatically. Regenerate the corpus first if
needed (from the repo root):

```sh
uv run python3 corpus.py
```

---

## 4. Train

All hyperparameters live in the `Config` dataclass at the top of
`train_sft.py`. Edit the defaults there, or override any field on the command
line.

### Single GPU

```sh
python train_sft.py
```

### Multiple GPUs (single node)

Launch with `torchrun`; the script auto-detects the distributed environment and
uses data-parallel training (each GPU processes a shard of every batch, and
gradients are averaged across GPUs). Single-GPU results are unchanged when
`world_size == 1`.

```sh
# 4 GPUs on one node
torchrun --standalone --nproc_per_node=4 train_sft.py
```

### Multiple nodes

```sh
# e.g. 2 nodes x 8 GPUs; run on each node with the same MASTER_ADDR
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
    train_sft.py
```

> `batch_size` is the **global** batch size summed across all GPUs and must be
> divisible by the number of GPUs. `micro_batch_size` is per-GPU.

### Overriding hyperparameters from the CLI

Every `Config` field is exposed as a flag. Examples:

```sh
torchrun --nproc_per_node=4 train_sft.py \
    --learning_rate 1e-4 \
    --num_steps 500 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --lora_rank 16

# Booleans use --flag / --no-flag:
python train_sft.py --no-moe_tie_weights --shuffle_dataset

# Continue from a pretrained adapter instead of a fresh LoRA init:
python train_sft.py --no-reset_weights --adapter_src /path/to/adapter
```

Run `python train_sft.py --help` to see every flag, its default, and a short
description (grouped by model / LoRA / optimization / data / output).

Key hyperparameters:

| Flag | Default | Meaning |
|------|---------|---------|
| `--model_path` | `unsloth/Nemotron-3-Nano-30B-A3B` | Base model (HF id or local path) |
| `--lora_rank` / `--lora_alpha` / `--lora_dropout` | 32 / 32 / 0.0 | LoRA config |
| `--max_seq_len` | 8192 | Max tokens per example |
| `--num_steps` | 1000 | Optimizer steps (clamped to dataset size) |
| `--batch_size` | 32 | Global batch size (across all GPUs) |
| `--micro_batch_size` | 4 | Per-GPU micro-batch for grad accumulation |
| `--learning_rate` | 2e-4 | Peak LR (linear decay to 0) |
| `--reset_weights` | True | Start from fresh LoRA init |
| `--in_proj_only` | False | Freeze all LoRA except `in_proj` |
| `--moe_tie_weights` | True | Tie LoRA across MoE experts (Tinker-style) |
| `--shuffle_dataset` | False | Shuffle corpus order (seeded) |
| `--output_dir` | `./output/weights` | Where the adapter is saved |

## 5. Output

Rank 0 writes to `output_dir`:

- `adapter_model.safetensors` (+ `adapter_config.json`) — the LoRA adapter,
  with `lm_head` keys renamed to the submission convention.
- `training_log.txt` — per-step loss / grad-norm / lr log.
