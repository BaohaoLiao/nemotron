"""SFT LoRA fine-tuning for Nemotron-3-Nano-30B-A3B on a local cluster.

Decoupled from the original Kaggle/Modal notebook: this file contains ONLY the
training logic. Installation is handled separately (see README.md / install.sh).

Single-GPU:
    python train_sft.py

Multi-GPU (single node, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 train_sft.py

Multi-GPU (multi node):
    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 train_sft.py

All hyperparameters are argparse flags defined in `parse_args()` below. Run
`python train_sft.py --help` to see them, or override any of them, e.g.:
    torchrun --nproc_per_node=4 train_sft.py --learning_rate 1e-4 --num_steps 500
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# When launched via torchrun, pin EACH process to a single GPU *before* torch
# initialises CUDA. Otherwise unsloth/HF `from_pretrained` loads every rank's
# full 30B model copy onto cuda:0, OOMing GPU 0 even at micro_batch_size=1.
# After this, each process sees exactly one GPU, exposed as index 0.
if "LOCAL_RANK" in os.environ:
    _local_rank = int(os.environ["LOCAL_RANK"])
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible:
        _devices = [d for d in _visible.split(",") if d != ""]
        if _local_rank < len(_devices):
            os.environ["CUDA_VISIBLE_DEVICES"] = _devices[_local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import torch
import torch.distributed as dist

# Repo root = .../nemotron (two levels up from training/sft_v1/).
REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — all defined as argparse flags. Edit the defaults here or
# override any of them on the command line (see `--help`).
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse all hyperparameters from the command line."""
    p = argparse.ArgumentParser(
        description="SFT LoRA fine-tuning for Nemotron-3-Nano-30B-A3B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / adapter ──────────────────────────────────────────────
    g = p.add_argument_group("model / adapter")
    g.add_argument(
        "--model_path",
        default="unsloth/Nemotron-3-Nano-30B-A3B",
        help="Base model (HF id or local path).",
    )
    g.add_argument(
        "--adapter_src",
        default="",
        help="Pretrained LoRA adapter to continue from (used when --no_reset_weights).",
    )
    g.add_argument(
        "--reset_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start from a fresh LoRA init instead of loading --adapter_src.",
    )

    # ── LoRA ─────────────────────────────────────────────────────────
    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_rank", type=int, default=32)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--lora_dropout", type=float, default=0.0)
    g.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "out_proj",
            "lm_head",
        ],
        help="LoRA target modules.",
    )
    g.add_argument(
        "--in_proj_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze all LoRA params except `in_proj`.",
    )
    g.add_argument(
        "--moe_tie_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie one LoRA side across all MoE experts (Tinker-style).",
    )

    # ── Optimization ─────────────────────────────────────────────────
    g = p.add_argument_group("optimization")
    g.add_argument("--max_seq_len", type=int, default=8192)
    g.add_argument("--num_steps", type=int, default=1000)
    g.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Global batch size summed across all GPUs; must divide by world size.",
    )
    g.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Per-GPU micro-batch size used for gradient accumulation.",
    )
    g.add_argument("--learning_rate", type=float, default=2e-4)
    g.add_argument("--adam_beta1", type=float, default=0.9)
    g.add_argument("--adam_beta2", type=float, default=0.95)
    g.add_argument("--adam_eps", type=float, default=1e-8)
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--grad_clip_norm", type=float, default=1e9)
    g.add_argument("--seed", type=int, default=42)

    # ── Data ─────────────────────────────────────────────────────────
    g = p.add_argument_group("data")
    g.add_argument("--corpus_dir", default=str(REPO_ROOT / "corpus"))
    g.add_argument("--corpus_index", default=str(REPO_ROOT / "corpus.jsonl"))
    g.add_argument("--segment_name", default="synthetic.jsonl")
    g.add_argument("--train_csv", default=str(REPO_ROOT / "train.csv"))
    g.add_argument(
        "--original_problems_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only problem_ids present in --train_csv.",
    )
    g.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # ── Output ───────────────────────────────────────────────────────
    g = p.add_argument_group("output")
    g.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "output" / "weights"),
    )
    g.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Also save the adapter every N optimizer steps to "
        "<output_dir>/step_<N> (0 = only save once at the end).",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Initialise torch.distributed if launched via torchrun, else single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world_size > 1:
            dist.init_process_group(backend="nccl")
            # Each process was pinned to a single GPU at import time via
            # CUDA_VISIBLE_DEVICES, so its one visible GPU is index 0.
            torch.cuda.set_device(0)
            return DistInfo(rank, world_size, local_rank, True)
    torch.cuda.set_device(0)
    return DistInfo(0, 1, 0, False)


def cleanup_distributed(dist_info: DistInfo) -> None:
    if dist_info.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_segment(path: Path) -> tuple[list[int], list[int]]:
    """Reconstruct (tokens, mask) from a segment file.

    Supports two formats:
      * .jsonl segment file: one JSON object per line, each with
        {"type": "masked"|"unmasked", "tokens": [...]}.
      * single-record .json file: {"tokens": [...], "mask": [...]}.
    """
    if path.suffix == ".json":
        with open(path) as f:
            rec = json.load(f)
        return rec["tokens"], rec["mask"]

    tokens: list[int] = []
    mask: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg = json.loads(line)
            seg_tokens = seg["tokens"]
            tokens.extend(seg_tokens)
            mask_val = 1 if seg["type"] == "unmasked" else 0
            mask.extend([mask_val] * len(seg_tokens))
    return tokens, mask


def load_examples(cfg: argparse.Namespace, log) -> list[dict]:
    """Load and pre-tokenize the training corpus into next-token examples."""
    index_path = Path(cfg.corpus_index)
    corpus_dir = Path(cfg.corpus_dir)
    assert index_path.is_file(), f"Corpus index not found: {index_path}"
    assert corpus_dir.is_dir(), f"Corpus dir not found: {corpus_dir}"

    index: list[dict] = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(json.loads(line))

    examples: list[dict] = []
    for rec in index:
        if not rec.get("included", True):
            continue
        pid = rec["problem_id"]
        segment = rec.get("segment", cfg.segment_name)
        seg_path = corpus_dir / pid / segment
        if not seg_path.is_file():
            log(f"WARNING: missing segment for {pid}: {seg_path}")
            continue
        tokens, mask = _load_segment(seg_path)
        if not tokens:
            continue
        if len(tokens) > cfg.max_seq_len:
            tokens = tokens[: cfg.max_seq_len]
            mask = mask[: cfg.max_seq_len]
        if not any(mask):
            continue
        examples.append(
            {
                "problem_id": pid,
                "tokens": tokens[:-1],
                "targets": tokens[1:],
                "weights": [float(m) for m in mask[1:]],
            }
        )

    if cfg.original_problems_only:
        import csv

        with open(cfg.train_csv) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        log(
            f"original_problems_only=True: filtered {before} -> {len(examples)} "
            f"examples using {len(original_ids)} ids from {cfg.train_csv}"
        )

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    log(
        f"Loaded {len(examples)} examples, {total_tokens:,} tokens "
        f"(unmasked={total_unmasked:,.0f})"
    )
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg: argparse.Namespace, device: torch.device, log):
    """Load base model, attach LoRA, patch the forward with Cut Cross-Entropy."""
    from unsloth import FastLanguageModel  # noqa: F401  (must import first)

    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear

    # ── GPU + kernel sanity check ────────────────────────────────────
    import causal_conv1d
    import mamba_ssm
    from causal_conv1d import causal_conv1d_fn

    cc = torch.cuda.get_device_capability(device.index or 0)
    log(f"GPU: {torch.cuda.get_device_name(device)}, sm_{cc[0] * 10 + cc[1]}")
    log(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    log(f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}")
    log(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    _x = torch.randn(1, 256, 32, device=device, dtype=torch.bfloat16)
    _w = torch.randn(256, 4, device=device, dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    log("causal_conv1d CUDA kernel: OK")

    # ── Load base model ──────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_path,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

    # ── Wrap in LoRA ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    FastLanguageModel.for_training(model)

    # ── Patch Mamba CUDA fast path ───────────────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    log("Patched is_fast_path_available = True")

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout
        )
        model.base_model._create_and_replace(
            _cfg,
            "default",
            target=_lm_head,
            target_name="lm_head",
            parent=_causal_lm,
        )
        log("Manually added LoRA to lm_head")
    else:
        log("lm_head already has LoRA")

    # ── Cast LoRA params to fp32 (base stays bf16 except MoE router) ──
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)
    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, f"{name} expected fp32"
            continue
        # Nemotron-H keeps the MoE router (`mixer.gate`) in fp32 on purpose.
        if ".mixer.gate." in name:
            assert param.dtype == torch.float32, f"{name} expected fp32"
            continue
        assert param.dtype == torch.bfloat16, f"{name} expected bf16"
    log("Verified: LoRA params fp32, base params bf16 (MoE router fp32)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # ── Patch forward with Cut Cross-Entropy (no logit materialization) ──
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_causal_forward(
        input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states, lm_weight, labels, reduction="none"
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce  # type: ignore[attr-defined]
        return loss

    _base.forward = _patched_causal_forward
    log("Patched CausalLM.forward with CCE (no logits materialization)")

    # ── Load adapter weights (unless reset_weights) ──────────────────
    if cfg.reset_weights:
        log("reset_weights=True — using fresh LoRA init")
    else:
        from peft import load_peft_weights

        assert cfg.adapter_src, "adapter_src must be set when reset_weights=False"
        log(f"Loading adapter from {cfg.adapter_src}...")
        adapter_weights = load_peft_weights(cfg.adapter_src)
        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        for ak, av in adapter_weights.items():
            if ak in model_sd:
                new_sd[ak] = av
                loaded += 1
                continue
            ak_with_default = ak.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_with_default in model_sd:
                new_sd[ak_with_default] = av
                loaded += 1
                continue
            ak_lm = ak.replace(".backbone.lm_head.", ".lm_head.")
            ak_lm_default = ak_lm.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_lm_default in model_sd:
                new_sd[ak_lm_default] = av
                loaded += 1
                continue
        model.load_state_dict(new_sd, strict=False)
        assert loaded == len(adapter_weights), (
            f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
        )
        log(f"  Loaded {loaded}/{len(adapter_weights)} weights into model")

    # ── Optionally freeze all LoRA params except in_proj ─────────────
    if cfg.in_proj_only:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(
        f"in_proj_only={cfg.in_proj_only}: {trainable_params:,} trainable / {frozen_params:,} frozen"
    )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# MoE weight tying
# ─────────────────────────────────────────────────────────────────────────────
def build_moe_tying(cfg: argparse.Namespace, model, log):
    """Identify MoE expert LoRA params to tie and return (tie_init, tie_grads)."""
    moe_tied_params: list[torch.Tensor] = []
    if not cfg.moe_tie_weights:
        return (lambda: None), (lambda: None)

    w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
    w2_proj_names = ("down_proj", ".w2.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".experts." not in name or ".lora_" not in name:
            continue
        is_w1 = any(p in name for p in w1_proj_names)
        is_w2 = any(p in name for p in w2_proj_names)
        is_A = ".lora_A." in name
        is_B = ".lora_B." in name
        should_tie = (is_w1 and is_A) or (is_w2 and is_B)
        if not should_tie:
            continue
        if param.dim() < 2 or param.shape[0] <= 1:
            continue
        moe_tied_params.append(param)

    def tie_init() -> None:
        with torch.no_grad():
            for p in moe_tied_params:
                mean = p.data.mean(dim=0, keepdim=True)
                p.data.copy_(mean.expand_as(p.data))

    def tie_grads() -> None:
        with torch.no_grad():
            for p in moe_tied_params:
                if p.grad is None:
                    continue
                grad_sum = p.grad.sum(dim=0, keepdim=True)
                p.grad.copy_(grad_sum.expand_as(p.grad))

    log(f"MoE weight tying: {len(moe_tied_params)} params identified for tying")
    tie_init()  # start from a tied state
    return tie_init, tie_grads


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: argparse.Namespace, dist_info: DistInfo) -> None:
    # After the per-rank CUDA_VISIBLE_DEVICES pin, each process sees exactly one
    # GPU as cuda:0 (true for both single- and multi-GPU launches).
    device = torch.device("cuda:0")
    training_log: list[str] = []

    def log(msg: str) -> None:
        if dist_info.is_main:
            print(msg, flush=True)
            training_log.append(msg)

    log(
        f"Distributed: {dist_info.distributed}, world_size={dist_info.world_size}, "
        f"rank={dist_info.rank}, local_rank={dist_info.local_rank}"
    )
    if dist_info.distributed:
        assert cfg.batch_size % dist_info.world_size == 0, (
            f"batch_size={cfg.batch_size} must be divisible by "
            f"world_size={dist_info.world_size}"
        )

    examples = load_examples(cfg, log)
    model, _ = build_model(cfg, device, log)
    _, tie_grads = build_moe_tying(cfg, model, log)

    gc.collect()
    torch.cuda.empty_cache()

    indices = list(range(len(examples)))
    if cfg.shuffle_dataset:
        random.Random(cfg.seed).shuffle(indices)
        log(f"shuffle_dataset=True: shuffled {len(indices)} examples (seed={cfg.seed})")
    else:
        log(f"shuffle_dataset=False: keeping corpus order ({len(indices)} examples)")

    max_steps = len(examples) // cfg.batch_size
    num_steps = min(cfg.num_steps, max_steps)
    if num_steps < cfg.num_steps:
        log(
            f"WARNING: num_steps={cfg.num_steps} exceeds max_steps={max_steps}; "
            f"clamping to {num_steps}."
        )
    log(
        f"Training: {num_steps} steps, global batch_size={cfg.batch_size}, "
        f"micro_batch_size={cfg.micro_batch_size}/GPU, lr={cfg.learning_rate}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    step = 0
    for batch_start in range(0, len(indices), cfg.batch_size):
        if step >= num_steps:
            break
        global_batch = indices[batch_start : batch_start + cfg.batch_size]
        # Shard the global batch across GPUs (strided). With world_size=1 this
        # is exactly the full batch, so single-GPU behaviour is unchanged.
        local_batch = global_batch[dist_info.rank :: dist_info.world_size]
        batch = [examples[i] for i in local_batch]

        n = len(batch)
        n_accum = max(1, math.ceil(n / cfg.micro_batch_size))
        local_loss_sum = 0.0
        local_weight_sum = 0.0

        for mb_start in range(0, n, cfg.micro_batch_size):
            mb = batch[mb_start : mb_start + cfg.micro_batch_size]
            n_micro = len(mb)
            max_len = max(len(e["tokens"]) for e in mb)

            padded_input = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_targets = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_weights = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            attention_mask = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            for i, e in enumerate(mb):
                seq_len = len(e["tokens"])
                padded_input[i, :seq_len] = torch.tensor(e["tokens"], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(
                    e["targets"], dtype=torch.long
                )
                padded_weights[i, :seq_len] = torch.tensor(
                    e["weights"], dtype=torch.float32
                )
                attention_mask[i, :seq_len] = 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids=padded_input,
                    attention_mask=attention_mask,
                    labels=padded_targets,
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce  # type: ignore[attr-defined]
                weighted_loss = per_token_ce * padded_weights
                weight_sum_t = padded_weights.sum()
                loss_sum_t = weighted_loss.sum()
                loss = (
                    loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0
                )

            (loss / n_accum).backward()
            local_loss_sum += loss_sum_t.item()
            local_weight_sum += weight_sum_t.item()
            del loss, per_token_ce, weighted_loss

        # ── Average gradients across GPUs (no-op when single-GPU) ────
        if dist_info.distributed:
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= dist_info.world_size

        # ── Optimizer step ──────────────────────────────────────────
        lr = cfg.learning_rate * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        tie_grads()  # keep MoE expert grads identical before clip+step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=cfg.grad_clip_norm
        )
        optimizer.step()
        optimizer.zero_grad()

        # ── Global loss for logging ─────────────────────────────────
        if dist_info.distributed:
            stat = torch.tensor([local_loss_sum, local_weight_sum], device=device)
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
            loss_sum, weight_sum = stat[0].item(), stat[1].item()
        else:
            loss_sum, weight_sum = local_loss_sum, local_weight_sum
        loss_mean = loss_sum / weight_sum if weight_sum > 0 else 0.0

        step += 1
        log(
            f"  step {step}/{num_steps}: loss:mean={loss_mean:.6f}, "
            f"grad_norm={grad_norm:.4f}, lr={lr:.2e}"
        )

        # ── Periodic checkpoint ─────────────────────────────────────
        if cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0:
            if dist_info.is_main:
                ckpt_dir = os.path.join(cfg.output_dir, f"step_{step}")
                save_adapter(cfg, model, training_log, log, save_dir=ckpt_dir)
            if dist_info.distributed:
                dist.barrier()

    log(
        f"Training complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )

    if dist_info.is_main:
        save_adapter(cfg, model, training_log, log)


def save_adapter(
    cfg: argparse.Namespace,
    model,
    training_log: list[str],
    log,
    save_dir: str | None = None,
) -> None:
    """Save the LoRA adapter, renaming lm_head keys to the submission convention."""
    from safetensors.torch import load_file, save_file

    save_dir = save_dir if save_dir is not None else cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)
    for _f in os.listdir(save_dir):
        if _f.startswith("adapter"):
            os.remove(os.path.join(save_dir, _f))
    model.save_pretrained(save_dir)

    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    tensors = load_file(st_path)
    renamed = {
        k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
        for k, v in tensors.items()
    }
    save_file(renamed, st_path)

    with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
        f.write("\n".join(training_log) + "\n")

    # Clean unsloth compiled cache.
    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil

        shutil.rmtree(_ucache)
    log(f"Saved adapter to {save_dir}")


def main() -> None:
    cfg = parse_args()
    dist_info = setup_distributed()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    try:
        train(cfg, dist_info)
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
