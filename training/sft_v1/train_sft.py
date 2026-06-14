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
        help="LoRA target modules. The Mamba mixer's `out_proj` is included by "
        "default. With the NemotronH fused training kernel "
        "(`mamba_split_conv1d_scan_combined`), out_proj is applied *inside* the "
        "kernel from the raw base weight (`outproj_weight=self.out_proj.weight`), "
        "so its LoRA wrapper would never be called and lora_B would stay exactly "
        "0 (a dead adapter) -- regardless of gradient checkpointing. Whenever "
        "`out_proj` is targeted, the build auto-sets `use_mem_eff_path=False` on "
        "the Mamba mixers so out_proj runs as a real module (keeps the fast conv "
        "+ SSD scan kernels, only un-fuses the final projection). Drop `out_proj` "
        "to skip the un-fuse and its small per-layer overhead.",
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
    g.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of passes over the dataset (reshuffled each epoch when "
        "--shuffle_dataset).",
    )
    g.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Optional hard cap on optimizer steps. 0 = run all --num_epochs "
        "fully; >0 stops early at whichever limit is reached first.",
    )
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
    g.add_argument(
        "--head_embedding_learning_rate",
        type=float,
        default=None,
        help="Learning rate for lm_head and embedding LoRA params. Defaults to --learning_rate.",
    )
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
    g.add_argument(
        "--versions_config",
        default=str(REPO_ROOT / "versions.json"),
        help="category->version JSON; trains only on matching corpus entries.",
    )
    g.add_argument(
        "--versions",
        nargs="*",
        default=[],
        metavar="CAT=VER",
        help="Override training version for specific categories.",
    )
    g.add_argument(
        "--max_gen_tokens",
        type=int,
        default=7680,
        help="Drop corpus entries whose completion (the tokens the model must "
        "generate, ending in \\boxed{...}<|im_end|>) exceeds this budget, so "
        "every training trace reaches its boxed answer within the inference "
        "generation window. 0 disables the filter.",
    )
    g.add_argument("--train_csv", default=str(REPO_ROOT / "train.csv"))
    g.add_argument(
        "--original_problems_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only problem_ids present in --train_csv.",
    )
    g.add_argument(
        "--aug_sample",
        "--aug-sample",
        type=int,
        default=-1,
        metavar="N",
        help="Cap how many augmentation (version='raw') examples to train on, "
        "stratified across the augmentation categories proportionally to their "
        "sizes. N<0 (default) keeps all augmentations; N=0 uses none. Only "
        "augmentation examples are sampled -- reasoning examples are untouched. "
        "Selection is deterministic given --seed.",
    )
    g.add_argument(
        "--aug_categories",
        "--aug-categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help="Whitelist of augmentation (version='raw') categories to keep. When "
        "given, raw entries whose category is NOT listed are dropped (reasoning "
        "entries are untouched). Use this to train on only the rotation "
        "augmentations (which carry real category names like 'bit_manipulation', "
        "'cryptarithm_deduce') and exclude the synthetic string-drills "
        "('matching', 'concatenation', 'splitting', 'spelling', 'lstrip'). "
        "Default (unset) keeps every augmentation category.",
    )
    g.add_argument(
        "--aug_include_wrong",
        "--aug-include-wrong",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include rotation augmentations whose solver answer was wrong "
        "(corpus field aug_correct=False). Default False trains only on "
        "answer-correct augmentations. Set True for the 'correct + wrong' "
        "ablation. Reasoning entries and string-drills are aug_correct=True and "
        "always kept.",
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


def _load_version_selection(cfg: argparse.Namespace, log) -> dict[str, str] | None:
    """Read the per-category version selection for training, or ``None`` to use
    every entry in the corpus index. Precedence: ``--versions cat=ver`` overrides
    > ``--versions_config`` file > none. Augmentation entries (version ``raw``)
    are always kept regardless of selection."""
    selection: dict[str, str] = {}
    cfg_path = getattr(cfg, "versions_config", None)
    if cfg_path:
        p = Path(cfg_path)
        if p.is_file():
            with open(p) as f:
                selection.update(json.load(f))
        else:
            log(f"WARNING: versions_config not found: {p}")
    for item in getattr(cfg, "versions", []) or []:
        cat, ver = item.split("=", 1)
        selection[cat] = ver
    return selection or None


def _stratified_sample(items: list[dict], n: int, key, seed: int) -> list[dict]:
    """Proportional-allocation stratified sample of ``n`` items.

    Groups ``items`` by ``key(item)`` and allocates ``n`` across the groups in
    proportion to each group's size (largest-remainder rounding so the parts sum
    to exactly ``n``), then randomly draws that many from each group with a
    ``random.Random(seed)`` for reproducibility. Returns all items when
    ``n >= len(items)``.
    """
    if n >= len(items):
        return list(items)
    groups: dict = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    total = len(items)
    quotas: dict = {}
    remainders: dict = {}
    for k, grp in groups.items():
        exact = n * len(grp) / total
        quotas[k] = int(exact)  # floor
        remainders[k] = exact - quotas[k]
    # Hand the leftover (from flooring) to the largest remainders; ties broken by
    # group key so the allocation is deterministic.
    leftover = n - sum(quotas.values())
    for k in sorted(groups, key=lambda k: (-remainders[k], str(k)))[:leftover]:
        quotas[k] += 1
    rng = random.Random(seed)
    selected: list[dict] = []
    for k, grp in groups.items():
        selected.extend(rng.sample(grp, min(quotas[k], len(grp))))
    return selected


def load_examples(cfg: argparse.Namespace, log) -> list[dict]:
    """Load and pre-tokenize the training corpus into next-token examples."""
    index_path = Path(cfg.corpus_index)
    corpus_dir = Path(cfg.corpus_dir)
    assert index_path.is_file(), f"Corpus index not found: {index_path}"
    assert corpus_dir.is_dir(), f"Corpus dir not found: {corpus_dir}"

    # Segment paths in a version-aware index are stored relative to the repo root
    # (e.g. "corpus/<category>/<version>/<id>/synthetic.jsonl"); resolve against it.
    seg_root = index_path.resolve().parent
    if seg_root.name == "data":
        seg_root = seg_root.parent

    version_selection = _load_version_selection(cfg, log)
    if version_selection:
        log(f"Training version selection: {version_selection}")

    index: list[dict] = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(json.loads(line))

    examples: list[dict] = []
    skipped_version = 0
    skipped_budget = 0
    for rec in index:
        if not rec.get("included", True):
            continue
        # Per-category version filtering. Entries without a version (legacy) or
        # the augmentation version "raw" are always kept.
        rec_version = rec.get("version")
        rec_category = rec.get("category")
        if (
            version_selection is not None
            and rec_version is not None
            and rec_version != "raw"
            and rec_category in version_selection
            and version_selection[rec_category] != rec_version
        ):
            skipped_version += 1
            continue
        pid = rec["problem_id"]
        segment = rec.get("segment", cfg.segment_name)
        # New-style index stores a repo-root-relative path; old style stored just
        # the segment filename under corpus_dir/<pid>/.
        if "/" in segment:
            seg_path = seg_root / segment
        else:
            seg_path = corpus_dir / pid / segment
        if not seg_path.is_file():
            log(f"WARNING: missing segment for {pid}: {seg_path}")
            continue
        tokens, mask = _load_segment(seg_path)
        if not tokens:
            continue
        # Generation-budget filter. The completion (unmasked tokens) is what the
        # model must produce at inference; its final tokens are \boxed{...} and
        # <|im_end|>. If the completion exceeds the generation window the boxed
        # answer is never reached (and corpus traces over the corpus limit were
        # already tail-truncated, so they end mid-reasoning with no answer at
        # all). Drop these so every trained trace finishes within budget.
        if cfg.max_gen_tokens and sum(mask) > cfg.max_gen_tokens:
            skipped_budget += 1
            continue
        if len(tokens) > cfg.max_seq_len:
            tokens = tokens[: cfg.max_seq_len]
            mask = mask[: cfg.max_seq_len]
        if not any(mask):
            continue
        examples.append(
            {
                "problem_id": pid,
                "category": rec_category,
                "version": rec_version,
                "aug_correct": rec.get("aug_correct", True),
                "tokens": tokens[:-1],
                "targets": tokens[1:],
                "weights": [float(m) for m in mask[1:]],
            }
        )

    if version_selection and skipped_version:
        log(
            f"version selection skipped {skipped_version} corpus entries "
            f"(non-matching versions)"
        )

    if cfg.max_gen_tokens and skipped_budget:
        log(
            f"generation budget (max_gen_tokens={cfg.max_gen_tokens}) skipped "
            f"{skipped_budget} corpus entries whose completion exceeds the budget"
        )

    if cfg.original_problems_only:
        import csv

        with open(cfg.train_csv) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        # Keep augmentation (version="raw") examples regardless -- their ids are
        # never in train_csv, so otherwise this would drop all of them before
        # --aug_sample can subsample them. Only reasoning examples are restricted
        # to the original problem ids.
        examples = [
            e
            for e in examples
            if e.get("version") == "raw" or e["problem_id"] in original_ids
        ]
        log(
            f"original_problems_only=True: filtered {before} -> {len(examples)} "
            f"examples using {len(original_ids)} ids from {cfg.train_csv} "
            f"(augmentation examples exempt)"
        )

    # Whitelist of augmentation categories. Drops raw entries whose category is
    # not listed (reasoning entries untouched) -- e.g. to keep only the rotation
    # augmentations and exclude the synthetic string-drills.
    if cfg.aug_categories is not None:
        allowed = set(cfg.aug_categories)
        before = len(examples)
        examples = [
            e
            for e in examples
            if e.get("version") != "raw" or e.get("category") in allowed
        ]
        log(
            f"aug_categories={sorted(allowed)}: filtered {before} -> "
            f"{len(examples)} examples (kept only listed augmentation categories)"
        )

    # Correct-vs-correct+wrong ablation. Wrong-answer rotation augmentations have
    # aug_correct=False; drop them unless --aug_include_wrong. Reasoning entries
    # and string-drills are aug_correct=True and unaffected.
    if not cfg.aug_include_wrong:
        before = len(examples)
        examples = [e for e in examples if e.get("aug_correct", True)]
        if before != len(examples):
            log(
                f"aug_include_wrong=False: dropped {before - len(examples)} "
                f"wrong-answer augmentation examples"
            )

    # Stratified cap on augmentation (version="raw") examples. Reasoning examples
    # are never sampled; only augmentations are thinned, proportionally across
    # their categories, so a small --aug_sample still covers every aug category.
    if cfg.aug_sample >= 0:
        aug = [e for e in examples if e.get("version") == "raw"]
        if len(aug) > cfg.aug_sample:
            from collections import Counter

            selected = _stratified_sample(
                aug, cfg.aug_sample, key=lambda e: e["category"], seed=cfg.seed
            )
            keep = {id(e) for e in selected}
            examples = [
                e for e in examples if e.get("version") != "raw" or id(e) in keep
            ]
            picked = Counter(e["category"] for e in selected)
            breakdown = ", ".join(f"{c}={picked[c]}" for c in sorted(picked))
            log(
                f"aug_sample={cfg.aug_sample}: kept {len(selected)}/{len(aug)} "
                f"augmentation examples (stratified: {breakdown})"
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

    for n, p in model.named_parameters():
        if "lora_embedding_" in n:
            p.requires_grad_(True)

    # ── Patch Mamba CUDA fast path ───────────────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    log("Patched is_fast_path_available = True")

    # ── Un-fuse out_proj when its LoRA is requested ──────────────────
    # The fused training kernel `mamba_split_conv1d_scan_combined` applies
    # out_proj internally from the raw base weight (outproj_weight=...weight),
    # bypassing the PEFT LoRA wrapper -> out_proj.lora_B never gets a gradient
    # and stays 0. Setting use_mem_eff_path=False routes the mixer through the
    # branch that calls `self.out_proj(scan_output)` as a real module (still
    # using the fast conv1d + chunk-scan kernels), so its LoRA trains.
    if "out_proj" in cfg.target_modules:
        n_unfused = 0
        for _m in model.modules():
            if hasattr(_m, "use_mem_eff_path") and hasattr(_m, "out_proj"):
                _m.use_mem_eff_path = False
                n_unfused += 1
        log(
            f"out_proj in target_modules: set use_mem_eff_path=False on "
            f"{n_unfused} Mamba mixers so out_proj LoRA receives gradients "
            f"(fast conv + scan kernels retained, only final projection un-fused)"
        )

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    # Only when lm_head is requested in --target_modules.
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if "lm_head" not in cfg.target_modules:
        log("lm_head not in target_modules; skipping lm_head LoRA")
    elif not isinstance(_lm_head, LoraLinear):
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
        if isinstance(lm_head, LoraLinear):
            base_w = lm_head.base_layer.weight
            lora_A = lm_head.lora_A["default"].weight
            lora_B = lm_head.lora_B["default"].weight
            scaling = lm_head.scaling["default"]
            lm_weight = base_w + scaling * lora_B @ lora_A
        else:
            lm_weight = lm_head.weight
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


def build_optimizer_param_groups(cfg: argparse.Namespace, model, log) -> list[dict]:
    """Split trainable params so head / embedding LoRAs can use a custom LR."""
    main_lr = cfg.learning_rate
    head_embedding_lr = (
        cfg.head_embedding_learning_rate
        if cfg.head_embedding_learning_rate is not None
        else main_lr
    )

    main_params = []
    head_embedding_params = []
    main_count = 0
    head_embedding_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_params = param.numel()
        is_lm_head_lora = "lm_head" in name and ".lora_" in name
        is_embedding_lora = "lora_embedding_" in name or (
            "embed_tokens" in name and ".lora_" in name
        )
        if is_lm_head_lora or is_embedding_lora:
            head_embedding_params.append(param)
            head_embedding_count += n_params
        else:
            main_params.append(param)
            main_count += n_params

    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": main_lr, "initial_lr": main_lr})
    if head_embedding_params:
        groups.append(
            {
                "params": head_embedding_params,
                "lr": head_embedding_lr,
                "initial_lr": head_embedding_lr,
            }
        )

    log(
        "Optimizer groups: "
        f"main={main_count:,} params lr={main_lr:.2e}; "
        f"head_embedding_lora={head_embedding_count:,} params "
        f"lr={head_embedding_lr:.2e}"
    )
    return groups


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
    log(
        f"shuffle_dataset={cfg.shuffle_dataset}: "
        f"{'reshuffling each epoch' if cfg.shuffle_dataset else 'keeping corpus order'} "
        f"({len(indices)} examples)"
    )

    steps_per_epoch = len(examples) // cfg.batch_size
    epoch_cap = cfg.num_epochs * steps_per_epoch
    if cfg.num_steps > 0:
        num_steps = min(cfg.num_steps, epoch_cap)
        if cfg.num_steps > epoch_cap:
            log(
                f"WARNING: num_steps={cfg.num_steps} exceeds "
                f"num_epochs*steps_per_epoch={epoch_cap}; clamping to {num_steps}."
            )
    else:
        num_steps = epoch_cap
    log(
        f"Training: {num_steps} steps ({steps_per_epoch} steps/epoch x "
        f"{cfg.num_epochs} epochs), global batch_size={cfg.batch_size}, "
        f"micro_batch_size={cfg.micro_batch_size}/GPU, lr={cfg.learning_rate}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    param_groups = build_optimizer_param_groups(cfg, model, log)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    step = 0
    stop = False
    for epoch in range(cfg.num_epochs):
        if stop:
            break
        if cfg.shuffle_dataset:
            random.Random(cfg.seed + epoch).shuffle(indices)
            log(
                f"epoch {epoch + 1}/{cfg.num_epochs}: reshuffled (seed={cfg.seed + epoch})"
            )
        for batch_start in range(0, len(indices), cfg.batch_size):
            if step >= num_steps:
                stop = True
                break
            global_batch = indices[batch_start : batch_start + cfg.batch_size]
            # Shard the global batch across GPUs (strided). With world_size=1
            # this is exactly the full batch, so single-GPU is unchanged.
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
                    padded_input[i, :seq_len] = torch.tensor(
                        e["tokens"], dtype=torch.long
                    )
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
                        loss_sum_t / weight_sum_t
                        if weight_sum_t > 0
                        else loss_sum_t * 0.0
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
            lr_factor = 1 - step / num_steps
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_factor
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
                f"  epoch {epoch + 1}/{cfg.num_epochs} step {step}/{num_steps}: "
                f"loss:mean={loss_mean:.6f}, grad_norm={grad_norm:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
        final_dir = os.path.join(cfg.output_dir, f"step_{step}")
        save_adapter(cfg, model, training_log, log, save_dir=final_dir)


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
    model.save_pretrained(save_dir, save_embedding_layers=False)

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
