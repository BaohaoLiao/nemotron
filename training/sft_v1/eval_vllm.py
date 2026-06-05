"""Evaluate the base model (optionally with a trained LoRA adapter) on test_500.

Uses vLLM for fast batched generation over the held-out `data/test_500.csv`
problems. Prompts are built with the SAME chat template / suffix used to create
the training corpus (`corpus.py`), and answers are scored with the SAME logic
used elsewhere in the repo (`reasoning.compare_answer`).

Examples:
    # Base model only
    python eval_vllm.py

    # With a trained LoRA adapter (the dir written by train_sft.py)
    python eval_vllm.py --lora output/weights

    # Quick smoke test on the first 20 problems
    python eval_vllm.py --limit 20

    # Multi-GPU tensor parallelism
    python eval_vllm.py --tensor-parallel-size 4

Results (per-problem prediction + correctness) are written to a JSONL file and a
per-category accuracy table is printed at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Repo root = .../nemotron (two levels up from training/sft_v1/).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Must match the official Kaggle metric prompt (and corpus.py) so eval prompts
# mirror both training and the leaderboard scoring.
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction + scoring. Mirrors the official Kaggle metric
# (`extract_final_answer` / `verify`) so local accuracy matches the leaderboard.
# ─────────────────────────────────────────────────────────────────────────────
def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer, prioritizing the last ``\boxed{...}``.

    For each ``\boxed{`` occurrence, take everything up to the last ``}`` before
    the next ``\boxed{`` (or end of text). This handles answers that themselves
    contain ``}`` (the model writes them literally, e.g. ``\boxed{}52}`` for the
    answer ``}52``) as well as nested LaTeX like ``\boxed{\frac{1}{2}}``. Falls
    back to common phrasings, then the last number, then the last line.
    """
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found.
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()

    # If no structured format is found, extract the last valid number.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    # Fallback: last non-empty line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify a prediction against the stored answer (official metric logic).

    Binary strings compare exactly; numbers compare within a relative tolerance
    (1e-2); everything else is a case-insensitive string match.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_categories(index_path: Path) -> dict[str, str]:
    """Map problem_id -> category from the corpus index (best-effort)."""
    cats: dict[str, str] = {}
    if not index_path.is_file():
        return cats
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cats[rec["problem_id"]] = rec.get("category", "unknown")
    return cats


def load_test_rows(csv_path: Path, limit: int | None) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Generation (single engine + optional data-parallel sharding)
# ─────────────────────────────────────────────────────────────────────────────
def build_prompts(tokenizer, rows: list[dict], enable_thinking: bool) -> list[str]:
    """Render chat prompts the same way the official Kaggle metric does."""
    prompts: list[str] = []
    for row in rows:
        user_content = row["prompt"] + PROMPT_SUFFIX
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)
    return prompts


def generate_completions(cfg: argparse.Namespace, rows: list[dict]) -> list[str]:
    """Load one vLLM engine and return the completion text for each row.

    Engine flags mirror the official Kaggle metric (dtype='auto', prefix
    caching + chunked prefill on).
    """
    from vllm import LLM, SamplingParams

    use_lora = bool(cfg.lora)
    llm = LLM(
        model=cfg.model,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_num_seqs=cfg.max_num_seqs,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="auto",
        max_model_len=cfg.max_model_len,
        trust_remote_code=True,
        enable_lora=use_lora,
        max_lora_rank=cfg.max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=cfg.seed,
    )

    tokenizer = llm.get_tokenizer()
    prompts = build_prompts(tokenizer, rows, cfg.enable_thinking)
    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )

    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, cfg.lora)

    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def _visible_gpus() -> list[str]:
    """List of visible GPU ids, honouring CUDA_VISIBLE_DEVICES if set."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _dp_worker(cfg, rows_shard, indices, gpu_ids, queue):
    """Data-parallel worker: pin to its GPU(s), generate, return completions.

    Runs in a fresh (spawned) interpreter, so setting CUDA_VISIBLE_DEVICES here
    — before vLLM/torch import CUDA inside generate_completions — pins this
    replica to exactly its slice of GPUs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    try:
        completions = generate_completions(cfg, rows_shard)
        queue.put((indices, completions, None))
    except Exception as exc:  # surface worker failures to the parent
        import traceback

        queue.put((indices, None, traceback.format_exc() or str(exc)))


def run_data_parallel(cfg: argparse.Namespace, rows: list[dict]) -> list[str]:
    """Shard `rows` across `--data-parallel-size` independent vLLM engines.

    Replica r owns rows[r::dp] (round-robin, so long and short problems spread
    evenly) and GPUs [r*tp : (r+1)*tp]. Each replica is a full model copy.
    """
    import multiprocessing as mp

    dp = cfg.data_parallel_size
    tp = cfg.tensor_parallel_size
    gpus = _visible_gpus()
    needed = dp * tp
    assert len(gpus) >= needed, (
        f"Need data_parallel_size*tensor_parallel_size={needed} GPUs, "
        f"but only {len(gpus)} visible: {gpus or '(none)'}"
    )

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = []
    for r in range(dp):
        indices = list(range(r, len(rows), dp))
        if not indices:
            continue
        shard = [rows[i] for i in indices]
        gpu_ids = gpus[r * tp : (r + 1) * tp]
        print(f"DP replica {r}: {len(shard)} problems on GPU(s) {','.join(gpu_ids)}")
        p = ctx.Process(target=_dp_worker, args=(cfg, shard, indices, gpu_ids, queue))
        p.start()
        procs.append(p)

    completions: list[str | None] = [None] * len(rows)
    for _ in procs:
        indices, comp, err = queue.get()
        if err is not None:
            for p in procs:
                p.terminate()
            raise RuntimeError(f"A data-parallel worker failed:\n{err}")
        for i, c in zip(indices, comp):
            completions[i] = c
    for p in procs:
        p.join()
    return [c if c is not None else "" for c in completions]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM eval on test_500 (optionally with a LoRA adapter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Base model (HF id or local path).",
    )
    p.add_argument(
        "--lora",
        default="",
        help="Path to a trained LoRA adapter dir (e.g. output/weights). "
        "Empty = evaluate the base model only.",
    )
    p.add_argument(
        "--test-csv",
        default=str(REPO_ROOT / "data" / "test_500.csv"),
        help="CSV with columns id,prompt,answer.",
    )
    p.add_argument(
        "--corpus-index",
        default=str(REPO_ROOT / "data" / "corpus.jsonl"),
        help="Corpus index used only to label problems by category.",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "eval_results.jsonl"),
        help="Where to write per-problem predictions (JSONL).",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Eval only first N problems."
    )

    # Sampling / generation.
    p.add_argument("--max-tokens", type=int, default=7680)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the thinking chat template (matches corpus.py).",
    )

    # vLLM engine.
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of independent vLLM engines (one full model copy each) to "
        "shard the test set across. Uses data_parallel_size*tensor_parallel_size "
        "GPUs total.",
    )
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-lora-rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()

    test_csv = Path(cfg.test_csv)
    assert test_csv.is_file(), f"Test CSV not found: {test_csv}"
    rows = load_test_rows(test_csv, cfg.limit)
    assert rows, "No rows loaded from test CSV."
    categories = load_categories(Path(cfg.corpus_index))

    use_lora = bool(cfg.lora)
    if use_lora:
        assert Path(cfg.lora).is_dir(), f"LoRA dir not found: {cfg.lora}"

    print(f"Loaded {len(rows)} problems from {test_csv}")
    print(f"Model      : {cfg.model}")
    print(f"LoRA       : {cfg.lora or '(none, base model)'}")
    print(f"Thinking   : {cfg.enable_thinking}")
    print(f"DP x TP    : {cfg.data_parallel_size} x {cfg.tensor_parallel_size}")
    print()

    if cfg.data_parallel_size > 1:
        completions = run_data_parallel(cfg, rows)
    else:
        completions = generate_completions(cfg, rows)

    # Score and write per-problem results.
    by_cat: dict[str, list[int]] = defaultdict(list)
    n_correct = 0
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out_f:
        for row, completion in zip(rows, completions):
            predicted = extract_final_answer(completion)
            correct = verify(row["answer"], predicted)
            n_correct += int(correct)
            cat = categories.get(row["id"], "unknown")
            by_cat[cat].append(int(correct))
            out_f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "category": cat,
                        "answer": row["answer"],
                        "predicted": predicted,
                        "correct": correct,
                        "completion": completion,
                    }
                )
                + "\n"
            )

    # Per-category accuracy table.
    total = len(rows)
    print()
    print("=" * 56)
    print(f"{'Category':<26}{'Correct':>8}{'Total':>7}{'Accuracy':>13}")
    print("-" * 56)
    for cat in sorted(by_cat):
        results = by_cat[cat]
        c, t = sum(results), len(results)
        print(f"{cat:<26}{c:>8}{t:>7}{c / t * 100:>12.1f}%")
    print("-" * 56)
    print(f"{'TOTAL':<26}{n_correct:>8}{total:>7}{n_correct / total * 100:>12.1f}%")
    print("=" * 56)
    print(f"\nWrote per-problem results to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
