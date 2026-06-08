"""Create synthetic training corpus from version-aware reasoning traces.

Reads traces written by reasoning.py at

    reasoning/<category>/<version>/<problem_id>.txt

and tokenises them into

    corpus/<category>/<version>/<problem_id>/synthetic.jsonl

recording the ``version`` of each entry in ``corpus.jsonl``. By default the
version per category is read from ``versions.json`` (see ``reasoners.versions``);
``--all-versions`` builds every version present on disk so training can pick.

The completion for each entry is:
    (reasoning text)</think>\\boxed{(answer)}<|im_end|>

Outputs:
- corpus.jsonl                                       - index (one row per built entry)
- corpus/<category>/<version>/<id>/synthetic.jsonl   - tokenised segment files

Usage:
    uv run corpus.py                       # build versions.json selection
    uv run corpus.py --config my.json
    uv run corpus.py --versions cryptarithm_deduce=v4
    uv run corpus.py --all-versions        # build every version found on disk
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from reasoners.versions import REGISTRY, available_versions, load_versions

TRAIN_CSV = Path(__file__).parent / "train.csv"
AUGMENTATIONS_DIR = Path(__file__).parent / "augmentations"
PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_INDEX = Path(__file__).parent / "corpus.jsonl"
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"

# Version tag used for augmentation entries (which have no solver version).
AUG_VERSION = "raw"

# Must match metric_reference.py / query.py
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

TOKEN_LIMIT = 8192


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def tokenize_prompt(
    prompt_text: str,
    chat_tokenizer: AutoTokenizer,
    *,
    suffix: str = PROMPT_SUFFIX,
) -> list[int]:
    """Tokenize a problem prompt using the chat template, matching query.py."""
    messages = [{"role": "user", "content": prompt_text + suffix}]
    return chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


@dataclass
class CorpusEntry:
    problem_id: str
    category: str
    version: str
    segment_path: str
    tokens: list[int]
    mask: list[int]
    masked_token_count: int
    unmasked_token_count: int
    answer: str
    included: bool = False

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "segment": self.segment_path,
            "category": self.category,
            "version": self.version,
            "masked_token_count": self.masked_token_count,
            "unmasked_token_count": self.unmasked_token_count,
            "token_count": self.token_count,
            "answer": self.answer,
            "included": self.included,
        }


def build_segments(
    tokens: list[int],
    mask: list[int],
) -> list[dict]:
    """Build segment list from tokens and mask."""
    if not tokens:
        return []

    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"

    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append(
                {
                    "type": current_type,
                    "pos": seg_start,
                    "tokens": tokens[seg_start:i],
                }
            )
            seg_start = i
            current_type = token_type

    segments.append(
        {
            "type": current_type,
            "pos": seg_start,
            "tokens": tokens[seg_start:],
        }
    )

    return segments


def _selected_pairs(args: argparse.Namespace) -> dict[str, list[str]]:
    """Return ``{category: [versions]}`` to build. By default one version per
    category from the config; ``--all-versions`` builds every version that has a
    ``reasoning/<category>/<version>/`` folder on disk."""
    if args.all_versions:
        out: dict[str, list[str]] = {}
        for cat in REGISTRY:
            vers = [
                v for v in available_versions(cat) if (REASONING_DIR / cat / v).is_dir()
            ]
            if vers:
                out[cat] = vers
        return out
    selected = load_versions(args.config)
    for item in args.versions:
        cat, ver = item.split("=", 1)
        selected[cat] = ver
    return {cat: [ver] for cat, ver in selected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="category->version JSON.")
    parser.add_argument("--versions", nargs="*", default=[], metavar="CAT=VER")
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Build every version present under reasoning/.",
    )
    args = parser.parse_args()

    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found. Run problems.py first.")
        return

    build_plan = _selected_pairs(args)

    # Load tokenizers
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
    )

    # Load problem prompts from train.csv
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # Load problem categories, grouped by category for iteration.
    ids_by_cat: dict[str, list[str]] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        ids_by_cat.setdefault(prob_raw["category"], []).append(prob_raw["id"])

    # Clean and recreate corpus directory
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    entries: list[CorpusEntry] = []
    oversize: dict[str, int] = {}

    # Iterate over (category, version) pairs and their reasoning traces.
    for category in sorted(build_plan):
        for version in build_plan[category]:
            ver_dir = REASONING_DIR / category / version
            if not ver_dir.is_dir():
                continue
            problem_ids = sorted(
                pid
                for pid in ids_by_cat.get(category, [])
                if (ver_dir / f"{pid}.txt").exists() and pid in prompts
            )
            for problem_id in problem_ids:
                answer = answers[problem_id]
                reasoning_text = (
                    (ver_dir / f"{problem_id}.txt").read_text().rstrip("\n")
                )

                # Extract answer from reasoning's \boxed{} so they match
                boxed_match = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
                reasoning_answer = boxed_match[-1] if boxed_match else answer
                completion_text = f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
                completion_ids = tokenizer.encode(
                    completion_text, add_special_tokens=False
                ).ids

                prompt_ids = tokenize_prompt(prompts[problem_id], chat_tokenizer)

                all_tokens = prompt_ids + completion_ids
                mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

                if len(all_tokens) > TOKEN_LIMIT:
                    # The boxed answer and <|im_end|> live at the very end of the
                    # completion. Tail-truncating to TOKEN_LIMIT would delete them
                    # and teach the model to never finish, so skip the trace
                    # entirely instead.
                    oversize[f"{category}/{version}"] = (
                        oversize.get(f"{category}/{version}", 0) + 1
                    )
                    continue

                unmasked_count = sum(mask)
                masked_count = len(mask) - unmasked_count

                rel_seg = f"corpus/{category}/{version}/{problem_id}/synthetic.jsonl"
                entry = CorpusEntry(
                    problem_id=problem_id,
                    category=category,
                    version=version,
                    segment_path=rel_seg,
                    tokens=all_tokens,
                    mask=mask,
                    masked_token_count=masked_count,
                    unmasked_token_count=unmasked_count,
                    answer=answer,
                    included=True,
                )

                segments = build_segments(all_tokens, mask)
                problem_dir = CORPUS_DIR / category / version / problem_id
                problem_dir.mkdir(parents=True, exist_ok=True)
                with open(problem_dir / "synthetic.jsonl", "w") as f:
                    for seg in segments:
                        json.dump(seg, f)
                        f.write("\n")

                entries.append(entry)

    if oversize:
        total_skipped = sum(oversize.values())
        print(
            f"Skipped {total_skipped} over-limit traces (> {TOKEN_LIMIT} tokens) "
            f"to avoid truncating the boxed answer:"
        )
        for key in sorted(oversize):
            print(f"  {key}: {oversize[key]}")

    # Process augmentations/*.txt (no reasoning, no \boxed{})
    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            # Parse [category], [prompt], and [completion] sections
            category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")

            problem_id = aug_path.stem

            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = tokenizer.encode(
                completion_text, add_special_tokens=False
            ).ids

            prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer, suffix="")

            all_tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            assert len(all_tokens) <= TOKEN_LIMIT, (
                f"augmented entry {problem_id} exceeds token limit: "
                f"{len(all_tokens)} > {TOKEN_LIMIT}"
            )

            unmasked_count = sum(mask)
            masked_count = len(mask) - unmasked_count

            entry = CorpusEntry(
                problem_id=problem_id,
                category=category,
                version=AUG_VERSION,
                segment_path=f"corpus/{AUG_VERSION}/{problem_id}/synthetic.jsonl",
                tokens=all_tokens,
                mask=mask,
                masked_token_count=masked_count,
                unmasked_token_count=unmasked_count,
                answer=completion,
                included=True,
            )

            segments = build_segments(all_tokens, mask)
            problem_dir = CORPUS_DIR / AUG_VERSION / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as sf:
                for seg in segments:
                    json.dump(seg, sf)
                    sf.write("\n")

            entries.append(entry)

    entries.sort(key=lambda e: (e.category, e.version, e.problem_id))

    # Write index JSONL
    with open(CORPUS_INDEX, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f)
            f.write("\n")

    # Stats grouped by category/version.
    cat_counts: dict[str, int] = {}
    cat_tokens: dict[str, int] = {}
    for e in entries:
        key = f"{e.category}/{e.version}"
        cat_counts[key] = cat_counts.get(key, 0) + 1
        cat_tokens[key] = cat_tokens.get(key, 0) + e.unmasked_token_count

    total_unmasked = sum(e.unmasked_token_count for e in entries)
    total_masked = sum(e.masked_token_count for e in entries)
    max_tokens = max((e.token_count for e in entries), default=0)

    print(f"Corpus (synthetic): {len(entries)} entries")
    print(f"Unmasked tokens: {total_unmasked:,}")
    print(f"Masked tokens:   {total_masked:,}")
    print(f"Max seq length:  {max_tokens:,}")
    print()
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]} runs, {cat_tokens[cat]:,} unmasked tokens")


if __name__ == "__main__":
    main()
