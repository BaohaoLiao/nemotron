"""Leave-one-out query-rotation augmenter (all categories).

Each problem is N example (input -> output) pairs that demo a hidden rule, plus
one held-out (question, answer) pair governed by the *same* rule. Any pair can
therefore serve as the query, so we mint a new, equally well-determined problem
by rotating which pair is held out:

    selected      = one original example pair (chosen by a seeded RNG)
    new_examples  = shuffle(original_examples - {selected} + [(question, answer)])
    new_question  = selected.input_value
    new_answer    = selected.output_value

i.e. the original held-out (question, answer) pair is folded back into the
examples (at a random position, since all examples are shuffled together), and
one former example becomes the new query. The example count is preserved (still
N), so the rotated problem is as determined as the original.

We generate at most ONE rotation per problem. We run the category's v1 solver on
the rotated problem and keep it only if the solver's answer matches
``new_answer`` under the official grader (``reasoning.compare_answer``: numeric
rel-tol 1e-2, strict for binary, case-insensitive otherwise -- so float-output
categories like gravity / unit_conversion pass within tolerance). A wrong answer
means the rotated example set no longer pins the rule well, so we re-select a
*different* pair as the query and retry -- up to ``MAX_RETRIES`` (3) distinct
selections. The first that verifies is kept; if all retries fail it is dropped.

We rotate only the categories where the v1 solver has accuracy headroom
(bit_manipulation and the cryptarithm / equation_numeric families); cipher,
gravity, numeral and unit_conversion are skipped because v1 already solves them
near-perfectly. The ``*_guess`` families are included: rotation folds the
original (question, answer) pair into the examples, so a guess problem's novel
query operator becomes represented in the examples and the new query (a former
example) is no longer novel -- the rotated problem is an ordinary solvable
problem with a ground-truth label, and only solver-verified rotations are kept.
test_500 problems are skipped so no held-out problem leaks into training.

Each category's prompt template (example line format and query line) is
*derived* from the original prompt and validated by reconstructing the original
prompt exactly; a problem whose prompt cannot be reconstructed is skipped, so a
malformed rotation is never emitted. The completion is the full v1 reasoning
trace ending in ``\\boxed{...}``; corpus.py wraps boxed augmentations exactly
like reasoning entries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from reasoning import compare_answer, extract_answer
from reasoners.store_types import Example, Problem
from reasoners.versions import get_solver

REPO_ROOT = Path(__file__).parent.parent
PROBLEMS_DIR = REPO_ROOT / "problems"
TEST_CSV = REPO_ROOT / "data" / "test_500.csv"
OUTPUT_DIR = REPO_ROOT / "augmentations"

# Categories to rotate. cipher / gravity / numeral / unit_conversion are
# intentionally omitted -- their v1 solver is already ~100% accurate, so rotation
# adds little. We rotate only the categories where v1 has accuracy headroom.
# Rotation makes a guess problem's novel query operator appear in the examples,
# so guess problems become ordinary solvable rotations; only solver-verified
# ones are kept.
CATEGORIES = (
    "bit_manipulation",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
)
SOLVER_VERSION = "v1"
DEFAULT_SEED = 0
MAX_RETRIES = 3

# Fraction of generated rotations to keep per category (deterministic, seeded
# subsample applied after generation). Categories absent here keep all (1.0).
# bit_manipulation produces by far the most rotations, so we halve it to keep
# the augmentation mix balanced across categories.
KEEP_FRACTION = {"bit_manipulation": 0.5}

# Placeholders for template derivation; never occur in problem data.
_IN = "\x00"
_OUT = "\x01"


def _subsample_by_category(problems: list[dict], seed: int) -> list[dict]:
    """Deterministically keep only ``KEEP_FRACTION[cat]`` of each category.

    Selection is seeded per category and order-preserving, so the kept set is
    reproducible given ``seed`` and independent of the other categories.
    """
    if not KEEP_FRACTION:
        return problems
    keep_ids: set[int] = set()
    by_cat: dict[str, list[dict]] = {}
    for p in problems:
        by_cat.setdefault(p["category"], []).append(p)
    for cat, items in by_cat.items():
        frac = KEEP_FRACTION.get(cat, 1.0)
        if frac >= 1.0:
            keep_ids.update(id(p) for p in items)
            continue
        k = round(len(items) * frac)
        order = list(range(len(items)))
        random.Random(f"{seed}:subsample:{cat}").shuffle(order)
        keep_ids.update(id(items[i]) for i in order[:k])
    return [p for p in problems if id(p) in keep_ids]


def _render(template: str, inp: str, out: str) -> str:
    return template.replace(_IN, inp).replace(_OUT, out)


def _make_rebuilder(prompt: str, examples: list[Example], question: str):
    """Return ``rebuild(new_examples, new_question) -> str`` or ``None``.

    Derives the example-line and query-line templates from the original prompt
    and validates them by reconstructing the original prompt exactly. Returns
    ``None`` (skip the problem) if the prompt does not match the expected shape.
    """
    lines = prompt.split("\n")
    n = len(examples)
    if n == 0:
        return None

    # Find the contiguous block of n lines that render the examples in order.
    start = None
    for s in range(0, len(lines) - n + 1):
        if all(
            examples[i].input_value in lines[s + i]
            and examples[i].output_value in lines[s + i]
            for i in range(n)
        ):
            start = s
            break
    if start is None:
        return None

    # Derive the example-line template from the first example and verify it
    # renders every example line exactly.
    e0 = examples[0]
    tmpl = lines[start].replace(e0.input_value, _IN, 1).replace(e0.output_value, _OUT, 1)
    for i in range(n):
        if (
            _render(tmpl, examples[i].input_value, examples[i].output_value)
            != lines[start + i]
        ):
            return None

    # Find and templatize the query line (first line after the block that
    # contains the question value).
    q_idx = None
    for j in range(start + n, len(lines)):
        if question in lines[j]:
            q_idx = j
            break
    if q_idx is None:
        return None
    q_tmpl = lines[q_idx].replace(question, _IN, 1)
    if q_tmpl.replace(_IN, question) != lines[q_idx]:
        return None

    header = lines[:start]
    between = lines[start + n : q_idx]
    after = lines[q_idx + 1 :]

    def rebuild(new_examples: list[Example], new_question: str) -> str:
        ex_lines = [_render(tmpl, e.input_value, e.output_value) for e in new_examples]
        return "\n".join(
            header + ex_lines + between + [q_tmpl.replace(_IN, new_question)] + after
        )

    # Validate: reconstructing the original must reproduce the prompt exactly.
    if rebuild(examples, question) != prompt:
        return None
    return rebuild


def _load_test_ids() -> set[str]:
    if not TEST_CSV.exists():
        return set()
    with open(TEST_CSV, newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def generate(
    seed: int = DEFAULT_SEED, max_retries: int = MAX_RETRIES
) -> list[dict]:
    """Generate one rotation per problem (correct trace preferred, else wrong).

    For each problem, try up to ``max_retries`` distinct query selections (in a
    seeded random order). Keep the first the category's v1 solver answers
    correctly (``correct=True``); if none verify, keep the first usable
    wrong-answer trace (``correct=False``) so the correct-vs-correct+wrong
    ablation has both. Drops a problem only if its prompt cannot be
    reconstructed or the solver returns no usable trace at all.
    """
    test_ids = _load_test_ids()
    problems: list[dict] = []
    # Per-category counters:
    # [attempted, correct, retried_correct, wrong, skip_format, no_fit].
    stats: dict[str, list[int]] = {c: [0, 0, 0, 0, 0, 0] for c in CATEGORIES}

    for path in sorted(PROBLEMS_DIR.glob("*.jsonl")):
        payload = json.loads(path.read_text().splitlines()[0])
        cat = payload.get("category")
        if cat not in CATEGORIES:
            continue
        pid = str(payload["id"])
        if pid in test_ids:
            continue
        orig = Problem.from_payload(payload)
        if len(orig.examples) < 1:
            continue

        rebuild = _make_rebuilder(orig.prompt, orig.examples, orig.question)
        if rebuild is None:
            stats[cat][4] += 1
            continue

        solver = get_solver(cat, SOLVER_VERSION)
        stats[cat][0] += 1
        rng = random.Random(f"{seed}:{pid}")
        order = list(range(len(orig.examples)))
        rng.shuffle(order)

        # One rotation per problem: try up to max_retries query selections and
        # keep the first the solver answers correctly. If none verify, keep the
        # first usable (wrong-answer) trace instead, tagged correct=False, so the
        # "train on correct + wrong" ablation has it; the correct-only ablation
        # filters it out at train time. The wrong trace keeps the solver's own
        # (wrong) boxed answer -- the reasoning is unchanged, only the label is.
        correct_res: tuple[str, str] | None = None
        wrong_res: tuple[str, str] | None = None
        for attempt, idx in enumerate(order[:max_retries]):
            selected = orig.examples[idx]
            new_examples = [e for i, e in enumerate(orig.examples) if i != idx]
            new_examples.append(Example(orig.question, orig.answer))
            # Shuffle all examples together so the folded-in pair is not always
            # last (avoids a positional cue).
            rng.shuffle(new_examples)
            new_question = selected.input_value
            new_answer = selected.output_value

            rotated = Problem(
                id=pid,
                category=cat,
                examples=new_examples,
                question=new_question,
                answer=new_answer,
            )
            trace = solver(rotated)
            if trace is None:
                continue
            if compare_answer(new_answer, extract_answer(trace)):
                correct_res = (trace, rebuild(new_examples, new_question))
                if attempt > 0:
                    stats[cat][2] += 1
                break
            if wrong_res is None:
                wrong_res = (trace, rebuild(new_examples, new_question))

        if correct_res is not None:
            trace, new_prompt = correct_res
            is_correct = True
            stats[cat][1] += 1
        elif wrong_res is not None:
            trace, new_prompt = wrong_res
            is_correct = False
            stats[cat][3] += 1
        else:
            # No usable trace from any attempt (solver returned None each time).
            stats[cat][5] += 1
            continue

        new_id = hashlib.sha256(f"rotation_{pid}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": new_id,
                "category": cat,
                "prompt": new_prompt,
                "completion": trace,
                "correct": is_correct,
            }
        )

    # Deterministically subsample over-represented categories (e.g. halve
    # bit_manipulation) before reporting/writing.
    n_before = len(problems)
    problems = _subsample_by_category(problems, seed)
    n_correct = sum(1 for p in problems if p["correct"])
    n_wrong = len(problems) - n_correct
    print(f"[rotation] solver={SOLVER_VERSION}, max_retries={max_retries}")
    print(
        f"[rotation] {'category':24s} {'attempt':>7} {'correct':>7} "
        f"{'retried':>7} {'wrong':>6} {'skipfmt':>8} {'nofit':>6}"
    )
    for c in CATEGORIES:
        a, k, r, w, s, nf = stats[c]
        print(f"[rotation] {c:24s} {a:7d} {k:7d} {r:7d} {w:6d} {s:8d} {nf:6d}")
    if KEEP_FRACTION and n_before != len(problems):
        kept_by_cat = Counter(p["category"] for p in problems)
        keep_summary = ", ".join(
            f"{c}={kept_by_cat[c]}(x{KEEP_FRACTION[c]})"
            for c in sorted(KEEP_FRACTION)
            if c in kept_by_cat
        )
        print(
            f"[rotation] subsampled {n_before} -> {len(problems)} traces "
            f"[{keep_summary}]"
        )
    print(
        f"[rotation] TOTAL: {len(problems)} traces "
        f"(correct={n_correct}, wrong={n_wrong})"
    )
    return problems


def _write(problems: list[dict]) -> None:
    """Write rotation files into augmentations/ without disturbing other files.

    Each file carries a ``[correct]`` marker (true/false) so the corpus and
    training can distinguish solver-correct rotations from wrong-answer ones.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    for p in problems:
        path = OUTPUT_DIR / f"{p['id']}.txt"
        correct = "true" if p["correct"] else "false"
        path.write_text(
            f"[category]\n{p['category']}\n[prompt]\n{p['prompt']}\n"
            f"[completion]\n{p['completion']}\n[correct]\n{correct}\n"
        )
    print(f"[rotation] wrote {len(problems)} files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    _write(generate())
