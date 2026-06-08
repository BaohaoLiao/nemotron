"""Generate deterministic reasoning text for each rule_found problem.

For every problem, the solver version is chosen per category from ``versions.json``
(see ``reasoners.versions``). Traces are written to

    reasoning/<category>/<version>/<problem_id>.txt

so different solver versions never overwrite each other and can be selected later
by corpus.py / training. Only the (category, version) folders being regenerated
are cleared, so other versions already on disk are preserved.

Usage:
    uv run reasoning.py                         # use versions.json
    uv run reasoning.py --config my.json        # use a different selection file
    uv run reasoning.py --versions cryptarithm_deduce=v4 bit_manipulation=v2
    uv run reasoning.py --all-versions          # generate EVERY registered version
    uv run reasoning.py --delete-investigations
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from reasoners.store_types import Problem
from reasoners.versions import (
    REGISTRY,
    all_pairs,
    get_solver,
    load_versions,
)

PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
INVESTIGATIONS_DIR = Path(__file__).parent / "investigations"
INVESTIGATION_CATEGORIES: set[str] = {
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}

SKIP_CATEGORIES: set[str] = set()



def extract_answer(reasoning_text: str) -> str:
    """Extract the answer from \\boxed{...}, matching metric_reference.extract_final_answer."""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


@dataclass
class CategoryCounts:
    rule_found: int = 0
    total: int = 0
    runtimes: list[float] = field(default_factory=list)


def _parse_version_overrides(items: list[str]) -> dict[str, str]:
    """Parse ``--versions cat=ver ...`` overrides into a dict."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--versions expects cat=ver, got {item!r}")
        cat, ver = item.split("=", 1)
        if cat not in REGISTRY:
            raise SystemExit(f"--versions: unknown category {cat!r}")
        if ver not in REGISTRY[cat]:
            raise SystemExit(
                f"--versions: {cat!r} has no version {ver!r} "
                f"(available: {sorted(REGISTRY[cat])})"
            )
        out[cat] = ver
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a category->version JSON (default: versions.json).",
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        default=[],
        metavar="CAT=VER",
        help="Override selected versions for specific categories.",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Generate EVERY registered version for every category.",
    )
    parser.add_argument(
        "--delete-investigations",
        action="store_true",
        help="Delete investigation files when answer is correct",
    )
    args = parser.parse_args()

    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found.")
        return

    # Resolve the (category, version) pairs to generate, and which version is the
    # "active" one per category (used to update problems.jsonl status/submission).
    selected = load_versions(args.config)
    selected.update(_parse_version_overrides(args.versions))
    if args.all_versions:
        pairs = all_pairs()
    else:
        pairs = [(cat, ver) for cat, ver in selected.items()]
    # Categories whose status we record in problems.jsonl (the active selection).
    active_version = dict(selected)

    # Read existing entries to preserve fields, then merge results back
    existing: dict[str, dict] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                existing[entry["id"]] = entry

    # Wipe only the (category, version) folders we are about to regenerate; leave
    # any other versions already on disk untouched.
    REASONING_DIR.mkdir(parents=True, exist_ok=True)
    for cat, ver in pairs:
        d = REASONING_DIR / cat / ver
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Group problem ids by category for efficient iteration.
    ids_by_cat: dict[str, list[str]] = {}
    for entry in existing.values():
        ids_by_cat.setdefault(entry["category"], []).append(entry["id"])

    stats: dict[str, bool] = {}
    category_stats: dict[str, CategoryCounts] = {}
    generated = 0
    skipped = 0

    for cat, ver in pairs:
        if cat in SKIP_CATEGORIES or cat not in REGISTRY:
            continue
        generator: Callable = get_solver(cat, ver)
        out_dir = REASONING_DIR / cat / ver
        is_active = active_version.get(cat) == ver
        stat_key = f"{cat}/{ver}"
        if stat_key not in category_stats:
            category_stats[stat_key] = CategoryCounts()

        for pid in ids_by_cat.get(cat, []):
            category_stats[stat_key].total += 1
            problem = Problem.load_from_json(pid)
            t0 = time.perf_counter()
            reasoning_text = generator(problem)
            elapsed = time.perf_counter() - t0
            category_stats[stat_key].runtimes.append(elapsed)

            if reasoning_text is None:
                if is_active:
                    existing[pid]["status"] = "rule_unknown"
                    existing[pid]["submission"] = ""
                skipped += 1
                continue

            submission = extract_answer(reasoning_text)
            result = compare_answer(problem.answer, submission)
            if is_active:
                stats[pid] = result
                existing[pid]["status"] = "rule_found" if result else "rule_unknown"
                existing[pid]["submission"] = submission
            if result:
                category_stats[stat_key].rule_found += 1

            with open(out_dir / f"{pid}.txt", "w") as f:
                f.write(reasoning_text)

            if is_active and cat in INVESTIGATION_CATEGORIES:
                inv_path = INVESTIGATIONS_DIR / f"{pid}.txt"
                if result and args.delete_investigations and inv_path.exists():
                    inv_path.unlink()

            generated += 1

    # Categories with no registered solver -> mark unknown in the active index.
    for entry in existing.values():
        if entry["category"] not in REGISTRY and entry["category"] not in SKIP_CATEGORIES:
            entry["status"] = "rule_unknown"
            entry["submission"] = ""


    # Update status for problems with investigation files (only if not already rule_found)
    hypothesis_formed = 0
    for inv_path in INVESTIGATIONS_DIR.glob("*.txt"):
        pid = inv_path.stem
        if pid not in existing:
            continue
        if existing[pid]["status"] == "rule_found":
            continue
        existing[pid]["status"] = "hypothesis_formed"
        hypothesis_formed += 1

    # Write merged results back to problems.jsonl
    with PROBLEMS_INDEX.open("w") as f:
        for entry in existing.values():
            entry.pop("has_investigation", None)
            f.write(json.dumps(entry) + "\n")

    # Print accuracy stats
    total = sum(c.total for c in category_stats.values())
    rule_found = sum(c.rule_found for c in category_stats.values())
    print(f"\nGenerated {generated} reasoning files in {REASONING_DIR}/")
    if skipped:
        print(f"Skipped {skipped} (no generator for category)")
    if hypothesis_formed:
        print(
            f"Hypothesis formed: {hypothesis_formed} (investigation without reasoning)"
        )
    w = 70
    print(f"\n{'=' * w}")
    print(
        f"{'Category/Version':<34} {'Found':>6} {'Total':>6} {'Accuracy':>10} {'Avg ms':>10}"
    )
    print(f"{'-' * w}")
    all_runtimes: list[float] = []
    for category_name, counts in sorted(category_stats.items()):
        acc = counts.rule_found / counts.total * 100 if counts.total else 0
        avg_ms = (
            sum(counts.runtimes) / len(counts.runtimes) * 1000 if counts.runtimes else 0
        )
        all_runtimes.extend(counts.runtimes)
        acc_str = f"{acc:.1f}%"
        print(
            f"{category_name:<34} {counts.rule_found:>6} {counts.total:>6} {acc_str:>10} {avg_ms:>10.1f}"
        )
    print(f"{'-' * w}")
    overall_acc = rule_found / total * 100 if total else 0
    overall_avg_ms = sum(all_runtimes) / len(all_runtimes) * 1000 if all_runtimes else 0
    overall_acc_str = f"{overall_acc:.1f}%"
    print(
        f"{'TOTAL':<34} {rule_found:>6} {total:>6} {overall_acc_str:>10} {overall_avg_ms:>10.1f}"
    )
    print(f"{'=' * w}")
    print("\nIf you were given an example to fix, please verify that example.")
    print(
        "\nIf the user has previously asked to run corpus.py, you should run `uv run corpus.py`"
    )


if __name__ == "__main__":
    main()
