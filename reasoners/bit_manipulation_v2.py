"""Improved reasoning generator for 8-bit bit-manipulation tasks.

This is a modified copy of ``reasoners.bit_manipulation``. The original solver
models every output bit as a constant / unary / 2-operand pair op whose operands
follow a rigid stride-+1 linear layout. Empirically that leaves ~15% of the
problems unsolved, because the real generative rule space (stated in the prompt:
"bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice
functions") produces transforms where each output bit depends on the input bits
at *fixed relative offsets* (rotations/shifts of the input), combined by a single
shared boolean function -- a translation-invariant / cellular-automaton style
rule.

This module adds that translation-invariant hypothesis as the primary solver:

    output[i] = g( x[(i + d) % 8]  for d in D )            # rotation semantics
    output[i] = g( x[ i + d] or 0  for d in D )            # shift  semantics

where ``g`` is one fixed boolean function and ``D`` a small set of offsets,
shared across every bit position and every example. Because ``g`` is constrained
by ``8 * n_examples`` cells instead of a single column, it generalizes to the
held-out question far better than per-bit fitting.

When no unambiguous translation-invariant rule is found we fall back to the
original stride-based solver, so this is a strict superset in coverage.

Measured on the 1602 ``bit_manipulation`` problems in ``problems.jsonl``:
original solver 1364/1602 (85.1%) -> this solver 1569/1602 (97.9%).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from reasoners.bit_manipulation import (
    reasoning_bit_manipulation as _reasoning_bit_manipulation_original,
)
from reasoners.store_types import Problem

N_BITS = 8
# Relative offsets to search: -7..7 covers every rotation/shift direction.
_OFFSETS: Tuple[int, ...] = tuple(range(-(N_BITS - 1), N_BITS))
# Maximum number of input bits an output bit may depend on.
_MAX_SUPPORT = 4


def _normalize_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    if len(bits) != N_BITS:
        return ""
    return bits


def _source_bit(bits: str, position: int, offset: int, mode: str) -> str:
    """Return the input bit feeding output ``position`` for relative ``offset``.

    ``mode == "rotate"`` wraps around (mod 8); ``mode == "shift"`` fills with 0.
    """
    j = position + offset
    if mode == "rotate":
        return bits[j % N_BITS]
    return bits[j] if 0 <= j < N_BITS else "0"


def _signed(offset: int) -> str:
    return f"+{offset}" if offset >= 0 else str(offset)


class _Rule:
    """A translation-invariant rule: shared function ``table`` over offsets ``D``."""

    __slots__ = ("mode", "offsets", "table")

    def __init__(
        self, mode: str, offsets: Tuple[int, ...], table: Dict[Tuple[str, ...], str]
    ) -> None:
        self.mode = mode
        self.offsets = offsets
        self.table = table

    def predict(self, bits: str) -> Optional[str]:
        out: List[str] = []
        for i in range(N_BITS):
            key = tuple(_source_bit(bits, i, d, self.mode) for d in self.offsets)
            if key not in self.table:
                return None
            out.append(self.table[key])
        return "".join(out)


def _build_table(
    inputs: Sequence[str], outputs: Sequence[str], offsets: Tuple[int, ...], mode: str
) -> Optional[Dict[Tuple[str, ...], str]]:
    """Build the shared truth table, or ``None`` if outputs are inconsistent."""
    table: Dict[Tuple[str, ...], str] = {}
    for x, o in zip(inputs, outputs):
        for i in range(N_BITS):
            key = tuple(_source_bit(x, i, d, mode) for d in offsets)
            if key in table:
                if table[key] != o[i]:
                    return None
            else:
                table[key] = o[i]
    return table


def _search_rules(
    inputs: Sequence[str], outputs: Sequence[str], question: str
) -> Tuple[Optional[_Rule], List[str]]:
    """Find the simplest unambiguous translation-invariant rule.

    Returns ``(rule, answers)`` where ``answers`` is the set (as a sorted list) of
    distinct question predictions produced by *all* consistent, fully-determining
    rules. The chosen ``rule`` is the simplest one (fewest offsets, then offsets
    closest to zero). When ``answers`` has more than one element the rule is
    ambiguous and the caller should fall back.
    """
    best: Optional[_Rule] = None
    best_key: Optional[Tuple[int, int, Tuple[int, ...]]] = None
    answers: set[str] = set()
    for mode in ("rotate", "shift"):
        for k in range(1, _MAX_SUPPORT + 1):
            for offsets in itertools.combinations(_OFFSETS, k):
                table = _build_table(inputs, outputs, offsets, mode)
                if table is None:
                    continue
                rule = _Rule(mode, offsets, table)
                pred = rule.predict(question)
                if pred is None:
                    continue
                answers.add(pred)
                rank = (k, sum(abs(d) for d in offsets), offsets)
                if best_key is None or rank < best_key:
                    best_key = rank
                    best = rule
    return best, sorted(answers)


# Names for recognizable shared boolean functions, keyed by support size.
def _name_function(rule: _Rule) -> Optional[str]:
    table = rule.table
    k = len(rule.offsets)
    keys = [tuple(format(v, "b").zfill(k)) for v in range(2**k)]

    def ev(fn) -> bool:
        return all(
            key in table and table[key] == fn(tuple(int(c) for c in key))
            for key in keys
        )

    if k == 1:
        if ev(lambda v: str(v[0])):
            return "copy"
        if ev(lambda v: str(1 - v[0])):
            return "NOT"
    if k == 2:
        a, b = 0, 1
        named = {
            "AND": lambda v: str(v[a] & v[b]),
            "OR": lambda v: str(v[a] | v[b]),
            "XOR": lambda v: str(v[a] ^ v[b]),
            "NAND": lambda v: str(1 - (v[a] & v[b])),
            "NOR": lambda v: str(1 - (v[a] | v[b])),
            "XNOR": lambda v: str(1 - (v[a] ^ v[b])),
        }
        for name, fn in named.items():
            if ev(fn):
                return name
    if k == 3:
        named3 = {
            "majority": lambda v: str(1 if sum(v) >= 2 else 0),
            "minority": lambda v: str(1 if sum(v) <= 1 else 0),
            "XOR3": lambda v: str(v[0] ^ v[1] ^ v[2]),
            "choice": lambda v: str(v[1] if v[0] else v[2]),
        }
        for name, fn in named3.items():
            if ev(fn):
                return name
    return None


def _emit_table(lines: List[str], rule: _Rule) -> None:
    k = len(rule.offsets)
    lines.append("g (" + "".join(f"s{j}" for j in range(k)) + " -> out)")
    for v in range(2**k):
        key = tuple(format(v, "b").zfill(k))
        if key in rule.table:
            lines.append("".join(key) + f" {rule.table[key]}")


# Human labels for each support size, used to narrate the search ladder.
_LADDER_LABELS = {
    1: "a single source bit (identity, NOT, a pure rotation, or a shift)",
    2: "a 2-bit window (e.g. XOR / AND / OR of two positions)",
    3: "a 3-bit window (e.g. majority / choice / 3-way XOR)",
    4: "a 4-bit window",
}

# A conflict witness: (offsets, mode, source-pattern, (ex_a, bit_a, out_a),
# (ex_b, bit_b, out_b)).
_Conflict = Tuple[
    Tuple[int, ...], str, Tuple[str, ...], Tuple[int, int, str], Tuple[int, int, str]
]


def _first_conflict(
    inputs: Sequence[str],
    outputs: Sequence[str],
    offsets: Tuple[int, ...],
    mode: str,
) -> Optional[_Conflict]:
    """Return the first pair of example bits that force the same source pattern to
    two different outputs (so no shared function over ``offsets`` can fit), else
    ``None`` if this window is consistent across every example."""
    seen: Dict[Tuple[str, ...], Tuple[int, int, str]] = {}
    for ei, (x, o) in enumerate(zip(inputs, outputs)):
        for i in range(N_BITS):
            key = tuple(_source_bit(x, i, d, mode) for d in offsets)
            if key in seen:
                if seen[key][2] != o[i]:
                    return offsets, mode, key, seen[key], (ei, i, o[i])
            else:
                seen[key] = (ei, i, o[i])
    return None


# Maximum number of candidate windows shown explicitly per window size. The full
# search tries far more; capping keeps the trace readable while still showing the
# enumerate-test-reject-accept procedure. Raise it for more exhaustive traces.
_SEARCH_DISPLAY_CAP = 12

# Section names emitted by the original per-bit solver (Attempt 2). Each enumerates
# many candidate operand rows; we cap them like the Attempt 1 search to keep the
# fallback trace from blowing past the corpus token limit.
_FALLBACK_SECTIONS = frozenset(
    {"Identity", "NOT", "Constant", "AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"}
)
# Maximum non-matching context rows shown per Attempt 2 section before summarizing.
# Matched rows (those that hit an output column) are ALWAYS kept on top of this.
_FALLBACK_ROW_CAP = 6


def _candidate_verdict(
    inputs: Sequence[str],
    outputs: Sequence[str],
    question: str,
    offsets: Tuple[int, ...],
    mode: str,
) -> Tuple[str, object]:
    """Test one candidate window. Returns ``(verdict, info)`` where verdict is
    ``"inconsistent"`` (info is a conflict witness), ``"undetermined"`` (info is the
    ``(bit, pattern)`` the question needs but the examples never show), or ``"win"``
    (consistent AND fully determines the question; info is ``None``)."""
    conflict = _first_conflict(inputs, outputs, offsets, mode)
    if conflict is not None:
        return "inconsistent", conflict
    table = _build_table(inputs, outputs, offsets, mode)
    assert table is not None  # consistent, since no conflict was found
    for i in range(N_BITS):
        key = tuple(_source_bit(question, i, d, mode) for d in offsets)
        if key not in table:
            return "undetermined", (i, "".join(key))
    return "win", None


def _cand_str(offsets: Tuple[int, ...], mode: str) -> str:
    """Compact candidate label, e.g. ``[-3,+7]s`` (s = shift, r = rotate)."""
    return (
        "["
        + ",".join(_signed(d) for d in offsets)
        + "]"
        + ("r" if mode == "rotate" else "s")
    )


def _window_shape(offsets: Tuple[int, ...], mode: str) -> str:
    """Human name for the *geometry* of a window (independent of the function g)."""
    if len(offsets) == 1:
        d = offsets[0]
        if d == 0:
            return "same position"
        direction = "left" if d > 0 else "right"
        verb = "rotate" if mode == "rotate" else "shift"
        return f"{verb} {direction} {abs(d)}"
    verb = "rotate" if mode == "rotate" else "shift"
    return f"{verb} taps " + ",".join(_signed(d) for d in offsets)


def _emit_search(
    lines: List[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    question: str,
    winner: _Rule,
) -> None:
    """Narrate the search compactly: enumerate offset windows smallest-first, test
    each, reject with concrete evidence, accept the first that works."""
    wk = len(winner.offsets)
    lines.append(
        "Search offset windows simplest first; r rotate (wrap), s shift (zero-fill)."
    )
    lines.append(
        "Accept if consistent (no pattern -> two outputs) and determining "
        "(all question patterns seen)."
    )
    lines.append("x reject inconsistent, ? reject undetermined.")

    for k in range(1, wk + 1):
        combos = sorted(
            itertools.combinations(_OFFSETS, k),
            key=lambda t: (sum(abs(d) for d in t), t),
        )
        rows: List[str] = []
        n_incons = 0
        n_undet = 0
        tested = 0
        accepted: Optional[Tuple[Tuple[int, ...], str]] = None
        for offsets in combos:
            for mode in ("rotate", "shift"):
                tested += 1
                verdict, info = _candidate_verdict(
                    inputs, outputs, question, offsets, mode
                )
                if verdict == "win":
                    accepted = (offsets, mode)
                    break
                if verdict == "inconsistent":
                    n_incons += 1
                    _o, _m, key, (ea, ba, oa), (eb, bb, ob) = info  # type: ignore[misc]
                    ev = f"x {''.join(key)}: {oa}@ex{ea + 1}b{ba} {ob}@ex{eb + 1}b{bb}"
                else:  # undetermined
                    n_undet += 1
                    bit_i, pat = info  # type: ignore[misc]
                    ev = f"? bit{bit_i} needs {pat}"
                if len(rows) < _SEARCH_DISPLAY_CAP:
                    rows.append(f"{_cand_str(offsets, mode)} {ev}")
            if accepted is not None:
                break

        lines.append("")
        lines.append(f"size {k}")
        for row in rows:
            lines.append(row)
        hidden = tested - len(rows) - (1 if accepted is not None else 0)
        if hidden > 0:
            parts = [f"{n_incons} inconsistent"]
            if n_undet:
                parts.append(f"{n_undet} undetermined")
            lines.append(f"+{hidden} more ({', '.join(parts)})")
        if accepted is not None:
            a_off, a_mode = accepted
            gname = (
                _name_function(winner)
                if (a_off == winner.offsets and a_mode == winner.mode)
                else None
            )
            tail = f" g={gname}" if gname else ""
            lines.append(f"{_cand_str(a_off, a_mode)} accept{tail}")
    lines.append("")


def _emit_search_failed(
    lines: List[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    question: str,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Narrate the exhaustive generic-rule search for the FAILURE case (no winner).

    Walks the full ladder 1..``_MAX_SUPPORT`` so the trace shows the search behind
    the give-up decision instead of just asserting it. Returns ``(survivors,
    answers)`` where ``survivors`` is ``(cand, geom, answer)`` for every window that
    was consistent AND determining -- if there are several with different answers,
    that is *why* the generic rule is ambiguous -- and ``answers`` is their sorted
    distinct predictions (empty when nothing survived at all)."""
    lines.append(
        "Search offset windows simplest first; r rotate (wrap), s shift (zero-fill)."
    )
    lines.append(
        "Accept if consistent (no pattern -> two outputs) and determining "
        "(all question patterns seen)."
    )
    lines.append("x reject inconsistent, ? reject undetermined.")

    survivors: List[Tuple[str, str, str]] = []
    answers: set[str] = set()
    for k in range(1, _MAX_SUPPORT + 1):
        combos = sorted(
            itertools.combinations(_OFFSETS, k),
            key=lambda t: (sum(abs(d) for d in t), t),
        )
        rows: List[str] = []
        pass_rows: List[str] = []
        n_incons = 0
        n_undet = 0
        n_pass = 0
        tested = 0
        for offsets in combos:
            for mode in ("rotate", "shift"):
                tested += 1
                verdict, info = _candidate_verdict(
                    inputs, outputs, question, offsets, mode
                )
                if verdict == "win":
                    n_pass += 1
                    table = _build_table(inputs, outputs, offsets, mode)
                    assert table is not None
                    ans = _Rule(mode, offsets, table).predict(question)
                    assert ans is not None
                    answers.add(ans)
                    survivors.append(
                        (_cand_str(offsets, mode), _window_shape(offsets, mode), ans)
                    )
                    if len(pass_rows) < _SEARCH_DISPLAY_CAP:
                        pass_rows.append(f"{_cand_str(offsets, mode)} pass -> {ans}")
                    continue
                if verdict == "inconsistent":
                    n_incons += 1
                    _o, _m, key, (ea, ba, oa), (eb, bb, ob) = info  # type: ignore[misc]
                    ev = f"x {''.join(key)}: {oa}@ex{ea + 1}b{ba} {ob}@ex{eb + 1}b{bb}"
                else:  # undetermined
                    n_undet += 1
                    bit_i, pat = info  # type: ignore[misc]
                    ev = f"? bit{bit_i} needs {pat}"
                if len(rows) < _SEARCH_DISPLAY_CAP:
                    rows.append(f"{_cand_str(offsets, mode)} {ev}")

        lines.append("")
        lines.append(f"size {k}")
        for row in rows:
            lines.append(row)
        hidden = tested - len(rows) - len(pass_rows)
        if hidden > 0:
            parts = [f"{n_incons} inconsistent"]
            if n_undet:
                parts.append(f"{n_undet} undetermined")
            if n_pass:
                parts.append(f"{n_pass} passing")
            lines.append(f"+{hidden} more ({', '.join(parts)})")
        for row in pass_rows:
            lines.append(row)
    lines.append("")
    return survivors, sorted(answers)


_ATTEMPT1_INTRO = (
    "Attempt 1: one generic rule for every bit.\n"
    "g is one shared function of input bits at fixed offsets "
    "(covers shift/rotate/XOR/AND/OR/NOT/majority/choice)."
)


def _emit_header(
    lines: List[str], inputs: Sequence[str], outputs: Sequence[str]
) -> None:
    n = len(outputs)
    lines.append(
        "We need to deduce the transformation by matching the example outputs."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")
    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        col = "".join(o[bit] for o in outputs)
        ones = col.count("1")
        h = "a" if ones in (0, n) else format(ones, "x")
        lines.append(f"{bit} {col} {h}")
    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")


def _emit_ti_body(
    lines: List[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    question: str,
    rule: "_Rule",
    answer: str,
) -> None:
    name = _name_function(rule)
    offsets_str = ", ".join(_signed(d) for d in rule.offsets)
    mode_word = "rotate" if rule.mode == "rotate" else "shift"
    _emit_search(lines, inputs, outputs, question, rule)
    lines.append("Rule")
    lines.append(
        "output[i] = g("
        + ", ".join(
            (
                f"input[(i{_signed(d)}) mod 8]"
                if rule.mode == "rotate"
                else f"input[i{_signed(d)}]"
            )
            for d in rule.offsets
        )
        + ")"
    )
    lines.append(f"{mode_word}, offsets {offsets_str}")
    _emit_table(lines, rule)
    if name is not None:
        lines.append(f"g = {name}")
    lines.append("")

    lines.append("Verify")
    for x, o in zip(inputs, outputs):
        got = rule.predict(x)
        ok = "ok" if got == o else f"MISMATCH expected {o}"
        lines.append(f"{x} -> {got} {ok}")
    lines.append("")

    lines.append(f"Applying to {question}")
    lines.append("Input")
    for i, bit in enumerate(question):
        lines.append(f"{i} {bit}")
    lines.append("Output")
    gname = name or "g"
    for i in range(N_BITS):
        srcs = [_source_bit(question, i, d, rule.mode) for d in rule.offsets]
        val = rule.table[tuple(srcs)]
        lines.append(f"{i} {gname}({','.join(srcs)}) = {val}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")


def _attempt1(
    problem: Problem,
) -> Tuple[
    Optional["_Rule"],
    List[str],
    Optional[List[str]],
    Optional[List[str]],
    Optional[str],
    Optional[str],
]:
    """Run the generic-rule search.

    Returns ``(rule, answers, inputs, outputs, question, reason)``. On success
    ``rule`` is set and ``reason`` is ``None``. On failure ``rule`` is ``None`` and
    ``reason`` is a short human explanation. ``inputs/outputs/question`` are ``None``
    only when the problem could not be normalized at all.
    """
    examples = problem.examples
    if not examples:
        return None, [], None, None, None, "there are no worked examples"

    inputs = [_normalize_bits(ex.input_value) for ex in examples]
    outputs = [_normalize_bits(ex.output_value) for ex in examples]
    question = _normalize_bits(problem.question)
    if (
        not question
        or any(not b for b in inputs + outputs)
        or len(inputs) != len(outputs)
    ):
        return (
            None,
            [],
            None,
            None,
            None,
            "the examples are not all valid 8-bit strings",
        )

    rule, answers = _search_rules(inputs, outputs, question)
    if rule is None:
        return (
            None,
            answers,
            inputs,
            outputs,
            question,
            "no single offset-window function is consistent across all examples",
        )
    if len(answers) != 1:
        return (
            None,
            answers,
            inputs,
            outputs,
            question,
            "several consistent offset-window rules disagree on the question "
            f"(candidates: {', '.join(answers)}), so the generic rule is ambiguous",
        )
    return rule, answers, inputs, outputs, question, None


def _strip_v1_header(trace: str) -> str:
    """Drop the original solver's header and example dumps (already shown above) so
    Attempt 2 begins at the per-bit matching."""
    lns = trace.split("\n")
    for idx, ln in enumerate(lns):
        if ln.startswith("When matching output"):
            return "\n".join(lns[idx:])
    while lns and (
        lns[0].startswith("We need to deduce")
        or lns[0].startswith("I will put my final answer")
        or lns[0] == ""
    ):
        lns.pop(0)
    return "\n".join(lns)


def _trim_fallback(trace: str) -> str:
    """Compact the candidate-row dumps in each Attempt 2 operator section.

    The original solver lists every operand pair it tries (up to ~56 rows for the
    NOT-variant families), which makes fallback traces overflow the corpus token
    limit. We always keep rows that actually matched an output column (the ones the
    decision logic relies on) plus the first ``_FALLBACK_ROW_CAP`` non-matching rows
    for context, and replace the remaining non-matching rows with a one-line summary
    -- mirroring the Attempt 1 search cap. Every per-bit ``Matching output`` /
    ``Left`` / ``Right`` / ``Selected`` decision block and the final answer are left
    untouched."""
    lines = trace.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        if line in _FALLBACK_SECTIONS:
            j = i + 1
            block: List[str] = []
            while j < n and lines[j] != "Matching output":
                block.append(lines[j])
                j += 1
            rows = [r for r in block if r.strip()]
            unmatched = [r for r in rows if " match " not in r]
            if len(unmatched) <= _FALLBACK_ROW_CAP:
                # Nothing worth hiding: keep verbatim, preserving the diff-group
                # blank-line layout.
                out.extend(block)
            else:
                # Keep every matched row; keep the first _FALLBACK_ROW_CAP
                # non-matching rows for context; summarize the rest.
                budget = _FALLBACK_ROW_CAP
                hidden = 0
                for r in rows:
                    if " match " in r:
                        out.append(r)
                    elif budget > 0:
                        out.append(r)
                        budget -= 1
                    else:
                        hidden += 1
                if hidden > 0:
                    out.append(f"+{hidden} more {line} rows (no match)")
                out.append("")
            i = j
            continue
        i += 1
    return "\n".join(out)


def _solve_translation_invariant(problem: Problem) -> Optional[str]:
    """Standalone generic-rule trace (success only); kept for diagnostics."""
    rule, answers, inputs, outputs, question, reason = _attempt1(problem)
    if rule is None or reason is not None or inputs is None or outputs is None:
        return None
    assert question is not None
    lines: List[str] = []
    _emit_header(lines, inputs, outputs)
    lines.append(_ATTEMPT1_INTRO)
    lines.append("")
    _emit_ti_body(lines, inputs, outputs, question, rule, answers[0])
    return "\n".join(lines)


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    """Two-attempt trace: try a generic rule first, fall back to per-bit analysis.

    The trace always narrates Attempt 1 (search for one translation-invariant
    rule). If that succeeds the trace ends there. If it fails, the trace explains
    why and continues with Attempt 2 (the original per-bit solver), so a model
    trained on it learns the routing decision, not just the winning branch.
    """
    rule, answers, inputs, outputs, question, reason = _attempt1(problem)

    # Could not normalize the problem at all: defer entirely to the original.
    if inputs is None or outputs is None or question is None:
        return _reasoning_bit_manipulation_original(problem)

    lines: List[str] = []
    _emit_header(lines, inputs, outputs)
    lines.append(_ATTEMPT1_INTRO)
    lines.append("")

    if rule is not None and reason is None:
        _emit_ti_body(lines, inputs, outputs, question, rule, answers[0])
        return "\n".join(lines)

    # Attempt 1 failed -> show the exhaustive search behind the give-up decision,
    # then explain, then run Attempt 2 (per-bit solver).
    survivors, distinct = _emit_search_failed(lines, inputs, outputs, question)
    if distinct:
        lines.append(
            f"{len(survivors)} windows pass but disagree: {', '.join(distinct)}"
        )
    else:
        lines.append("No window of size 1..4 is consistent and determining.")
    lines.append("")
    lines.append(f"Attempt 1 fails: {reason}.")
    lines.append("Fall back to per-bit matching.")
    lines.append("")
    lines.append(
        "Attempt 2: match each output column against "
        "Identity/NOT/Constant/AND/OR/XOR (and NOT variants), then combine."
    )
    lines.append("")
    fallback = _reasoning_bit_manipulation_original(problem)
    if fallback is None:
        return None
    lines.append(_trim_fallback(_strip_v1_header(fallback)))
    return "\n".join(lines)
