"""Cipher-digit (cryptarithm) reasoning generator -- v3.

Clean-room reimplementation inspired by the *approach* of the public
``equation_symbolic`` solver (lkevincc0/kaggle-nemotron-equation-symbolic): model
each puzzle as an injective symbol<->digit cipher over a symbol-arithmetic rule,
recover the cipher by CONSTRAINT PROPAGATION (forward checking), and narrate the
deduction so every committed digit is either *forced* (the only value consistent
with the shown equations) or an explicitly *labelled guess*. No source code from
that repository is used -- only the general idea (a cipher CSP solved by
forward-checking with a forced/guess deduction trace), which is not protectable.

Design choices that keep every emitted trace answer-independent / learnable:
  * Standard digit order (most-significant first), base 10 only. Puzzles that
    would need reversed-digit / little-endian modes are left to the fallback,
    rather than emitting a cipher the reader cannot derive.
  * The cipher is recovered purely from the worked examples (no answer hint), so
    the propagation a learner sees is exactly what determines the digits.
  * Each digit is committed as ``forced`` (unique remaining candidate) or, when a
    genuine choice remains, an explicit ``guess`` -- never hidden.

Falls back to the v1 concatenation solver when no consistent arithmetic cipher is
found (e.g. pure-concatenation puzzles, reversed-digit modes, or unseen query
operators).
"""

from __future__ import annotations

import itertools
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

from reasoners.cryptarithm import reasoning_cryptarithm as _reasoning_original
from reasoners.store_types import Problem

# ───────────────────────── operation library ─────────────────────────
# Each entry maps two 2-digit operands to a signed integer result, or ``None``
# when the operation is undefined for those operands. Sign handling (whether the
# output carries a leading operator-symbol prefix) is applied by ``_eval``.


def _f_sub(a: int, b: int) -> Optional[int]:
    return a - b if a >= b else None


def _f_rsub(a: int, b: int) -> Optional[int]:
    return b - a if b >= a else None


def _f_fdiv(a: int, b: int) -> Optional[int]:
    return a // b if b else None


def _f_rdiv(a: int, b: int) -> Optional[int]:
    return b // a if a else None


def _f_mod(a: int, b: int) -> Optional[int]:
    return a % b if b else None


def _f_rmod(a: int, b: int) -> Optional[int]:
    return b % a if a else None


def _f_lcm(a: int, b: int) -> Optional[int]:
    return a * b // gcd(a, b) if (a and b) else 0


def _off(base, delta):
    def g(a: int, b: int) -> Optional[int]:
        v = base(a, b)
        if v is None:
            return None
        r = v + delta
        return r if r >= 0 else None

    return g


# name -> callable. ``sub_signed`` / ``rsub_signed`` may return negative values
# (sign-prefixed); ``neg_absdiff`` is always <= 0. All others must be >= 0.
_OPS = {
    "add": lambda a, b: a + b,
    "sub": _f_sub,
    "rsub": _f_rsub,
    "sub_signed": lambda a, b: a - b,
    "rsub_signed": lambda a, b: b - a,
    "absdiff": lambda a, b: abs(a - b),
    "neg_absdiff": lambda a, b: -abs(a - b),
    "mul": lambda a, b: a * b,
    "gcd": gcd,
    "lcm": _f_lcm,
    "fdiv": _f_fdiv,
    "rdiv": _f_rdiv,
    "mod": _f_mod,
    "rmod": _f_rmod,
    "min": min,
    "max": max,
    "add_p1": _off(lambda a, b: a + b, 1),
    "add_m1": _off(lambda a, b: a + b, -1),
    "mul_p1": _off(lambda a, b: a * b, 1),
    "mul_m1": _off(lambda a, b: a * b, -1),
    "absdiff_p1": _off(lambda a, b: abs(a - b), 1),
    "absdiff_m1": _off(lambda a, b: abs(a - b), -1),
}

_SIGNED = {"sub_signed", "rsub_signed"}

# Human-readable arithmetic for one operation, e.g. "34 * 39".
_EXPR = {
    "add": "{a} + {b}",
    "sub": "{a} - {b}",
    "rsub": "{b} - {a}",
    "sub_signed": "{a} - {b}",
    "rsub_signed": "{b} - {a}",
    "absdiff": "|{a} - {b}|",
    "neg_absdiff": "-|{a} - {b}|",
    "mul": "{a} * {b}",
    "gcd": "gcd({a}, {b})",
    "lcm": "lcm({a}, {b})",
    "fdiv": "{a} / {b}",
    "rdiv": "{b} / {a}",
    "mod": "{a} mod {b}",
    "rmod": "{b} mod {a}",
    "min": "min({a}, {b})",
    "max": "max({a}, {b})",
    "add_p1": "{a} + {b} + 1",
    "add_m1": "{a} + {b} - 1",
    "mul_p1": "{a} * {b} + 1",
    "mul_m1": "{a} * {b} - 1",
    "absdiff_p1": "|{a} - {b}| + 1",
    "absdiff_m1": "|{a} - {b}| - 1",
}

# Try-order (simplest / most common first) so the search commits to the
# canonical operation when several fit the examples.
_PRIORITY = [
    "add",
    "sub",
    "rsub",
    "absdiff",
    "mul",
    "sub_signed",
    "rsub_signed",
    "neg_absdiff",
    "gcd",
    "lcm",
    "min",
    "max",
    "fdiv",
    "rdiv",
    "mod",
    "rmod",
    "add_p1",
    "add_m1",
    "mul_p1",
    "mul_m1",
    "absdiff_p1",
    "absdiff_m1",
]


def _eval(op_name: str, a: int, b: int, has_sign: bool) -> Optional[int]:
    """Return the non-negative MAGNITUDE the result symbols must encode, honoring
    the sign convention, or ``None`` if the op is inconsistent with ``has_sign``.
    """
    fn = _OPS[op_name]
    v = fn(a, b)
    if v is None:
        return None
    if op_name in _SIGNED:
        if (v < 0) != has_sign:
            return None
        return -v if v < 0 else v
    if op_name == "neg_absdiff":
        # -|a-b| is <= 0; a sign prefix is required unless the value is 0.
        if v < 0:
            return -v if has_sign else None
        return 0 if not has_sign else None
    # plain non-negative op: never sign-prefixed.
    if has_sign or v < 0:
        return None
    return v


# ───────────────────────── parsing ─────────────────────────

# fact = (s0, s1, opsym, s3, s4, has_sign, result_symbols)
_Fact = Tuple[str, str, str, str, str, bool, Tuple[str, ...]]


def _parse(problem: Problem) -> Optional[Tuple[List[_Fact], str, str, str]]:
    facts: List[_Fact] = []
    for ex in problem.examples:
        iv = str(ex.input_value)
        ov = str(ex.output_value)
        if len(iv) != 5 or not ov:
            return None
        opsym = iv[2]
        has_sign = len(ov) > 1 and ov[0] == opsym
        res = ov[1:] if has_sign else ov
        if not res:
            return None
        facts.append((iv[0], iv[1], opsym, iv[3], iv[4], has_sign, tuple(res)))
    q = str(problem.question)
    if len(q) != 5 or not facts:
        return None
    return facts, q, q[2], str(problem.answer)


def _content_symbols(facts: Sequence[_Fact], q: str) -> List[str]:
    opchars = {f[2] for f in facts} | {q[2]}
    syms = set()
    for s0, s1, _op, s3, s4, _hs, res in facts:
        syms.update([s0, s1, s3, s4])
        syms.update(res)
    syms.update([q[0], q[1], q[3], q[4]])
    return sorted(syms - opchars)


# ───────────────────────── op narrowing ─────────────────────────


def _can_make_len(op_name: str, target: int) -> bool:
    """Whether ``op_name`` can produce a result of ``target`` decimal digits from
    two 2-digit operands (0..99 each)."""
    if op_name in ("add", "add_p1", "add_m1", "sub_signed", "rsub_signed"):
        return target <= 3  # a+b <= 198
    if op_name in ("mul", "mul_p1", "mul_m1", "lcm"):
        return target <= 4  # up to 9801
    # |diff|, gcd, div, mod, min, max, neg_absdiff, offsets: <= 99
    return target <= 2


def _narrow(facts_for_op: Sequence[_Fact]) -> List[str]:
    """Per-operator candidate ops from structural features only (sign pattern and
    result lengths) -- no digit search yet."""
    signs = {f[5] for f in facts_for_op}
    lens = {len(f[6]) for f in facts_for_op}
    max_len = max(lens)
    if signs == {True}:
        pool = [
            t for t in _PRIORITY if t in ("neg_absdiff", "sub_signed", "rsub_signed")
        ]
    elif signs == {False}:
        pool = [t for t in _PRIORITY if t != "neg_absdiff"]
    else:  # mixed: only signed subtractions explain both branches
        pool = [t for t in _PRIORITY if t in ("sub_signed", "rsub_signed")]
    return [t for t in pool if _can_make_len(t, max_len)]


# ───────────────────────── forward-checking CSP ─────────────────────────

_Eq = Tuple[_Fact, str]  # (fact, op_name)


def _feasible(
    fact: _Fact, op_name: str, domains: Dict[str, set]
) -> Optional[Dict[str, set]]:
    """For one equation under ``op_name``, return the digits each involved symbol
    can take in SOME assignment consistent with current ``domains`` (forward
    checking). ``None`` if the equation has no consistent assignment at all."""
    s0, s1, _op, s3, s4, has_sign, res = fact
    operands = [s0, s1, s3, s4]
    distinct = list(dict.fromkeys(operands))
    involved = set(operands) | set(res)
    feas: Dict[str, set] = {s: set() for s in involved}
    found = False

    def rec(i: int, assign: Dict[str, int], used: set) -> None:
        nonlocal found
        if i == len(distinct):
            a = assign[s0] * 10 + assign[s1]
            b = assign[s3] * 10 + assign[s4]
            mag = _eval(op_name, a, b, has_sign)
            if mag is None:
                return
            digs = str(mag)
            if len(digs) != len(res):
                return
            rmap: Dict[str, int] = {}
            usedr = set(used)
            for sym, ch in zip(res, digs):
                d = int(ch)
                if sym in assign:
                    if assign[sym] != d:
                        return
                elif sym in rmap:
                    if rmap[sym] != d:
                        return
                else:
                    if d not in domains.get(sym, set()) or d in usedr:
                        return
                    rmap[sym] = d
                    usedr.add(d)
            found = True
            for sym, d in assign.items():
                feas[sym].add(d)
            for sym, d in rmap.items():
                feas[sym].add(d)
            return
        sym = distinct[i]
        for d in domains.get(sym, set()):
            if d in used:
                continue
            assign[sym] = d
            used.add(d)
            rec(i + 1, assign, used)
            used.discard(d)
            del assign[sym]

    rec(0, {}, set())
    return feas if found else None


def _propagate(domains: Dict[str, set], eqs: Sequence[_Eq]) -> bool:
    """Forward-check + AllDifferent to a fixed point. Mutates ``domains``.
    Returns False on contradiction (some domain emptied)."""
    changed = True
    while changed:
        changed = False
        singles = {s: next(iter(d)) for s, d in domains.items() if len(d) == 1}
        used = set(singles.values())
        for s, d in domains.items():
            if len(d) > 1:
                nd = d - used
                if nd != d:
                    domains[s] = nd
                    changed = True
        for fact, op_name in eqs:
            feas = _feasible(fact, op_name, domains)
            if feas is None:
                return False
            for s, fs in feas.items():
                nd = domains[s] & fs
                if nd != domains[s]:
                    domains[s] = nd
                    changed = True
        if any(not d for d in domains.values()):
            return False
    return True


def _search(symbols: Sequence[str], eqs: Sequence[_Eq]) -> Optional[Dict[str, int]]:
    """Find one symbol->digit assignment (injective) consistent with every
    equation, via forward-checking + MRV backtracking."""

    def rec(domains: Dict[str, set]) -> Optional[Dict[str, int]]:
        if not _propagate(domains, eqs):
            return None
        unknown = [s for s in symbols if len(domains[s]) > 1]
        if not unknown:
            return {s: next(iter(domains[s])) for s in symbols}
        pivot = min(unknown, key=lambda x: (len(domains[x]), x))
        for d in sorted(domains[pivot]):
            child = {k: set(v) for k, v in domains.items()}
            child[pivot] = {d}
            got = rec(child)
            if got is not None:
                return got
        return None

    init = {s: set(range(10)) for s in symbols}
    return rec(init)


# ───────────────────────── solution ─────────────────────────


class _Solution:
    __slots__ = ("mapping", "op_of", "qop", "qop_name", "answer", "facts", "q")

    def __init__(self, mapping, op_of, qop, qop_name, answer, facts, q):
        self.mapping = mapping
        self.op_of = op_of
        self.qop = qop
        self.qop_name = qop_name
        self.answer = answer
        self.facts = facts
        self.q = q


def _encode_query(mapping: Dict[str, int], q: str, qop_name: str) -> Optional[str]:
    a = mapping[q[0]] * 10 + mapping[q[1]]
    b = mapping[q[3]] * 10 + mapping[q[4]]
    fn = _OPS[qop_name]
    v = fn(a, b)
    if v is None:
        return None
    if qop_name in _SIGNED:
        neg = v < 0
        mag = -v if neg else v
    elif qop_name == "neg_absdiff":
        if v == 0:
            neg, mag = False, 0
        else:
            neg, mag = True, -v
    else:
        if v < 0:
            return None
        neg, mag = False, v
    inv = {d: s for s, d in mapping.items()}
    digs = str(mag)
    if any(int(c) not in inv for c in digs):
        return None  # query needs a digit no symbol encodes -> underdetermined
    return (q[2] if neg else "") + "".join(inv[int(c)] for c in digs)


def _ordered_combos(cand: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """All per-operator op assignments, simplest-first by summed priority rank."""
    opsyms = sorted(cand)
    pools = [cand[op] for op in opsyms]
    combos = []
    for choice in itertools.product(*pools):
        combo = dict(zip(opsyms, choice))
        rank = sum(_PRIORITY.index(o) for o in choice)
        combos.append((rank, combo))
    combos.sort(key=lambda t: (t[0], sorted(t[1].items())))
    return [c for _, c in combos]


def _solve(problem: Problem) -> Optional[_Solution]:
    parsed = _parse(problem)
    if parsed is None:
        return None
    facts, q, qop, _answer = parsed
    symbols = _content_symbols(facts, q)
    if not symbols or len(symbols) > 10:
        return None
    if qop not in {f[2] for f in facts}:
        return None  # unseen query operator -> defer to fallback

    by_op: Dict[str, List[_Fact]] = {}
    for f in facts:
        by_op.setdefault(f[2], []).append(f)
    cand = {op: _narrow(fs) for op, fs in by_op.items()}
    if any(not c for c in cand.values()):
        return None

    # Cap the combinatorial outer loop defensively.
    total = 1
    for c in cand.values():
        total *= len(c)
    if total > 4000:
        return None

    for combo in _ordered_combos(cand):
        eqs = [(f, combo[f[2]]) for f in facts]
        mapping = _search(symbols, eqs)
        if mapping is None:
            continue
        ans = _encode_query(mapping, q, combo[qop])
        if ans is None:
            continue
        return _Solution(mapping, combo, qop, combo[qop], ans, facts, q)
    return None


# ───────────────────────── deduction trace ─────────────────────────


def _derive_steps(
    symbols: Sequence[str], eqs: Sequence[_Eq], mapping: Dict[str, int]
) -> Tuple[List[str], List[Tuple[str, int, str, List[int]]]]:
    """Re-run forward checking, committing forced symbols first and falling back
    to a labelled guess (using ``mapping``) only when a real choice remains.
    Returns (initial per-symbol candidate lines, ordered commit steps)."""
    domains = {s: set(range(10)) for s in symbols}
    _propagate(domains, eqs)
    initial = [
        f"  {s}: {{{', '.join(str(d) for d in sorted(domains[s]))}}}" for s in symbols
    ]
    steps: List[Tuple[str, int, str, List[int]]] = []
    determined: Dict[str, int] = {}
    guard = 0
    while len(determined) < len(symbols) and guard < 100:
        guard += 1
        _propagate(domains, eqs)
        forced = sorted(
            s for s in symbols if s not in determined and len(domains[s]) == 1
        )
        if forced:
            for s in forced:
                d = next(iter(domains[s]))
                steps.append((s, d, "forced", [d]))
                determined[s] = d
            continue
        rem = [s for s in symbols if s not in determined]
        if not rem:
            break
        pivot = min(rem, key=lambda x: (len(domains[x]), x))
        choices = sorted(domains[pivot])
        d = mapping[pivot]
        steps.append((pivot, d, "guess", choices))
        domains[pivot] = {d}
        determined[pivot] = d
    return initial, steps


def _op_expr(op_name: str, a: int, b: int) -> str:
    return _EXPR[op_name].format(a=a, b=b)


# ───────────────────────── concatenation reading ─────────────────────────


def _op_concat_dir(facts_for_op: Sequence[_Fact]) -> Optional[str]:
    """'fwd' (output = left||right) / 'rev' (right||left) / None: whether EVERY
    example of one operator reads as plain concatenation. Prompt-computable."""
    fwd = rev = True
    for s0, s1, _op, s3, s4, has_sign, res in facts_for_op:
        if has_sign or len(res) != 4:
            return None
        out = "".join(res)
        if out != s0 + s1 + s3 + s4:
            fwd = False
        if out != s3 + s4 + s0 + s1:
            rev = False
    if fwd:
        return "fwd"
    if rev:
        return "rev"
    return None


def _concat_apply(q: str, direction: str) -> str:
    """Concatenate the query operands in ``direction`` ('fwd'|'rev')."""
    if direction == "rev":
        return q[3] + q[4] + q[0] + q[1]
    return q[0] + q[1] + q[3] + q[4]


def _emit_header(lines: List[str], problem: Problem) -> None:
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append(
        "Each input is two 2-digit groups of symbols around an operator symbol. "
        "Either the output simply CONCATENATES the operands' symbols, or the "
        "symbols encode digits (one injective symbol<->digit cipher shared by "
        "every line) and each operator denotes one fixed arithmetic operation "
        "(a negative result keeps the operator symbol as a sign prefix). Decide "
        "which by testing the examples."
    )
    lines.append("")
    lines.append("Examples:")
    for ex in problem.examples:
        lines.append(f"  {ex.input_value} = {ex.output_value}")
    lines.append("")


def _emit_concat_check(
    lines: List[str], facts: Sequence[_Fact], q: str, qop: str
) -> Optional[str]:
    """Step 1: show, per operator, whether its examples are plain concatenation.
    Returns the query operator's concat direction ('fwd'/'rev') or None. This is
    the answer-independent gate that routes concat vs cipher."""
    by_op: Dict[str, List[_Fact]] = {}
    for f in facts:
        by_op.setdefault(f[2], []).append(f)
    lines.append(
        "Step 1 - test the simplest reading, plain concatenation (output = the "
        "two operands' symbols in order, forward or reversed):"
    )
    for op in sorted(by_op):
        direction = _op_concat_dir(by_op[op])
        s0, s1, _o, s3, s4, _hs, res = by_op[op][0]
        out = "".join(res) if (not _hs and len(res) == 4) else ("?" + "".join(res))
        if direction == "fwd":
            verdict = f"= {s0}{s1}{s3}{s4} forward-concatenation (holds for every '{op}' example)"
        elif direction == "rev":
            verdict = f"= {s3}{s4}{s0}{s1} reversed-concatenation (holds for every '{op}' example)"
        else:
            verdict = "is NOT plain concatenation (some example differs)"
        lines.append(f"  operator '{op}': {s0}{s1}{op}{s3}{s4} -> {out}; {verdict}")
    lines.append("")
    return _op_concat_dir(by_op[qop]) if qop in by_op else None


def _emit_concat_answer(
    lines: List[str], q: str, qop: str, direction: str, conceded: bool
) -> str:
    label = "forward" if direction == "fwd" else "reversed"
    if conceded:
        lines.append(
            "No operator reads cleanly as concatenation and no consistent digit "
            f"cipher fits the examples. Best effort: read the query operator '{qop}' "
            f"as {label} concatenation."
        )
    else:
        lines.append(
            f"The query operator '{qop}' is {label} concatenation; apply it to the "
            "query."
        )
    ans = _concat_apply(q, direction)
    lines.append(f"  {q[0]}{q[1]}{qop}{q[3]}{q[4]} -> {ans}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{-}} is \\boxed{{{ans}}}")
    return ans


def _emit_cipher_section(lines: List[str], sol: _Solution) -> None:
    """Step 2+3: recover the cipher by constraint propagation, VERIFY it against
    every example (the learnable gate), then apply it to the query."""
    facts, q = sol.facts, sol.q
    mapping, op_of = sol.mapping, sol.op_of
    inv = {d: s for s, d in mapping.items()}
    symbols = sorted(mapping)
    eqs = [(f, op_of[f[2]]) for f in facts]
    by_op: Dict[str, List[_Fact]] = {}
    for f in facts:
        by_op.setdefault(f[2], []).append(f)

    lines.append(
        "The query operator is not plain concatenation, so the symbols must stand "
        "for digits. Step 2 - recover the symbol<->digit cipher."
    )
    lines.append(
        "Narrow each operator to the operations whose result sign and length fit "
        "every example:"
    )
    for op in sorted(by_op):
        lines.append(
            f"  '{op}' -> {{{', '.join(_narrow(by_op[op]))}}}  (take {op_of[op]})"
        )
    lines.append("")
    lines.append(
        "Solve the cipher by constraint propagation. Start every symbol at "
        "{0..9}, then for each example keep only digits that let the operation "
        "reproduce the output; AllDifferent removes a fixed digit from the rest."
    )
    initial, steps = _derive_steps(symbols, eqs, mapping)
    lines.append("Candidate digits after one pass over the examples:")
    lines.extend(initial)
    lines.append("")
    lines.append("Commit symbols in order (forced = only digit left):")
    for s, d, kind, choices in steps:
        if kind == "forced":
            lines.append(f"  {s} = {d}  forced (only remaining digit)")
        else:
            opts = ", ".join(str(c) for c in choices)
            lines.append(
                f"  {s} = {d}  guess (choices {{{opts}}}; the examples do not "
                f"single one out, take {d} and verify)"
            )
    lines.append("")
    lines.append("Recovered cipher: " + ", ".join(f"{s}={mapping[s]}" for s in symbols))
    lines.append(
        "Operator meanings: " + ", ".join(f"'{op}'={op_of[op]}" for op in sorted(by_op))
    )
    lines.append("")

    lines.append("Step 3 - verify every example under this cipher:")
    for s0, s1, opsym, s3, s4, has_sign, res in facts:
        a = mapping[s0] * 10 + mapping[s1]
        b = mapping[s3] * 10 + mapping[s4]
        op = op_of[opsym]
        mag = _eval(op, a, b, has_sign)
        ds = str(mag) if mag is not None else "?"
        enc = (opsym if has_sign else "") + "".join(inv.get(int(c), "?") for c in ds)
        exp = (opsym if has_sign else "") + "".join(res)
        ok = "OK" if enc == exp else "MISMATCH"
        lines.append(
            f"  {s0}{s1}{opsym}{s3}{s4}: {_op_expr(op, a, b)} = "
            f"{'-' if has_sign else ''}{ds} -> {enc} (expected {exp}) {ok}"
        )
    lines.append("All examples reproduced; the cipher is valid. Apply it to the query.")
    lines.append("")

    qa = mapping[q[0]] * 10 + mapping[q[1]]
    qb = mapping[q[3]] * 10 + mapping[q[4]]
    lines.append(f"Apply to the question {q[0]}{q[1]}{q[2]}{q[3]}{q[4]}:")
    lines.append(
        f"  {q[0]}{q[1]}={qa}, {q[3]}{q[4]}={qb}, '{q[2]}'={sol.qop_name}: "
        f"{_op_expr(sol.qop_name, qa, qb)} -> encode back to symbols -> {sol.answer}."
    )
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{-}} is \\boxed{{{sol.answer}}}")


def reasoning_cryptarithm(problem: Problem) -> Optional[str]:
    """Unified verify-gated trace. Every trace shows the same answer-independent
    decision sequence so the routing is learnable:
      1. test plain concatenation per operator (prompt-computable);
      2. if the query operator concatenates -> apply it;
      3. else recover a symbol<->digit cipher and VERIFY it reproduces every
         example (the gate), then apply it;
      4. if neither fits -> concede the concatenation reading as best effort.
    """
    parsed = _parse(problem)
    if parsed is None:
        return _reasoning_original(problem)
    facts, q, qop, _answer = parsed

    try:
        sol = _solve(problem)
        lines: List[str] = []
        _emit_header(lines, problem)
        qdir = _emit_concat_check(lines, facts, q, qop)

        if sol is not None:
            # Cipher reproduces every example -> the verify gate passes.
            _emit_cipher_section(lines, sol)
            return "\n".join(lines)

        # No arithmetic cipher. Use concatenation: the query operator's own
        # direction if it concatenates, otherwise concede to forward concat
        # (matching the original solver's default).
        conceded = qdir is None
        _emit_concat_answer(lines, q, qop, qdir or "fwd", conceded)
        return "\n".join(lines)
    except Exception:
        return _reasoning_original(problem)
