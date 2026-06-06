"""Cipher-digit (cryptarithm_deduce) reasoning generator -- v2.

The original ``reasoners.cryptarithm`` models these puzzles as concatenation, but
empirically only ~8% are. They are CIPHER-DIGIT puzzles: a per-problem injective
symbol<->digit cipher over a SYMBOL-DIGIT arithmetic rule. Each 5-char input
``AB<op>CD`` has the operator symbol at index 2; ``AB`` and ``CD`` are two 2-digit
operands combined by the operator's operation, and the output is the result
written back in the same cipher symbols (negative results prepend the operator
symbol). The cipher is SHARED across every fact and each operator symbol denotes
one fixed operation.

Solving strategy (the "mul anchor", after kemshim's Kaggle writeup): a fact whose
result has 4 digits ("mul") admits only ~6-20 candidate digit assignments. We
crack the cipher starting from the most-constrained fact, propagate the shared
cipher to every other fact (determining each operator + remaining digits), and
apply the query operator. The first consistent assignment, found under a
frequency-prior operation order, is committed. When no consistent cipher+rule is
found we defer to the original concatenation solver.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from reasoners.cryptarithm import reasoning_cryptarithm as _reasoning_original
from reasoners.store_types import Problem

# Operation try-order: most common first (frequency prior).
_OP_PRIOR = (
    "mul",
    "concat",
    "rconcat",
    "add",
    "sub",
    "rsub",
    "mul1",
    "mulm1",
    "add1",
    "addm1",
    "absdiff",
    "negabs",
    "maxmod",
)
_OP_RANK = {op: i for i, op in enumerate(_OP_PRIOR)}

_SEARCH_BUDGET = 1.5  # seconds per problem before deferring to the fallback


def _value(a: int, b: int, op: str) -> Tuple[str, object]:
    """Return ``(kind, value)``: kind ``"s"`` (digit string) for concat ops, else
    ``"i"`` (signed int)."""
    if op == "concat":
        return ("s", f"{a:02d}{b:02d}")
    if op == "rconcat":
        return ("s", f"{b:02d}{a:02d}")
    if op == "add":
        return ("i", a + b)
    if op == "add1":
        return ("i", a + b + 1)
    if op == "addm1":
        return ("i", a + b - 1)
    if op == "sub":
        return ("i", a - b)
    if op == "rsub":
        return ("i", b - a)
    if op == "mul":
        return ("i", a * b)
    if op == "mul1":
        return ("i", a * b + 1)
    if op == "mulm1":
        return ("i", a * b - 1)
    if op == "absdiff":
        return ("i", abs(a - b))
    if op == "negabs":
        return ("i", -abs(a - b))
    if op == "maxmod":
        return ("i", max(a, b) % min(a, b) if min(a, b) else 0)
    raise ValueError(op)


def _out_digits(kind: str, val: object) -> Tuple[bool, str]:
    """The digit string the output symbols must encode, and whether it is negative."""
    if kind == "s":
        return False, str(val)
    n = int(val)  # type: ignore[arg-type]
    return n < 0, str(abs(n))


def _candidate_ops(ov: str) -> Tuple[str, ...]:
    """Operations possible for an output by its (signed) digit length."""
    neg = ov.startswith("-")
    rlen = len(ov.lstrip("-"))
    if neg:
        return ("sub", "rsub", "negabs")
    if rlen == 4:
        return ("concat", "rconcat", "mul", "mul1", "mulm1")
    if rlen == 2:
        return (
            "add",
            "add1",
            "addm1",
            "sub",
            "rsub",
            "absdiff",
            "maxmod",
            "concat",
            "rconcat",
        )
    return _OP_PRIOR


def _op_order(ops: Tuple[str, ...]) -> List[str]:
    return sorted(ops, key=lambda o: _OP_RANK.get(o, 99))


_Fact = Tuple[str, str, str, str, str, str]  # s0,s1,opsym,s3,s4,out


def _parse(problem: Problem) -> Optional[Tuple[List[_Fact], str]]:
    facts: List[_Fact] = []
    for ex in problem.examples:
        iv = str(ex.input_value)
        ov = str(ex.output_value)
        if len(iv) == 5:
            facts.append((iv[0], iv[1], iv[2], iv[3], iv[4], ov))
    q = str(problem.question)
    if len(q) != 5 or not facts:
        return None
    return facts, q


class _Solution:
    __slots__ = ("cipher", "op_of_sym", "answer")

    def __init__(
        self, cipher: Dict[str, int], op_of_sym: Dict[str, str], answer: str
    ) -> None:
        self.cipher = cipher
        self.op_of_sym = op_of_sym
        self.answer = answer


def _solve(problem: Problem) -> Optional[_Solution]:
    parsed = _parse(problem)
    if parsed is None:
        return None
    facts, q = parsed
    dsyms = set()
    for s0, s1, _op, s3, s4, ov in facts:
        dsyms.update([s0, s1, s3, s4])
        dsyms.update(ov.lstrip("-"))
    if len(dsyms) > 10:
        return None

    # Mul-anchor ordering: 4-digit results first, then fewest distinct symbols.
    def fact_key(f: _Fact) -> Tuple[int, int]:
        s0, s1, _op, s3, s4, ov = f
        rlen = len(ov.lstrip("-"))
        nsym = len(set([s0, s1, s3, s4] + list(ov.lstrip("-"))))
        return (0 if rlen == 4 else 1, nsym)

    order = sorted(facts, key=fact_key)
    t0 = time.time()
    cipher: Dict[str, int] = {}
    used = [False] * 10
    op_of: Dict[str, str] = {}
    result: List[Optional[str]] = [None]

    def unify(symbols: List[str], digstr: str) -> Optional[List[Tuple[str, int]]]:
        applied: List[Tuple[str, int]] = []
        local: Dict[str, int] = {}
        for sym, dch in zip(symbols, digstr):
            dv = int(dch)
            if sym in cipher:
                if cipher[sym] != dv:
                    _undo(applied)
                    return None
            elif sym in local:
                if local[sym] != dv:
                    _undo(applied)
                    return None
            elif used[dv]:
                _undo(applied)
                return None
            else:
                local[sym] = dv
                cipher[sym] = dv
                used[dv] = True
                applied.append((sym, dv))
        return applied

    def _undo(applied: List[Tuple[str, int]]) -> None:
        for sym, dv in applied:
            used[dv] = False
            del cipher[sym]

    def solve_fact(fact: _Fact, cont) -> bool:
        s0, s1, opsym, s3, s4, ov = fact
        neg = ov.startswith("-")
        ovd = ov.lstrip("-")
        osyms = [s0, s1, s3, s4]
        unset = [s for s in dict.fromkeys(osyms) if s not in cipher]
        ops = (op_of[opsym],) if opsym in op_of else _op_order(_candidate_ops(ov))

        def place(j: int) -> bool:
            if time.time() - t0 > _SEARCH_BUDGET:
                raise TimeoutError
            if j == len(unset):
                a = cipher[s0] * 10 + cipher[s1]
                b = cipher[s3] * 10 + cipher[s4]
                for op in ops:
                    kind, val = _value(a, b, op)
                    vneg, ds = _out_digits(kind, val)
                    if vneg != neg or len(ds) != len(ovd):
                        continue
                    applied = unify(list(ovd), ds)
                    if applied is None:
                        continue
                    new_op = opsym not in op_of
                    if new_op:
                        op_of[opsym] = op
                    if cont():
                        return True
                    if new_op:
                        del op_of[opsym]
                    _undo(applied)
                return False
            s = unset[j]
            for dd in range(10):
                if used[dd]:
                    continue
                used[dd] = True
                cipher[s] = dd
                if place(j + 1):
                    return True
                used[dd] = False
                del cipher[s]
            return False

        return place(0)

    def at_end() -> bool:
        qop = q[2]
        qsyms = [q[0], q[1], q[3], q[4]]
        if qop not in op_of or any(s not in cipher for s in qsyms):
            return False
        a = cipher[q[0]] * 10 + cipher[q[1]]
        b = cipher[q[3]] * 10 + cipher[q[4]]
        kind, val = _value(a, b, op_of[qop])
        vneg, ds = _out_digits(kind, val)
        inv = {v: k for k, v in cipher.items()}
        if any(int(ch) not in inv for ch in ds):
            return False
        result[0] = (qop if vneg else "") + "".join(inv[int(ch)] for ch in ds)
        return True

    def rec(i: int) -> bool:
        if i == len(order):
            return at_end()
        return solve_fact(order[i], lambda: rec(i + 1))

    try:
        rec(0)
    except TimeoutError:
        return None
    if result[0] is None:
        return None
    return _Solution(dict(cipher), dict(op_of), result[0])


_OP_SYMBOL = {
    "add": "+",
    "add1": "+",
    "addm1": "+",
    "sub": "-",
    "rsub": "-",
    "mul": "*",
    "mul1": "*",
    "mulm1": "*",
    "absdiff": "-",
    "negabs": "-",
    "maxmod": "mod",
    "concat": "||",
    "rconcat": "||",
}


def _op_expr(a: int, b: int, op: str) -> str:
    """Human-readable arithmetic for one operation, e.g. '34 * 39 = 1326'."""
    _kind, val = _value(a, b, op)
    table = {
        "add": f"{a} + {b}",
        "add1": f"{a} + {b} + 1",
        "addm1": f"{a} + {b} - 1",
        "sub": f"{a} - {b}",
        "rsub": f"{b} - {a}",
        "mul": f"{a} * {b}",
        "mul1": f"{a} * {b} + 1",
        "mulm1": f"{a} * {b} - 1",
        "absdiff": f"|{a} - {b}|",
        "negabs": f"-|{a} - {b}|",
        "maxmod": f"max({a},{b}) mod min({a},{b})",
        "concat": f"concat({a:02d},{b:02d})",
        "rconcat": f"concat({b:02d},{a:02d})",
    }
    if _kind == "s":
        return f"{table[op]} = {val}"
    return f"{table[op]} = {val}"


def _decode(sym_seq: str, cipher: Dict[str, int]) -> str:
    return "".join(str(cipher[c]) for c in sym_seq)


def _emit_trace(problem: Problem, sol: "_Solution") -> str:
    facts, q = _parse(problem)  # type: ignore[misc]
    cipher = sol.cipher
    op_of = sol.op_of_sym
    inv = {v: k for k, v in cipher.items()}
    lines: List[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append(
        "Each input is two 2-digit numbers AB and CD around an operator symbol; the"
    )
    lines.append(
        "output is the numeric result re-encoded in the SAME symbols. A single"
    )
    lines.append(
        "symbol<->digit cipher is shared by every line, and each operator symbol"
    )
    lines.append(
        "stands for one fixed operation. Negative results keep the operator symbol"
    )
    lines.append("as a sign prefix.")
    lines.append("")

    # Anchor: the fact with a 4-digit result is the most constraining (a mul/concat).
    anchor = None
    for s0, s1, opsym, s3, s4, ov in facts:
        if (
            len(ov.lstrip("-")) == 4
            and all(c in cipher for c in (s0, s1, s3, s4))
            and opsym in op_of
        ):
            anchor = (s0, s1, opsym, s3, s4, ov)
            break
    if anchor is not None:
        s0, s1, opsym, s3, s4, ov = anchor
        a = cipher[s0] * 10 + cipher[s1]
        b = cipher[s3] * 10 + cipher[s4]
        lines.append(
            f"Anchor on the 4-digit line {s0}{s1}{opsym}{s3}{s4} = {ov}: a 4-digit"
        )
        lines.append("result must be a product or concatenation, which pins the digits")
        lines.append(
            f"tightly. Here {s0}{s1}={a}, {s3}{s4}={b}, operator '{opsym}' = "
            f"{op_of[opsym]}: {_op_expr(a, b, op_of[opsym])}."
        )
        lines.append("")

    lines.append("Recovered cipher (symbol = digit):")
    lines.append("  " + ", ".join(f"{k}={cipher[k]}" for k in sorted(cipher)))
    lines.append("")
    lines.append("Operator meanings:")
    for opsym in sorted(op_of):
        lines.append(f"  '{opsym}' = {op_of[opsym]}")
    lines.append("")

    lines.append("Verify every example under this cipher and operators:")
    for s0, s1, opsym, s3, s4, ov in facts:
        if any(c not in cipher for c in (s0, s1, s3, s4)) or opsym not in op_of:
            continue
        a = cipher[s0] * 10 + cipher[s1]
        b = cipher[s3] * 10 + cipher[s4]
        op = op_of[opsym]
        _kind, val = _value(a, b, op)
        vneg, ds = _out_digits(_kind, val)
        if any(int(c) not in inv for c in ds):
            continue
        enc = (opsym if vneg else "") + "".join(inv[int(c)] for c in ds)
        ok = "OK" if enc == ov else "MISMATCH"
        lines.append(
            f"  {s0}{s1}{opsym}{s3}{s4}: {_op_expr(a, b, op)} -> encode {ds} -> "
            f"{enc} (expected {ov}) {ok}"
        )
    lines.append("")

    qop = q[2]
    qa = cipher[q[0]] * 10 + cipher[q[1]]
    qb = cipher[q[3]] * 10 + cipher[q[4]]
    _kind, val = _value(qa, qb, op_of[qop])
    vneg, ds = _out_digits(_kind, val)
    lines.append(f"Apply to the question {q[0]}{q[1]}{qop}{q[3]}{q[4]}:")
    lines.append(
        f"  {q[0]}{q[1]}={qa}, {q[3]}{q[4]}={qb}, '{qop}' = {op_of[qop]}: "
        f"{_op_expr(qa, qb, op_of[qop])}."
    )
    lines.append(f"  Encode {ds} back to symbols -> {sol.answer}.")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{-}} is \\boxed{{{sol.answer}}}")
    return "\n".join(lines)


_LABEL_POOL = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def _label_map(facts: List[_Fact], q: str) -> Dict[str, str]:
    """Assign each distinct digit-symbol a clean uniform label (operator symbols
    keep their raw character). Mirrors kemshim's symbol remap so the search reads
    uniformly regardless of which punctuation the puzzle used."""
    labels: Dict[str, str] = {}

    def add(sym: str) -> None:
        if sym not in labels and len(labels) < len(_LABEL_POOL):
            labels[sym] = _LABEL_POOL[len(labels)]

    for s0, s1, _opsym, s3, s4, ov in facts:
        for c in (s0, s1, s3, s4):
            add(c)
        for c in ov.lstrip("-"):
            add(c)
    for c in (q[0], q[1], q[3], q[4]):
        add(c)
    return labels


def _narrow_display(ov: str) -> List[str]:
    """Operator candidate set for an output, by signed digit length (kemshim's
    r.len pruning)."""
    neg = ov.startswith("-")
    rlen = len(ov.lstrip("-"))
    if neg:
        return ["sub", "rsub", "negabs"]
    if rlen == 4:
        return ["concat", "rconcat", "mul", "mul1", "mulm1"]
    if rlen == 3:
        return ["mul", "mul1", "mulm1", "add1", "addm1"]
    if rlen == 2:
        return ["add", "add1", "addm1", "sub", "rsub", "absdiff", "maxmod"]
    return list(_OP_PRIOR)


def _enum_anchor(
    s0: str, s1: str, s3: str, s4: str, ov: str, ops: List[str]
) -> List[Tuple[str, int, int, Dict[str, int]]]:
    """Enumerate every (op, a, b) whose result re-encodes consistently with the
    anchor line's symbol pattern -- the "mul pattern table"."""
    neg = ov.startswith("-")
    ovd = ov.lstrip("-")
    out: List[Tuple[str, int, int, Dict[str, int]]] = []
    seen = set()
    for op in ops:
        for a in range(100):
            for b in range(100):
                kind, val = _value(a, b, op)
                vneg, ds = _out_digits(kind, val)
                if vneg != neg or len(ds) != len(ovd):
                    continue
                assign: Dict[str, int] = {}
                used = set()
                ok = True
                pairs = [
                    (s0, a // 10),
                    (s1, a % 10),
                    (s3, b // 10),
                    (s4, b % 10),
                ] + list(zip(ovd, (int(c) for c in ds)))
                for sym, d in pairs:
                    if sym in assign:
                        if assign[sym] != d:
                            ok = False
                            break
                    elif d in used:
                        ok = False
                        break
                    else:
                        assign[sym] = d
                        used.add(d)
                if not ok:
                    continue
                key = (op, tuple(sorted(assign.items())))
                if key in seen:
                    continue
                seen.add(key)
                out.append((op, a, b, assign))
    return out


def _lab(seq: str, labels: Dict[str, str]) -> str:
    return "".join(labels.get(c, c) for c in seq)


def _emit_search_trace(problem: Problem, sol: "_Solution") -> str:
    """Detailed search trace: remap, operator narrowing, the mul-anchor candidate
    table, and the per-fact propagation that pins the cipher -- demonstrating HOW
    the answer is searched out, not just the verified result."""
    facts, q = _parse(problem)  # type: ignore[misc]
    cipher = sol.cipher
    op_of = sol.op_of_sym
    inv = {v: k for k, v in cipher.items()}
    labels = _label_map(facts, q)
    lines: List[str] = []

    lines.append(
        "We infer the transformation rule from the facts by bounded rule-based trial."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # --- remap block: parse every fact into labelled operands/operator/output ---
    lines.append("remap (index 2 is the operator):")
    for i, (s0, s1, opsym, s3, s4, ov) in enumerate(facts):
        neg = ov.startswith("-")
        ovd = ov.lstrip("-")
        a_lab = labels[s0] + labels[s1]
        b_lab = labels[s3] + labels[s4]
        r_lab = ("-" if neg else "") + _lab(ovd, labels)
        symset = sorted(set(a_lab + b_lab + _lab(ovd, labels)))
        lines.append(
            f"fact{i} [{s0}{s1}{opsym}{s3}{s4}] = [{ov}]: "
            f"{a_lab} {opsym} {b_lab} = {r_lab}  "
            f"(r.len={len(ovd)}{' neg' if neg else ''}, sym={symset}, |sym|={len(symset)})"
        )
    qa_lab = labels[q[0]] + labels[q[1]]
    qb_lab = labels[q[3]] + labels[q[4]]
    lines.append(
        f"query [{q[0]}{q[1]}{q[2]}{q[3]}{q[4]}]: {qa_lab} {q[2]} {qb_lab} = ?"
    )
    lines.append("")

    # --- narrowing: candidate operations per operator symbol, by r.len ---
    lines.append("narrowing operator candidates by result length:")
    by_opsym: Dict[str, List[str]] = {}
    for s0, s1, opsym, s3, s4, ov in facts:
        cands = _narrow_display(ov)
        prev = by_opsym.get(opsym)
        by_opsym[opsym] = (
            [c for c in cands if c in prev] if prev is not None else list(cands)
        )
    for opsym in sorted(by_opsym):
        lines.append(f"  '{opsym}' -> {{{', '.join(by_opsym[opsym])}}}")
    lines.append("")

    # --- mul anchor: the 4-digit fact with the most distinct symbols ---
    anchor_idx = None
    best = -1
    for i, (s0, s1, opsym, s3, s4, ov) in enumerate(facts):
        if len(ov.lstrip("-")) == 4 and opsym in op_of:
            nsym = len(set([s0, s1, s3, s4] + list(ov.lstrip("-"))))
            if nsym > best:
                best = nsym
                anchor_idx = i
    if anchor_idx is None:
        # No 4-digit anchor in the solved facts -> compact trace is clearer.
        return _emit_trace(problem, sol)

    s0, s1, aopsym, s3, s4, aov = facts[anchor_idx]
    anchor_ops = _narrow_display(aov)
    lines.append(
        f"Mul anchor: fact{anchor_idx} [{s0}{s1}{aopsym}{s3}{s4}] (4-digit result"
        " constrains the cipher most tightly)."
    )
    table = _enum_anchor(s0, s1, s3, s4, aov, anchor_ops)
    # The winning candidate matches the full solution on this fact.
    win = None
    for k, (op, a, b, assign) in enumerate(table):
        if op == op_of[aopsym] and all(
            cipher.get(sym) == d for sym, d in assign.items()
        ):
            win = k
            break
    cap = 24
    lines.append(
        f"Enumerate (operation, operands) consistent with the anchor pattern "
        f"{_lab(s0 + s1, labels)} {aopsym} {_lab(s3 + s4, labels)} = "
        f"{_lab(aov.lstrip('-'), labels)} -> {len(table)} candidates:"
    )
    for k, (op, a, b, assign) in enumerate(table[:cap]):
        mark = "  <-- consistent with all facts" if k == win else ""
        asg = ",".join(f"{labels[sym]}={d}" for sym, d in sorted(assign.items()))
        lines.append(f"  R{k}: {_op_expr(a, b, op)} -> {asg}{mark}")
    if len(table) > cap:
        lines.append(f"  ... {len(table) - cap} more candidates")
    lines.append("")

    if win is None:
        return _emit_trace(problem, sol)

    # --- propagation: starting from the winning anchor, fix each remaining fact ---
    wop, wa, wb, wassign = table[win]
    lines.append(
        f"Candidates are tried in order; R{win} is the first that extends to every"
        " fact. Propagate it:"
    )
    cur_ops: Dict[str, str] = {aopsym: wop}
    order = [anchor_idx] + [i for i in range(len(facts)) if i != anchor_idx]
    for i in order:
        s0, s1, opsym, s3, s4, ov = facts[i]
        if any(c not in cipher for c in (s0, s1, s3, s4)) or opsym not in op_of:
            continue
        a = cipher[s0] * 10 + cipher[s1]
        b = cipher[s3] * 10 + cipher[s4]
        winop = op_of[opsym]
        lines.append(f"fact{i} [{s0}{s1}{opsym}{s3}{s4}] = [{ov}]:")
        if opsym in cur_ops and i != anchor_idx:
            lines.append(f"  operator '{opsym}' already fixed = {cur_ops[opsym]}")
        # Try the candidate operations (prior order) and reject the ones that miss.
        cands = (
            [cur_ops[opsym]]
            if (opsym in cur_ops and i != anchor_idx)
            else _op_order(tuple(_narrow_display(ov)))
        )
        for op in cands:
            _kind, val = _value(a, b, op)
            vneg, ds = _out_digits(_kind, val)
            neg = ov.startswith("-")
            if vneg != neg or len(ds) != len(ov.lstrip("-")):
                lines.append(f"  try {op}: {_op_expr(a, b, op)} (length/sign off) NG")
                continue
            enc = (opsym if vneg else "") + "".join(
                inv[int(c)] if int(c) in inv else "?" for c in ds
            )
            if op == winop and enc == ov:
                lines.append(
                    f"  try {op}: {_op_expr(a, b, op)} -> encodes {enc} = {ov} OK -> accept"
                )
                break
            lines.append(
                f"  try {op}: {_op_expr(a, b, op)} -> encodes {enc} != {ov} NG"
            )
        cur_ops[opsym] = winop
    lines.append("")

    # --- recovered cipher + query application ---
    lines.append(
        "Recovered cipher: "
        + ", ".join(f"{labels[k]}({k})={cipher[k]}" for k in sorted(cipher))
    )
    qop = q[2]
    qa = cipher[q[0]] * 10 + cipher[q[1]]
    qb = cipher[q[3]] * 10 + cipher[q[4]]
    _kind, val = _value(qa, qb, op_of[qop])
    vneg, ds = _out_digits(_kind, val)
    lines.append(
        f"Query {q[0]}{q[1]}{qop}{q[3]}{q[4]}: {qa_lab}={qa}, {qb_lab}={qb}, "
        f"'{qop}' = {op_of[qop]}: {_op_expr(qa, qb, op_of[qop])}."
    )
    lines.append(f"Encode {ds} back to original symbols -> {sol.answer}.")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{-}} is \\boxed{{{sol.answer}}}")
    return "\n".join(lines)


def reasoning_cryptarithm(problem: Problem) -> Optional[str]:
    """Mul-anchored cipher solver with fallback to the original concatenation solver."""
    sol = _solve(problem)
    if sol is None:
        return _reasoning_original(problem)
    try:
        return _emit_search_trace(problem, sol)
    except Exception:
        return _emit_trace(problem, sol)
