"""v3 reasoning generator for 8-bit bit-manipulation tasks.

This is :mod:`reasoners.bit_manipulation_v2` with a purely *cosmetic* compaction
pass applied to the finished trace. v2's longest traces run past the corpus /
generation token budget (so the ``\\boxed{}`` answer gets truncated off the end),
driven mostly by the per-bit operator search dumping every candidate operand
layout. v3 trims only that redundancy -- never any reasoning -- so the boxed
answer is byte-for-byte identical to v2 and accuracy is unchanged; the traces
just fit the budget.

The full worked-example dumps (each ``Output N: <bits>`` followed by its eight
per-bit rows) are kept verbatim, so the header reads exactly like v1/v2.

Two transforms, both strictly information-preserving:

1. **Winner-safe cap on the operator enumeration.** Each operator family
   (Identity / NOT / Constant / AND / OR / XOR and their NOT-variants) lists
   every candidate operand layout it tries -- up to ~56 rows for the NOT-variant
   families. Only the rows that actually ``match`` an output column are referenced
   by the Left/Right/Best decision that follows, so every matched row is kept
   verbatim; the non-matching rows are capped at :data:`_OP_ROW_CAP` for context
   and the rest replaced by a ``+N more ... reject`` summary. Because the winner
   (a matched row) is never hidden, the forced-choice search stays intact -- this
   is the same enumerate/test/keep-winner/summarize-rest pattern v2 already uses
   for the offset-pair search, now applied to the per-bit operator search too.

2. **Collapse dead-end op blocks.** A fallback "Matching output" block whose
   every row is ``absent`` means that operator family matched no output column;
   the eight identical ``absent`` rows are replaced by a single
   ``(no column matches)`` line. Blocks with any real match are kept verbatim,
   since the Left/Right narrowing that follows references those matches.

Because the routing, search, derivation and verify steps are all preserved, v3
keeps exactly v2's learnable structure (show the search, force the winner,
verify against the examples) while fitting the token budget.
"""

from __future__ import annotations

import re
from typing import List, Optional

from reasoners.bit_manipulation_v2 import (
    reasoning_bit_manipulation as _reasoning_bit_manipulation_v2,
)
from reasoners.store_types import Problem

# A per-bit "Matching output" row: "0 absent" or "4 12 21 23 32".
_MATCH_ROW = re.compile(r"^[0-7] (?:absent|[\d ]+)$")

# Operator-family section headers in the per-bit (v1) search. Each is followed by
# a block of candidate operand rows ending at the next "Matching output" line.
# Only bare headers (whole line equals the name) start an enumeration; the
# "Selecting" summary uses suffixed lines like "AND none", which are left alone.
_OP_SECTIONS = frozenset(
    {"Identity", "NOT", "Constant", "AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"}
)
# Non-matching operand rows shown per operator section before summarizing. Rows
# that match an output column (the winner candidates the decision relies on) are
# ALWAYS kept on top of this, so the winner is never hidden behind the cap.
_OP_ROW_CAP = 12


def _compact(trace: str) -> str:
    """Cap the per-bit operator enumeration and collapse dead-end op blocks.

    Pure line filtering: every matched (winner) row and all reasoning the
    derivation references are kept, so the final ``\\boxed{}`` answer is
    preserved exactly. The worked-example dumps are left untouched.
    """
    lines = trace.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 1. Operator enumeration: cap non-matching candidate rows. The block runs
        #    from the bare operator header to the next "Matching output"; rows that
        #    match an output column are kept (the decision below references them),
        #    the rest are capped and summarized.
        if line in _OP_SECTIONS:
            j = i + 1
            block: List[str] = []
            while j < n and lines[j] != "Matching output":
                block.append(lines[j])
                j += 1
            rows = [r for r in block if r.strip()]
            unmatched = [r for r in rows if " match " not in r]
            out.append(line)
            if len(unmatched) <= _OP_ROW_CAP:
                out.extend(block)  # nothing worth hiding; keep verbatim layout
            else:
                budget = _OP_ROW_CAP
                hidden = 0
                for r in rows:
                    if " match " in r:
                        out.append(r)
                    elif budget > 0:
                        out.append(r)
                        budget -= 1
                    else:
                        hidden += 1
                if hidden:
                    out.append(f"+{hidden} more {line} rows reject (no column match)")
                out.append("")
            i = j
            continue

        # 2. Dead-end "Matching output" block: collapse an all-"absent" block to
        #    one line; keep blocks that contain any real match verbatim.
        if line == "Matching output":
            j = i + 1
            block: List[str] = []
            while j < n and _MATCH_ROW.match(lines[j]):
                block.append(lines[j])
                j += 1
            out.append(line)
            if block and all("absent" in r for r in block):
                out.append("(no column matches)")
            else:
                out.extend(block)
            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    """v2 trace with redundant restatement removed; identical boxed answer."""
    trace = _reasoning_bit_manipulation_v2(problem)
    if trace is None:
        return None
    return _compact(trace)
