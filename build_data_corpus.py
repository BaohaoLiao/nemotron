"""Rebuild data/corpus.jsonl from the freshly generated root corpus.jsonl,
marking the 500 held-out test_500 problems as included=False.

Rule (verified against the committed data/corpus.jsonl):
    data/corpus.jsonl == root corpus.jsonl, with included=False for exactly the
    problem_ids listed in data/test_500.csv, included=True otherwise.

The eval/train split depends on this holdout marking, so we assert the excluded
set equals the test_500 id set before writing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).parent
ROOT_CORPUS = BASE / "corpus.jsonl"
DATA_CORPUS = BASE / "data" / "corpus.jsonl"
TEST_CSV = BASE / "data" / "test_500.csv"


def main() -> None:
    test_ids: set[str] = set()
    with open(TEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            test_ids.add(row["id"])
    assert len(test_ids) == 500, f"expected 500 test ids, got {len(test_ids)}"

    rows: list[dict] = []
    with open(ROOT_CORPUS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    excluded: set[str] = set()
    for r in rows:
        pid = r["problem_id"]
        if pid in test_ids:
            r["included"] = False
            excluded.add(pid)
        else:
            r["included"] = True

    missing = test_ids - excluded
    if missing:
        print(
            f"WARNING: {len(missing)} test_500 ids have no corpus entry "
            f"(not in training, no leak): {sorted(missing)[:10]}"
        )

    DATA_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_CORPUS, "w") as f:
        for r in rows:
            json.dump(r, f)
            f.write("\n")

    incl = sum(1 for r in rows if r["included"])
    exc = sum(1 for r in rows if not r["included"])
    print(f"root corpus.jsonl entries : {len(rows)}")
    print(f"data/corpus.jsonl written : included={incl} excluded={exc}")
    print(f"excluded == test_500 set  : {excluded == test_ids}")


if __name__ == "__main__":
    main()
