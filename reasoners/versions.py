"""Central registry of reasoning-solver versions per category.

A single ``versions.json`` at the repo root maps each problem category to the
solver version to use. All three pipeline tools read this registry so the choice
is consistent and version-controlled:

  * reasoning.py  -- generates traces with the selected solver, saving them to
                     ``reasoning/<category>/<version>/<id>.txt``;
  * corpus.py     -- tokenises those traces into ``corpus/<category>/<version>/``
                     and records the version in ``corpus.jsonl``;
  * training      -- selects, per category, which version's segments to train on.

Adding a new solver = add its function to ``REGISTRY`` here; nothing else needs
to import it directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from reasoners.bit_manipulation import reasoning_bit_manipulation as _bit_v1
from reasoners.bit_manipulation_v2 import reasoning_bit_manipulation as _bit_v2
from reasoners.cipher import reasoning_cipher as _cipher_v1
from reasoners.cryptarithm import reasoning_cryptarithm as _crypt_v1
from reasoners.cryptarithm_v2 import reasoning_cryptarithm as _crypt_v2
from reasoners.cryptarithm_v3 import reasoning_cryptarithm as _crypt_v3
from reasoners.cryptarithm_v4 import reasoning_cryptarithm as _crypt_v4
from reasoners.equation_numeric import reasoning_equation_numeric as _eqn_v1
from reasoners.equation_numeric_v2 import reasoning_equation_numeric as _eqn_v2
from reasoners.gravity import reasoning_gravity as _grav_v1
from reasoners.numeral import reasoning_numeral as _num_v1
from reasoners.unit_conversion import reasoning_unit_conversion as _unit_v1

# category -> {version -> solver function}
REGISTRY: dict[str, dict[str, Callable]] = {
    "bit_manipulation": {"v1": _bit_v1, "v2": _bit_v2},
    "cipher": {"v1": _cipher_v1},
    "equation_numeric_deduce": {"v1": _eqn_v1, "v2": _eqn_v2},
    "equation_numeric_guess": {"v1": _eqn_v1, "v2": _eqn_v2},
    "cryptarithm_deduce": {
        "v1": _crypt_v1,
        "v2": _crypt_v2,
        "v3": _crypt_v3,
        "v4": _crypt_v4,
    },
    "cryptarithm_guess": {
        "v1": _crypt_v1,
        "v2": _crypt_v2,
        "v3": _crypt_v3,
        "v4": _crypt_v4,
    },
    "gravity": {"v1": _grav_v1},
    "numeral": {"v1": _num_v1},
    "unit_conversion": {"v1": _unit_v1},
}

# Fallback version for any category absent from versions.json.
DEFAULT_VERSIONS: dict[str, str] = {cat: "v1" for cat in REGISTRY}

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_CONFIG = REPO_ROOT / "versions.json"


def load_versions(config_path: str | Path | None = None) -> dict[str, str]:
    """Read the ``category -> version`` selection from ``versions.json``.

    Categories omitted from the file fall back to :data:`DEFAULT_VERSIONS`.
    Unknown categories or versions raise ``ValueError`` so a typo fails loudly
    instead of silently training on the wrong traces.
    """
    path = Path(config_path) if config_path else VERSIONS_CONFIG
    chosen = dict(DEFAULT_VERSIONS)
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        for cat, ver in data.items():
            if cat not in REGISTRY:
                raise ValueError(f"{path.name}: unknown category {cat!r}")
            if ver not in REGISTRY[cat]:
                raise ValueError(
                    f"{path.name}: category {cat!r} has no version {ver!r} "
                    f"(available: {sorted(REGISTRY[cat])})"
                )
            chosen[cat] = ver
    return chosen


def get_solver(category: str, version: str) -> Callable:
    """Return the solver function for ``(category, version)``."""
    return REGISTRY[category][version]


def available_versions(category: str) -> list[str]:
    """Versions registered for ``category`` (empty if the category is unknown)."""
    return sorted(REGISTRY.get(category, {}))


def all_pairs() -> list[tuple[str, str]]:
    """Every ``(category, version)`` pair in the registry."""
    return [(cat, ver) for cat, vers in REGISTRY.items() for ver in vers]
