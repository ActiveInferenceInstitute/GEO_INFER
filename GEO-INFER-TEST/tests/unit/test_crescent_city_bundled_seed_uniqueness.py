"""Pins the bundled Crescent City seed invariant for the civic-intel demo.

The demo's ``_BUNDLED_SEED_RELATIVES`` lists every module checkout that may
ship a reviewed ``crescent-city-geo-intel/v1`` seed (BAYES and PLACE), and
``bundled_contract_path()`` resolves the first existing candidate.  That
first-match resolution is only safe because every bundled candidate is
byte-identical: together the candidates carry exactly ONE distinct reviewed
seed, and the canonical BAYES packaged loader reads that same seed.

Post-consolidation (RISK's duplicate was deleted) exactly one *packaged*
canonical copy ships — the BAYES resource consumed by
``geo_infer_bayes.civic_intel.load_crescent_city_contract`` — while the PLACE
relative is PLACE-owned dashboard data mirroring the same bytes.

These tests pin that invariant so a future drift between the bundled copies
(or a divergent schema) is surfaced here in one place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

# GEO-INFER-TEST/tests/conftest.py puts the GEO-INFER-TEST script directory
# (which holds the ``demo`` package) on sys.path.
from demo.crescent_city_civic_intel_demo import (
    _BUNDLED_SEED_RELATIVES,
    bundled_contract_path,
    load_bundled_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_EXPECTED_SCHEMA = "crescent-city-geo-intel/v1"


def _candidate_paths() -> list[Path]:
    """Resolve every ``_BUNDLED_SEED_RELATIVES`` entry against the repo root."""
    return [REPO_ROOT / relative for relative in _BUNDLED_SEED_RELATIVES]


def _existing_candidates() -> list[Path]:
    return [path for path in _candidate_paths() if path.is_file()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bayes_canonical_contract() -> Any:
    """Load the packaged BAYES seed through the canonical shared loader."""
    from geo_infer_bayes.civic_intel import load_crescent_city_contract

    return load_crescent_city_contract(None)


def test_candidates_resolve_to_exactly_one_distinct_bundled_copy() -> None:
    """Every existing bundled candidate carries the SAME reviewed seed bytes.

    ``bundled_contract_path()`` prefers the first existing relative, which is
    only deterministic in outcome if all bundled candidates are byte-identical
    — i.e. the checkout ships exactly one distinct bundled copy regardless of
    which mirror the first-match resolution lands on.
    """
    existing = _existing_candidates()
    assert existing, (
        "no bundled crescent-city-geo-intel seed found; expected at least one "
        f"of {_BUNDLED_SEED_RELATIVES}"
    )

    digests = {_digest(path) for path in existing}
    assert len(digests) == 1, (
        "bundled crescent-city-geo-intel candidates diverged; exactly one "
        f"distinct seed is allowed, got SHA-256 set {digests}"
    )

    # The resolver's first match must be one of the bundled candidates.
    resolved = bundled_contract_path()
    assert resolved in existing


def test_resolved_bundled_seed_loads_to_schema_v1() -> None:
    """The existing bundled copy loads as a crescent-city-geo-intel/v1 contract."""
    contract = load_bundled_contract()
    assert contract["schema"] == _EXPECTED_SCHEMA
    assert bundled_contract_path().is_file()


def test_bayes_canonical_loader_reads_the_same_bundled_copy() -> None:
    """The BAYES packaged loader yields the identical v1 seed bytes."""
    resolved = bundled_contract_path()

    contract = _load_bayes_canonical_contract()
    assert contract is not None
    assert contract["schema"] == _EXPECTED_SCHEMA

    packaged = (
        REPO_ROOT / "GEO-INFER-BAYES/src/geo_infer_bayes/crescent-city-geo-intel.json"
    )
    assert packaged.is_file()
    assert _digest(packaged) == _digest(resolved)
