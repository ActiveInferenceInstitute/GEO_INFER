"""Validate optional GNN notation audit metadata without changing model semantics.

The producer records available declarations before explicit caller options take
precedence. Only the derived timestep is authoritative here; space, units and
time-index records remain an audit trail, not new inference inputs or proof.
"""

from __future__ import annotations

import math
from typing import Any

_H3_NOTE = (
    "state_ids must still be supplied explicitly; derivation "
    "never fabricates H3 cell IDs"
)


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} requires exactly {sorted(expected)}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def validate_metadata_derivation(provenance: Any, *, step_seconds: float) -> None:
    """Admit the documented optional record, preserving strict unknown-key refusal.

    The record is retained by the artifact snapshot and its canonical digest.
    It neither authenticates the source digest nor re-derives source declarations.
    Explicit caller space options may override the recorded declaration; Gaussian
    units remain explicit and are never inferred from the categorical writer label.
    """
    base = {"producer", "source_sha256"}
    if not isinstance(provenance, dict) or set(provenance) not in (
        base,
        base | {"metadata_derivation"},
    ):
        raise ValueError(
            "provenance requires producer and source_sha256, with only "
            "optional metadata_derivation"
        )
    if "metadata_derivation" not in provenance:
        return
    record = provenance["metadata_derivation"]
    _keys(
        record,
        {"step_seconds", "space_kind", "units", "time_index", "discrete_time_declared"},
        "metadata_derivation",
    )
    step = record["step_seconds"]
    _keys(step, {"value", "derived", "sources"}, "derived step_seconds")
    value = step["value"]
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0 or value != step_seconds or step["derived"] is not True:
        raise ValueError(
            "derived step_seconds must equal the artifact's positive timestep"
        )
    sources = step["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("derived step_seconds requires nonempty sources")
    for source in sources:
        _keys(source, {"section", "key", "raw_value", "interpretation"}, "step source")
        if source["section"] not in ("Time", "ModelParameters") or source[
            "key"
        ] not in ("timestep", "stepseconds", "dt"):
            raise ValueError("step source must identify a supported declaration")
        _text(source["raw_value"], "step source raw_value")
        _text(source["interpretation"], "step source interpretation")
    space = record["space_kind"]
    if not isinstance(space, dict) or type(space.get("derived")) is not bool:
        raise ValueError("derived space_kind requires a boolean derived flag")
    if space["derived"]:
        expected = {"value", "derived", "source", "raw_value"}
        if space.get("value") == "h3":
            expected.add("note")
        _keys(space, expected, "derived space_kind")
        if (
            space["value"] not in ("categorical", "h3")
            or space["source"] != "## Time SpaceKind"
            or not isinstance(space["raw_value"], str)
            or space["raw_value"].lower() != space["value"]
            or (space["value"] == "h3" and space["note"] != _H3_NOTE)
        ):
            raise ValueError("derived space_kind does not match its declaration record")
    else:
        _keys(space, {"value", "derived", "source"}, "space_kind writer default")
        if space["value"] != "categorical" or space["source"] != "writer default":
            raise ValueError("space_kind writer default must retain the producer label")
    _keys(record["units"], {"derived", "reason"}, "derived units")
    if record["units"]["derived"] is not False:
        raise ValueError("coordinate units cannot be derived by this contract")
    _text(record["units"]["reason"], "units reason")
    index = record["time_index"]
    _keys(index, {"variable", "state_space_declaration", "ontology_term"}, "time_index")
    if any(
        value is not None and not isinstance(value, str) for value in index.values()
    ):
        raise ValueError("time_index entries must be strings or null")
    if index["variable"] is None and any(
        index[key] is not None for key in ("state_space_declaration", "ontology_term")
    ):
        raise ValueError("time_index corroboration requires a declared variable")
    if type(record["discrete_time_declared"]) is not bool:
        raise ValueError("discrete_time_declared must be a boolean audit flag")
