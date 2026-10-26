"""Optional declaration audit records retain strict wire and numerical contracts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference
from geo_infer_act.core.gnn_gaussian_contract import (
    GaussianGNNArtifact,
    run_gaussian_gnn_inference,
)


def categorical_data():
    return dict(
        schema_version="gnn-geo-infer/1",
        model_type="categorical",
        model_name="metadata categorical",
        dimensions=dict(states=2, observations=2, actions=2),
        matrices=dict(
            A=[[0.8, 0.2], [0.2, 0.8]],
            B=[[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            C=[0, 0],
            D=[0.5, 0.5],
            E=[0.01, 0.99],
        ),
        space=dict(kind="categorical", state_ids=["west", "east"]),
        time=dict(step_seconds=60),
        provenance=dict(producer="metadata regression", source_sha256="0" * 64),
    )


def gaussian_data():
    return dict(
        schema_version="gnn-geo-infer/2",
        model_type="linear_gaussian",
        model_name="metadata gaussian",
        dimensions=dict(states=1, observations=1, controls=1),
        matrices=dict(F=[[0.9]], G=[[1]], H=[[1]], Q=[[0.1]], R=[[0.5]]),
        initial_belief=dict(mean=[0], covariance=[[1]]),
        units=dict(states=["m"], observations=["m"], controls=["N"]),
        time=dict(domain="discrete", step_seconds=60),
        provenance=dict(producer="metadata regression", source_sha256="0" * 64),
    )


def derivation():
    return {
        "step_seconds": {
            "value": 60.0,
            "derived": True,
            "sources": [
                {
                    "section": "Time",
                    "key": "timestep",
                    "raw_value": "60s",
                    "interpretation": "60 s = 60 seconds",
                }
            ],
        },
        "space_kind": {
            "value": "categorical",
            "derived": False,
            "source": "writer default",
        },
        "units": {"derived": False, "reason": "Coordinate units remain explicit."},
        "time_index": {
            "variable": "t",
            "state_space_declaration": "t[1,type=int]",
            "ontology_term": "Time",
        },
        "discrete_time_declared": True,
    }


@pytest.fixture(
    params=[
        (
            GNNArtifact,
            categorical_data,
            "7b91f4b163252d6c462c8a5ad8e8955743d774b8d7cbfe88fd06c2afd7102c99",
        ),
        (
            GaussianGNNArtifact,
            gaussian_data,
            "988f5c25ef808f343de8e1d2ea06b6cbcf7b272449c59cd4d5c3795510dc9be2",
        ),
    ],
    ids=["categorical", "gaussian"],
)
def contract(request):
    return request.param


def test_legacy_digest_and_optional_snapshot_roundtrip(contract, tmp_path):
    factory, source, expected_digest = contract
    data = source()
    assert factory.from_dict(data).digest == expected_digest
    record = derivation()
    data["provenance"]["metadata_derivation"] = record
    artifact = factory.from_dict(data)
    assert artifact.digest != expected_digest
    expected = deepcopy(record)
    record["step_seconds"]["value"] = 999
    copied = artifact.to_dict()
    copied["provenance"]["metadata_derivation"]["units"]["reason"] = "changed"
    assert artifact.to_dict()["provenance"]["metadata_derivation"] == expected
    path = tmp_path / "derived.json"
    artifact.write(path)
    assert factory.load(path).digest == artifact.digest


@pytest.mark.parametrize(
    "case",
    [
        "record_not_object",
        "unknown_record_key",
        "unknown_provenance_key",
        "missing_step_key",
        "step_mismatch",
        "bool_step",
        "unbounded_integer",
        "nonfinite_step",
        "false_derived",
        "empty_sources",
        "unknown_source_key",
        "bad_source_section",
        "bad_source_key",
        "empty_source_text",
        "bad_space_flag",
        "unknown_space_key",
        "false_space_claim",
        "bad_space_raw",
        "bad_h3_note",
        "derived_units",
        "missing_units_reason",
        "bad_time_index",
        "orphan_time_index",
        "nonbool_discrete",
    ],
)
def test_malformed_or_conflicting_records_are_rejected(contract, case):
    factory, source, _ = contract
    data, record = source(), derivation()
    data["provenance"]["metadata_derivation"] = record
    if case == "record_not_object":
        data["provenance"]["metadata_derivation"] = []
    elif case == "unknown_record_key":
        record["extra"] = True
    elif case == "unknown_provenance_key":
        data["provenance"]["extra"] = True
    elif case == "missing_step_key":
        del record["step_seconds"]["derived"]
    elif case in {"step_mismatch", "bool_step", "unbounded_integer", "nonfinite_step"}:
        record["step_seconds"]["value"] = {
            "step_mismatch": 120,
            "bool_step": True,
            "unbounded_integer": 10**400,
            "nonfinite_step": float("inf"),
        }[case]
    elif case == "false_derived":
        record["step_seconds"]["derived"] = False
    elif case == "empty_sources":
        record["step_seconds"]["sources"] = []
    elif case == "unknown_source_key":
        record["step_seconds"]["sources"][0]["extra"] = True
    elif case in {"bad_source_section", "bad_source_key", "empty_source_text"}:
        field, value = {
            "bad_source_section": ("section", "Guessed"),
            "bad_source_key": ("key", "automatic"),
            "empty_source_text": ("raw_value", " "),
        }[case]
        record["step_seconds"]["sources"][0][field] = value
    elif case == "bad_space_flag":
        record["space_kind"]["derived"] = 0
    elif case == "unknown_space_key":
        record["space_kind"]["extra"] = True
    elif case == "false_space_claim":
        record["space_kind"]["value"] = "h3"
    elif case in {"bad_space_raw", "bad_h3_note"}:
        record["space_kind"] = {
            "value": "h3",
            "derived": True,
            "source": "## Time SpaceKind",
            "raw_value": "h3" if case == "bad_h3_note" else "categorical",
            "note": "bad"
            if case == "bad_h3_note"
            else "state_ids must still be supplied explicitly; derivation never fabricates H3 cell IDs",
        }
    elif case == "derived_units":
        record["units"]["derived"] = True
    elif case == "missing_units_reason":
        del record["units"]["reason"]
    elif case == "bad_time_index":
        record["time_index"]["variable"] = 5
    elif case == "orphan_time_index":
        record["time_index"]["variable"] = None
    elif case == "nonbool_discrete":
        record["discrete_time_declared"] = 1
    with pytest.raises(ValueError):
        factory.from_dict(data)


def test_nested_duplicate_keys_remain_rejected(contract):
    factory, source, _ = contract
    data = source()
    data["provenance"]["metadata_derivation"] = derivation()
    raw = json.dumps(data).replace(
        '"derived": true', '"derived": true, "derived": false', 1
    )
    with pytest.raises(ValueError, match="Duplicate"):
        factory(raw)


def test_declared_space_is_advisory_after_explicit_override(contract):
    factory, source, _ = contract
    data, record = source(), derivation()
    record["space_kind"] = {
        "value": "h3",
        "derived": True,
        "source": "## Time SpaceKind",
        "raw_value": "H3",
        "note": "state_ids must still be supplied explicitly; derivation never fabricates H3 cell IDs",
    }
    record["time_index"] = dict(
        variable=None, state_space_declaration=None, ontology_term=None
    )
    record["discrete_time_declared"] = False
    data["provenance"]["metadata_derivation"] = record
    assert (
        factory.from_dict(data).to_dict()["provenance"]["metadata_derivation"] == record
    )


@pytest.mark.parametrize("kind", ["categorical", "gaussian"])
def test_optional_metadata_does_not_change_numerical_trace(kind):
    data = categorical_data() if kind == "categorical" else gaussian_data()
    factory = GNNArtifact if kind == "categorical" else GaussianGNNArtifact
    runner = run_gnn_inference if kind == "categorical" else run_gaussian_gnn_inference
    records = [dict(timestamp="2026-01-01T00:00:00Z", observation=0)]
    if kind == "gaussian":
        records[0].update(observation=[1.0], control=[0.25])
    base = runner(factory.from_dict(data), records)
    data["provenance"]["metadata_derivation"] = derivation()
    derived = runner(factory.from_dict(data), records)
    assert base["artifact_sha256"] != derived["artifact_sha256"]
    assert {k: v for k, v in base.items() if k != "artifact_sha256"} == {
        k: v for k, v in derived.items() if k != "artifact_sha256"
    }


def test_factored_provenance_is_not_extended():
    from geo_infer_act.core.gnn_factored_contract import FactoredGNNArtifact

    data = json.loads((Path(__file__).with_name("factored_example.json")).read_text())
    data.update(
        schema_version="gnn-geo-infer/factored/1",
        model_type="categorical_factored",
        time={"step_seconds": 60},
        provenance={
            "producer": "test",
            "source_kind": "explicit_factored_json",
            "source_sha256": "0" * 64,
            "metadata_derivation": derivation(),
        },
    )
    with pytest.raises(ValueError, match="provenance"):
        FactoredGNNArtifact.from_dict(data)
