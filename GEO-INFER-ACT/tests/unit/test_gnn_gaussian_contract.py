"""Unequal-axis analytic filtering and fail-closed Gaussian interchange tests."""

from copy import deepcopy

import numpy as np
import pytest

from geo_infer_act.core.gnn_gaussian_contract import (
    GaussianGNNArtifact,
    run_gaussian_gnn_inference,
)

# Declared comparison precision for the admitted E4 fixture (GNN witness v2).
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12


def gaussian_data():
    return dict(
        schema_version="gnn-geo-infer/2",
        model_type="linear_gaussian",
        model_name="rectangular",
        dimensions=dict(states=3, observations=2, controls=1),
        matrices=dict(
            F=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
            G=[[1.0], [2.0], [0.0]],
            H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Q=(np.eye(3) * 0.1).tolist(),
            R=[[0.5, 0.0], [0.0, 2.0]],
        ),
        initial_belief=dict(
            mean=[0.0, 0.0, 0.0], covariance=np.diag([1.0, 4.0, 9.0]).tolist()
        ),
        units=dict(states=["m", "m/s", "K"], observations=["m", "m/s"], controls=["N"]),
        time=dict(domain="discrete", step_seconds=2),
        provenance=dict(producer="analytic fixture", source_sha256="0" * 64),
    )


def records():
    return [
        dict(timestamp="2026-09-04T00:00:00Z", observation=[1.0, 2.0], control=[0.25]),
        dict(timestamp="2026-09-04T00:00:02Z", observation=[2.0, 1.0], control=[-0.5]),
    ]


def test_exact_rectangular_posterior_and_one_transition(tmp_path):
    data = gaussian_data()
    artifact = GaussianGNNArtifact.from_dict(data)
    trace = run_gaussian_gnn_inference(artifact, records())
    first, second = trace["steps"]
    np.testing.assert_allclose(
        first["posterior_mean"],
        [2 / 3, 4 / 3, 0],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        first["posterior_covariance"],
        np.diag([1 / 3, 4 / 3, 9]),
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        first["next_prior_mean"],
        [19 / 12, 11 / 6, 0],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        first["next_prior_covariance"],
        np.diag([43 / 30, 43 / 30, 47 / 20]),
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    np.testing.assert_allclose(
        second["prior_mean"],
        first["next_prior_mean"],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    expected = 0.5 * (2 * np.log(2 * np.pi) + np.log(9) + 1 / 1.5 + 4 / 6)
    assert first["negative_log_evidence"] == pytest.approx(
        expected, rel=REFERENCE_RTOL, abs=REFERENCE_ATOL
    )
    assert run_gaussian_gnn_inference(artifact, records()) == trace
    dest = tmp_path / "model.json"
    artifact.write(dest)
    assert GaussianGNNArtifact.load(dest).digest == artifact.digest
    data["matrices"]["F"][0][0] = 99
    assert artifact.to_dict()["matrices"]["F"][0][0] == 2


@pytest.mark.parametrize(
    "case",
    [
        "generator",
        "singular_R",
        "negative_Q",
        "asymmetric_cov",
        "wrong_G",
        "missing_units",
        "bool_F",
        "huge_dims",
    ],
)
def test_invalid_artifact_rejected(case):
    data = gaussian_data()
    if case == "generator":
        data["time"]["domain"] = "continuous"
    elif case == "singular_R":
        data["matrices"]["R"][0][0] = 0
    elif case == "negative_Q":
        data["matrices"]["Q"][0][0] = -0.1
    elif case == "asymmetric_cov":
        data["initial_belief"]["covariance"][0][1] = 0.1
    elif case == "wrong_G":
        data["matrices"]["G"] = [[1.0, 2.0, 3.0]]
    elif case == "missing_units":
        del data["units"]["controls"]
    elif case == "bool_F":
        data["matrices"]["F"][0][0] = True
    elif case == "huge_dims":
        data["dimensions"]["states"] = 100_000_000
    with pytest.raises(ValueError):
        GaussianGNNArtifact.from_dict(data)


@pytest.mark.parametrize(
    "case",
    [
        "gap",
        "naive",
        "wrong_observation",
        "wrong_control",
        "nonfinite",
        "missing_control",
        "too_many",
    ],
)
def test_invalid_records_rejected(case):
    artifact = GaussianGNNArtifact.from_dict(gaussian_data())
    inputs = deepcopy(records())
    if case == "gap":
        inputs[1]["timestamp"] = "2026-09-04T00:00:04Z"
    elif case == "naive":
        inputs[0]["timestamp"] = "2026-09-04T00:00:00"
    elif case == "wrong_observation":
        inputs[0]["observation"] = [1.0]
    elif case == "wrong_control":
        inputs[0]["control"] = [1.0, 2.0]
    elif case == "nonfinite":
        inputs[0]["observation"][0] = float("nan")
    elif case == "missing_control":
        del inputs[0]["control"]
    with pytest.raises(ValueError):
        run_gaussian_gnn_inference(
            artifact, inputs, max_steps=1 if case == "too_many" else 10_000
        )


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        GaussianGNNArtifact('{"schema_version": "a", "schema_version": "b"}')


def test_shape_rejected_before_dense_allocation(monkeypatch):
    from geo_infer_act.core import gnn_gaussian_contract as contract

    data = gaussian_data()
    data["matrices"]["F"] = [[1.0]]

    def forbidden(*args, **kwargs):
        raise AssertionError("dense allocation preceded shape validation")

    monkeypatch.setattr(contract.np, "asarray", forbidden)
    with pytest.raises(ValueError, match="shape"):
        GaussianGNNArtifact.from_dict(data)


@pytest.mark.parametrize("field", ["Q", "R", "covariance"])
def test_extreme_indefinite_covariance_cannot_pass_nan_comparisons(field):
    data = gaussian_data()
    data["dimensions"] = dict(states=2, observations=2, controls=1)
    data["matrices"] = dict(
        F=[[1.0, 0.0], [0.0, 1.0]],
        G=[[1.0], [0.0]],
        H=[[1.0, 0.0], [0.0, 1.0]],
        Q=[[1.0, 0.0], [0.0, 1.0]],
        R=[[1.0, 0.0], [0.0, 1.0]],
    )
    data["initial_belief"] = dict(mean=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 1.0]])
    data["units"]["states"] = ["m", "m"]
    container = data["initial_belief"] if field == "covariance" else data["matrices"]
    container[field] = [[-1e308, 0.0], [0.0, 1e308]]
    with pytest.raises(ValueError, match="positive"):
        GaussianGNNArtifact.from_dict(data)


def test_runtime_rejects_innovation_overflow():
    data = gaussian_data()
    data["matrices"]["H"][0][0] = 1e308
    artifact = GaussianGNNArtifact.from_dict(data)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="innovation overflowed"):
            run_gaussian_gnn_inference(artifact, records())
