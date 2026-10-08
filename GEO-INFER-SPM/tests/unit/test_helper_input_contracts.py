"""Input rejection and numerical oracles for public SPM construction helpers."""

import subprocess
import sys
import inspect

import numpy as np
import pytest

from geo_infer_spm.models.data_models import SPMData
from geo_infer_spm.utils.helpers import (
    create_design_matrix,
    create_spatial_basis_functions,
    generate_coordinates,
    generate_synthetic_data,
)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.bool_(False)])
@pytest.mark.parametrize("grid", ["regular", "random", "clustered"])
def test_coordinate_count_is_a_positive_integer(count, grid):
    with pytest.raises(ValueError, match="n_points"):
        generate_coordinates(grid, n_points=count)


@pytest.mark.parametrize(
    "bounds",
    [(2, 1, 0, 1), (0, 1, 2, 1), (0, 1, 2), (0, np.inf, 0, 1), (0, 1, np.nan, 1)],
)
def test_coordinate_bounds_are_finite_and_ordered(bounds):
    with pytest.raises(ValueError, match="bounds"):
        generate_coordinates(bounds=bounds)


@pytest.mark.parametrize(
    "options",
    [
        {"n_clusters": 0},
        {"n_clusters": 1.5},
        {"n_clusters": True},
        {"cluster_std": -1},
        {"cluster_std": np.nan},
        {"cluster_std": np.inf},
        {"cluster_std": "wide"},
    ],
)
def test_cluster_options_reject_invalid_inputs_without_consuming_rng(options):
    rng = np.random.default_rng(17)
    control = np.random.default_rng(17)
    with pytest.raises(ValueError, match="n_clusters|cluster_std"):
        generate_coordinates("clustered", random_seed=rng, **options)
    assert rng.integers(2**32) == control.integers(2**32)


@pytest.mark.parametrize("method", ["gaussian", "polynomial", "fourier"])
def test_colocated_basis_is_finite(method):
    with np.errstate(all="raise"):
        basis = create_spatial_basis_functions(
            np.zeros((4, 2)), n_basis=6, method=method, random_seed=3
        )
    assert basis.shape == (4, 6)
    assert np.isfinite(basis).all()
    if method == "gaussian":
        np.testing.assert_array_equal(basis, np.ones((4, 6)))
    elif method == "polynomial":
        np.testing.assert_array_equal(basis[:, 0], np.ones(4))
        np.testing.assert_array_equal(basis[:, 1:], np.zeros((4, 5)))


@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_basis_count_is_positive_integer(count):
    with pytest.raises(ValueError, match="n_basis"):
        create_spatial_basis_functions(np.zeros((4, 2)), n_basis=count)


def test_polynomial_default_terminates_and_matches_total_degree_oracle():
    # Execute the production function's definition with real NumPy in a bounded
    # child. Importing the entire SPM package in that child would also initialize
    # optional samplers, conflating their startup with this algorithm's budget.
    code = """
import numpy as np
import sys
namespace = {"np": np}
exec("from __future__ import annotations\\n" + sys.argv[1], namespace)
coords = np.array([[-1., -2.], [0., 0.], [1., 2.]])
x, y = ((coords - coords.mean(axis=0)) / coords.std(axis=0)).T
expected = np.column_stack([np.ones(3), y, x, y*y, x*y, x*x, y**3, x*y*y, x*x*y, x**3])
actual = namespace["create_spatial_basis_functions"](coords, method="polynomial")
np.testing.assert_allclose(actual, expected)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, inspect.getsource(create_spatial_basis_functions)],
        timeout=20,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    # The public module path is already imported in the parent; check its actual
    # callable as well as the bounded production-definition execution above.
    coords = np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    x, y = ((coords - coords.mean(axis=0)) / coords.std(axis=0)).T
    expected = np.column_stack(
        [np.ones(3), y, x, y * y, x * y, x * x, y**3, x * y * y, x * x * y, x**3]
    )
    np.testing.assert_allclose(
        create_spatial_basis_functions(coords, method="polynomial"), expected
    )


@pytest.mark.parametrize(
    "coords", [np.empty((0, 2)), np.zeros(3), np.zeros((3, 3)), np.array([[np.nan, 0]])]
)
@pytest.mark.parametrize(
    "helper", [create_spatial_basis_functions, generate_synthetic_data]
)
def test_coordinate_consumers_reject_invalid_shape_and_nonfinite_values(coords, helper):
    with pytest.raises(ValueError, match="coordinates"):
        helper(coords)


def test_radial_signal_at_coincident_coordinates_is_finite():
    with np.errstate(all="raise"):
        data = generate_synthetic_data(
            np.zeros((4, 2)), effects={"trend": "radial"}, noise_level=0, random_seed=1
        )
    np.testing.assert_array_equal(data.data, np.full(4, 5.0))


def _data():
    return SPMData(
        data=np.arange(3.0),
        coordinates=np.zeros((3, 2)),
        covariates={
            "x": np.arange(3.0),
            "y": np.arange(3.0) + 2,
            "kind": np.array(["a", "b", "a"]),
        },
    )


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x * missing",
        "y ~ missing * x",
        "y ~ 0",
        "y ~",
        "~ x",
        "y ~ x ~ y",
        "y ~ x +",
    ],
)
def test_formula_rejects_silent_omission_and_empty_design(formula):
    with pytest.raises(ValueError):
        create_design_matrix(_data(), formula=formula)


def test_supported_formula_interaction_matches_exact_values():
    data = _data()
    design = create_design_matrix(data, formula="response ~ x + x * y")
    assert design.names == ["intercept", "x", "x:y"]
    np.testing.assert_array_equal(
        design.matrix,
        np.column_stack(
            [
                np.ones(3),
                data.covariates["x"],
                data.covariates["x"] * data.covariates["y"],
            ]
        ),
    )


@pytest.mark.parametrize("levels", [[], ["a", "a"], ["a", "c"]])
def test_factor_encoding_rejects_ambiguous_levels(levels):
    with pytest.raises(ValueError, match="Factor"):
        create_design_matrix(_data(), factors={"kind": levels})
