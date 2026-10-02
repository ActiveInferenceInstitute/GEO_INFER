"""Adversarial coverage of the data-only algorithm persistence boundary."""

import json

import pytest

from geo_infer_art.core.generation.custom_algorithms import (
    CustomAlgorithmFramework,
    example_cellular_growth_algorithm,
)


def entry(key="cellular_growth"):
    return {
        "registry_key": key,
        "metadata": {"description": "", "parameters": {}, "example_usage": ""},
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"schema_version": True, "algorithms": {}},
        {"schema_version": 2, "algorithms": {}},
        {"schema_version": 1, "algorithms": []},
        {"schema_version": 1, "algorithms": {"": entry()}},
        {"schema_version": 1, "algorithms": {"bad": entry("os.system")}},
        {"schema_version": 1, "algorithms": {"bad": entry(["cellular_growth"])}},
        {
            "schema_version": 1,
            "algorithms": {"bad": {**entry(), "source": "raise RuntimeError()"}},
        },
        {
            "schema_version": 1,
            "algorithms": {
                "bad": {"registry_key": "cellular_growth", "metadata": None}
            },
        },
    ],
)
def test_rejects_untrusted_format_atomically(tmp_path, payload):
    path = tmp_path / "algorithms.json"
    path.write_text(json.dumps(payload))
    framework = CustomAlgorithmFramework()
    with pytest.raises(ValueError):
        framework.load_algorithms_from_file(str(path))
    assert framework.list_algorithms() == []


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1,"algorithms":{}}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        CustomAlgorithmFramework().load_algorithms_from_file(str(path))


def test_loading_collision_is_atomic(tmp_path):
    framework = CustomAlgorithmFramework()
    framework.register_algorithm("existing", example_cellular_growth_algorithm)
    path = tmp_path / "collision.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "algorithms": {"new": entry(), "existing": entry()}}
        )
    )
    with pytest.raises(ValueError, match="already registered"):
        framework.load_algorithms_from_file(str(path))
    assert framework.list_algorithms() == ["existing"]


@pytest.mark.parametrize(
    "constant", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"]
)
def test_nonfinite_json_metadata_rejected_without_mutation(tmp_path, constant):
    framework = CustomAlgorithmFramework()
    framework.register_algorithm("existing", example_cellular_growth_algorithm)
    path = tmp_path / "nonfinite.json"
    payload = {"schema_version": 1, "algorithms": {"new": entry()}}
    payload["algorithms"]["new"]["metadata"]["parameters"] = {"nested": [None]}
    path.write_text(json.dumps(payload).replace("null", constant))
    with pytest.raises(ValueError, match="Non-finite JSON value"):
        framework.load_algorithms_from_file(str(path))
    assert framework.list_algorithms() == ["existing"]
    framework.save_algorithms_to_file(str(tmp_path / "still-valid.json"))


@pytest.mark.parametrize("name", [None, 1, True, "", []])
def test_registration_requires_nonempty_string_name(name):
    framework = CustomAlgorithmFramework()
    with pytest.raises(ValueError, match="name must be a nonempty string"):
        framework.register_algorithm(name, example_cellular_growth_algorithm)
    assert framework.list_algorithms() == []


def test_builtin_roundtrip_preserves_globals_and_seed(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    source = CustomAlgorithmFramework()
    source.register_algorithm("growth", example_cellular_growth_algorithm)
    path = tmp_path / "registry.json"
    source.save_algorithms_to_file(str(path))
    target = CustomAlgorithmFramework()
    target.load_algorithms_from_file(str(path))
    figures = [
        framework.execute_algorithm("growth", None, 32, 24, seed=42)
        for framework in (source, target)
    ]
    try:
        for figure in figures:
            figure.canvas.draw()
        np.testing.assert_array_equal(
            np.asarray(figures[0].canvas.buffer_rgba()),
            np.asarray(figures[1].canvas.buffer_rgba()),
        )
    finally:
        for figure in figures:
            plt.close(figure)
