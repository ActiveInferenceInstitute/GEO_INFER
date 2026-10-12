"""Unit-level negative controls for the executable H3 ACT contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = (
    REPO_ROOT / "GEO-INFER-TEST" / "validate_h3_active_inference_contract.py"
)


def load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "geo_infer_validate_h3_active_inference_contract", VALIDATOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pymdp_metadata_contract_rejects_cosmetic_backend_dictionary() -> None:
    validator = load_validator_module()

    with pytest.raises(AssertionError, match="missing action posterior"):
        validator._assert_pymdp_metadata(
            {
                "backend": "inferactively-pymdp",
                "pymdp_version": validator.REQUIRED_PYMDP_VERSION,
                "h3_version": "4.5.0",
            },
            "cosmetic",
        )


def test_pymdp_metadata_contract_rejects_non_normalized_posterior() -> None:
    validator = load_validator_module()

    with pytest.raises(AssertionError, match="posterior not normalized"):
        validator._assert_pymdp_metadata(
            {
                "backend": "inferactively-pymdp",
                "pymdp_version": validator.REQUIRED_PYMDP_VERSION,
                "h3_version": "4.5.0",
                "action_posterior": [0.4, 0.4],
                "negative_expected_free_energy": [0.1, 0.2],
            },
            "non_normalized",
        )


def test_h3_contract_scans_tool_directories_for_legacy_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = load_validator_module()
    module_root = tmp_path / "GEO-INFER-SPACE"
    tool_file = module_root / "src" / "geo_infer_space" / "tools" / "legacy.py"
    tool_file.parent.mkdir(parents=True)
    tool_file.write_text("import h3\ncell = h3.geo_to_h3(1, 2, 3)\n")

    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "RUNTIME_SOURCE_ROOTS", [module_root])
    monkeypatch.setattr(validator, "DOC_FILES", [])

    with pytest.raises(AssertionError, match="h3.geo_to_h3"):
        validator._validate_no_h3_v3_calls()
