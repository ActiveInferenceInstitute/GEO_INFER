#!/usr/bin/env python3
"""Validate the GEO-INFER-ACT script orchestration contract.

This validator checks that Active Inference scripts are thin CLI wrappers over
``geo_infer_act.runners`` and that the package exposes schema-backed scenario
execution entrypoints.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "GEO-INFER-ACT"
ACT_SRC = ACT_ROOT / "src"
SCRIPT_PATHS = [
    ACT_ROOT / "debug_models.py",
    ACT_ROOT / "verify_pipeline.py",
    ACT_ROOT / "examples" / "simple_model.py",
    ACT_ROOT / "examples" / "modern_active_inference.py",
    ACT_ROOT / "examples" / "spatial_inference_demo.py",
    ACT_ROOT / "examples" / "h3_active_inference.py",
    ACT_ROOT / "examples" / "ecological_model.py",
    ACT_ROOT / "examples" / "urban_planning.py",
    ACT_ROOT / "examples" / "run_all_examples.py",
]
MAX_WRAPPER_STATEMENTS = 90


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def _module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _statement_count(path: Path) -> int:
    tree = ast.parse(path.read_text(), filename=str(path))
    return sum(isinstance(node, ast.stmt) for node in ast.walk(tree))


def validate_package_contract() -> None:
    runners = importlib.import_module("geo_infer_act.runners")
    required = {
        "RunConfig",
        "ScenarioRunResult",
        "SuiteRunResult",
        "SCENARIO_NAMES",
        "load_run_config",
        "run_scenario",
        "run_all_scenarios",
    }
    missing = sorted(name for name in required if not hasattr(runners, name))
    if missing:
        _fail(f"geo_infer_act.runners missing exports: {missing}")

    schemas_dir = ACT_SRC / "geo_infer_act" / "schemas"
    required_schemas = {
        "run_config.schema.json",
        "run_manifest.schema.json",
        "step_metrics.schema.json",
        "h3_diagnostics.schema.json",
    }
    missing_schemas = sorted(
        name for name in required_schemas if not (schemas_dir / name).exists()
    )
    if missing_schemas:
        _fail(f"Missing ACT runner schema files: {missing_schemas}")

    manifest_schema = json.loads((schemas_dir / "run_manifest.schema.json").read_text())
    schema_text = json.dumps(manifest_schema)
    for required_field in (
        "artifact_type",
        "mime_type",
        "sha256",
        "figure_metadata_path",
        "figure_data_path",
        "data_sources",
        "plotted_metrics",
        "alt_text",
    ):
        if required_field not in schema_text:
            _fail(f"run_manifest.schema.json missing {required_field}")


def validate_wrappers() -> None:
    for path in SCRIPT_PATHS:
        if not path.exists():
            _fail(f"Missing script wrapper: {path}")
        functions = _defined_functions(path)
        imports = _module_names(path)
        if "main" not in functions:
            _fail(f"{path} does not expose main()")
        if "argparse" not in imports:
            _fail(f"{path} does not use argparse for CLI flags")
        if not any(name.startswith("geo_infer_act.runners") for name in imports):
            _fail(f"{path} does not dispatch into geo_infer_act.runners")
        statements = _statement_count(path)
        if statements > MAX_WRAPPER_STATEMENTS:
            _fail(
                f"{path} has {statements} AST statements; expected a thin wrapper "
                f"with <= {MAX_WRAPPER_STATEMENTS}"
            )
        text = path.read_text()
        if "subprocess." in text:
            _fail(f"{path} shells out with subprocess instead of package runners")


def main() -> None:
    validate_package_contract()
    validate_wrappers()
    print("[OK] GEO-INFER-ACT script orchestration contract is valid")


if __name__ == "__main__":
    main()
