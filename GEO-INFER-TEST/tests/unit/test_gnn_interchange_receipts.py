"""GNN layout and receipt failures are explicit and machine-readable."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "GEO-INFER-TEST" / "validate_gnn_interchange.py"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "gnn_interchange_receipt_regression", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exporter_tree(root: Path, prefix: str, body: str) -> None:
    package = root / "src" / prefix.replace(".", "/")
    package.mkdir(parents=True)
    (package / "geo_infer.py").write_text(body)


@pytest.mark.parametrize("prefix", ["gnn.export", "export"])
def test_declared_layout_is_imported_from_selected_checkout(
    tmp_path, capsys, prefix
) -> None:
    exporter_tree(tmp_path, prefix, "VALUE = 3\n")
    assert (
        load_validator().detect_export_module_prefix(tmp_path, Path(sys.executable))
        == prefix
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert prefix in captured.err


def test_broken_modern_layout_never_falls_back_to_legacy(tmp_path) -> None:
    exporter_tree(
        tmp_path, "gnn.export", "raise RuntimeError('genuine exporter failure')\n"
    )
    exporter_tree(tmp_path, "export", "VALUE = 3\n")
    with pytest.raises(RuntimeError, match="genuine exporter failure"):
        load_validator().detect_export_module_prefix(tmp_path, Path(sys.executable))


def test_failed_interchange_writes_json_failure_without_stdout_noise(tmp_path) -> None:
    output = tmp_path / "receipts" / "interchange.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    receipt = json.loads(output.read_text())
    assert receipt["schema_version"] == "1.0" and receipt["success"] is False
    assert receipt["error"]["type"] == "ValueError"
    original = output.read_bytes()
    duplicate = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert duplicate.returncode != 0 and output.read_bytes() == original


def step7_receipt(tmp_path):
    output = tmp_path / "output"
    directory = output / "7_export_output"
    directory.mkdir(parents=True)
    rows = []
    for name in ("categorical", "gaussian"):
        path = directory / (name + ".json")
        path.write_text("{}")
        rows.append(
            {
                "file_name": name + ".md",
                "exports": {"geo_infer": {"success": True, "export_file": str(path)}},
            }
        )
    return (
        output,
        directory / "export_results.json",
        {
            "summary": {"successful_exports": 2},
            "files_exported": rows,
        },
    )


def test_numbered_step7_receipt_keeps_the_exact_selected_output_roster(tmp_path):
    output, path, receipt = step7_receipt(tmp_path)
    path.write_text(json.dumps(receipt))
    selected = load_validator().step7_export_paths(output)
    assert set(selected) == {"categorical.md", "gaussian.md"}
    assert all(
        item.is_file() and item.is_relative_to(output) for item in selected.values()
    )


@pytest.mark.parametrize(
    "case",
    [
        "one_success",
        "empty_roster",
        "duplicate_model",
        "unknown_model",
        "extra_format",
        "failed_export",
        "missing_path",
        "escaped_path",
        "directory_path",
        "symlink_path",
    ],
)
def test_numbered_step7_refuses_incomplete_or_uncontained_outputs(tmp_path, case):
    output, path, receipt = step7_receipt(tmp_path)
    rows = receipt["files_exported"]
    export = rows[0]["exports"]["geo_infer"]
    if case == "one_success":
        receipt["summary"]["successful_exports"] = 1
    elif case == "empty_roster":
        receipt["files_exported"] = []
    elif case == "duplicate_model":
        rows[1]["file_name"] = rows[0]["file_name"]
    elif case == "unknown_model":
        rows[0]["file_name"] = "unselected.md"
    elif case == "extra_format":
        rows[0]["exports"]["json"] = {"success": True}
    elif case == "failed_export":
        export["success"] = False
    elif case == "missing_path":
        del export["export_file"]
    elif case == "escaped_path":
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        export["export_file"] = str(outside)
    elif case == "directory_path":
        export["export_file"] = str(output)
    elif case == "symlink_path":
        alias = output / "alias.json"
        alias.symlink_to(Path(export["export_file"]))
        export["export_file"] = str(alias)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Step 7"):
        load_validator().step7_export_paths(output)
