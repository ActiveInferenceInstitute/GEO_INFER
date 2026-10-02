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
