"""Structural acceptance for the dedicated native Windows PLACE lane.

These checks do not run PLACE processes or establish native Windows behavior.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/place-download-windows.yml"


def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def step(name):
    return next(
        s
        for s in workflow()["jobs"]["place-download"]["steps"]
        if s.get("name") == name
    )


def test_native_matrix_triggers_and_bounded_readonly_execution():
    document = workflow()
    triggers = document.get("on", document.get(True))
    assert set(triggers) == {"workflow_dispatch", "push", "pull_request"}
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main", "develop"]
    assert document["permissions"] == {"contents": "read"}
    job = document["jobs"]["place-download"]
    assert job["runs-on"] == "windows-latest"
    assert job["strategy"] == {
        "fail-fast": False,
        "matrix": {"python": ["3.11", "3.12"]},
    }
    assert job["timeout-minutes"] == 25
    assert job["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    for pool in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert job["env"][pool] == "1"
    for entry in job["steps"]:
        if "uses" in entry:
            assert re.fullmatch(r"[\w/-]+@[0-9a-f]{40}", entry["uses"])
        assert "continue-on-error" not in entry
    checkout = job["steps"][0]
    assert checkout["with"]["persist-credentials"] is False
    assert (
        checkout["with"]["ref"]
        == "${{ github.event.pull_request.head.sha || github.sha }}"
    )


def test_profile_is_locked_selected_place_with_declared_test_tooling():
    sync = step("Install the declared locked PLACE acceptance profile")["run"].split()
    assert sync == [
        "uv",
        "sync",
        "--locked",
        "--package",
        "geo-infer-place",
        "--extra",
        "dev",
        "--extra",
        "integrations",
        "--no-default-groups",
    ]
    assert (
        step("Install the declared locked PLACE acceptance profile")["timeout-minutes"]
        == 15
    )
    isolated = step("Verify isolated imports and retain the installed inventory")["run"]
    assert ".venv\\Scripts\\python.exe -I -" in isolated
    for module in (
        "geo_infer_place.core.regional_layers",
        "geo_infer_place.core._regional_download_worker",
        "geo_infer_place.core.bioregion_visualization",
        "geo_infer_test.execution",
        "geo_infer_test.selection",
        "geo_infer_test.testing",
        "requests",
        "psutil",
    ):
        assert f'"{module}"' in isolated
    assert "origin.is_relative_to" in isolated
    assert "importlib.metadata.distributions()" in isolated
    assert "finally:" in isolated and '"runtime.json"' in isolated


def test_acceptance_uses_both_whole_files_and_fresh_canonical_runner():
    acceptance = step("Run canonical worker, batch and exact offline replay acceptance")
    assert acceptance["run"].split() == [
        ".venv\\Scripts\\python.exe",
        "GEO-INFER-TEST/run_unified_tests.py",
        "--paths",
        "GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py",
        "GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py",
        "--workers",
        "1",
        "--timeout",
        "300",
        "--show-failures",
        "--results-dir",
        "place-windows-evidence/receipts",
    ]
    assert acceptance["timeout-minutes"] == 7
    assert acceptance["env"] == {
        "NO_PROXY": "127.0.0.1,::1",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
    }
    custody = step("Require fresh canonical custody and real worker evidence")["run"]
    for required in (
        "junit.xml",
        "selection.json",
        "stdout.log",
        "stderr.log",
        "lock_sha256",
        "unaccounted",
        "test_captured_sources_rebuild_exact_layer_bytes",
        "test_parent_deadline_stops_and_reaps_real_worker[drip]",
        "test_parent_deadline_stops_and_reaps_real_worker[headers]",
        "test_batch_deadline_preserves_existing_artifacts_after_prior_download",
    ):
        assert required in custody


def test_embedded_python_and_observer_parse_without_executing_imports_or_processes():
    for entry in workflow()["jobs"]["place-download"]["steps"]:
        if entry.get("shell") == "python":
            ast.parse(entry["run"])
    powershell = step("Verify isolated imports and retain the installed inventory")[
        "run"
    ]
    ast.parse(powershell.split("@'\n", 1)[1].split("\n'@", 1)[0])
    tree = ast.parse(step("Prepare loopback guard and real worker observations")["run"])
    plugin = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "plugin"
            for target in node.targets
        )
    )
    observer = textwrap.dedent(plugin)
    ast.parse(observer)
    assert "ipaddress.ip_address(parts.hostname).is_loopback" in observer
    assert "loopback(request.url)" in observer and "loopback(args[3])" in observer
    assert "process = ORIGINAL_POPEN(args, *positional, **kwargs)" in observer
    assert "result = ORIGINAL_DOWNLOAD(*args, **kwargs)" in observer
    for field in (
        "test",
        "start_monotonic",
        "elapsed_seconds",
        "command",
        "pid",
        "returncode",
        "stdout_closed",
        "stderr_closed",
        "output_bytes",
        "output_sha256",
    ):
        assert field in observer
    assert "os.name =" not in observer and "killpg =" not in observer


def test_declared_plugins_collect_real_place_inventory_with_strict_root_config(
    tmp_path,
):
    """Exercise the workflow's actual plugin configuration without worker bodies."""
    tree = ast.parse(step("Prepare loopback guard and real worker observations")["run"])
    plugin = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "plugin"
            for target in node.targets
        )
    )
    (tmp_path / "place_windows_observer.py").write_text(textwrap.dedent(plugin))
    assignment = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("PYTEST_PLUGINS=")
    )
    environment = dict(
        os.environ,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS=assignment.split("=", 1)[1].strip(),
        PYTEST_ADDOPTS="",
        PYTHONPATH=str(tmp_path),
        PLACE_EVIDENCE=str(tmp_path),
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "geo_infer_test.selection",
        "-c",
        str(ROOT / "pyproject.toml"),
        "-p",
        "no:cacheprovider",
        "--collect-only",
        str(
            ROOT / "GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py"
        ),
        str(
            ROOT
            / "GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py"
        ),
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_parent_deadline_stops_and_reaps_real_worker[drip]" in result.stdout
    assert "test_captured_sources_rebuild_exact_layer_bytes" in result.stdout
    assert not (tmp_path / "workers.jsonl").exists()


def test_failure_evidence_upload_is_job_owned_with_lock_and_byte_inventory():
    steps = workflow()["jobs"]["place-download"]["steps"]
    upload = steps[-1]
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "place-windows-evidence/"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["retention-days"] == 14
    assert (
        "${{ matrix.python }}-${{ github.run_id }}-${{ github.run_attempt }}"
        in upload["with"]["name"]
    )
    seal = step("Seal retained evidence and recheck lock custody")
    assert seal["if"] == "always()"
    assert "artifact-inventory.json" in seal["run"] and "hashlib.sha256" in seal["run"]
    assert "assert current == candidate" in seal["run"]
    candidate = step("Bind platform, candidate, lock and source inventory")["run"]
    assert "exist_ok=False" in candidate
    for key in (
        "candidate.json",
        "inventory",
        "revision",
        "lock_sha256",
        "GITHUB_RUN_ATTEMPT",
    ):
        assert key in candidate


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "missing_junit",
        "lock_mismatch",
        "unaccounted",
        "unreaped",
        "missing_drip",
        "missing_replay",
        "duplicate_attempt",
        "revision_mismatch",
        "incomplete_custody",
        "altered_artifact",
    ],
)
def test_custody_gate_accepts_canonical_layout_and_rejects_incomplete_evidence(
    tmp_path, monkeypatch, failure
):
    """Exercise the upload prerequisite against finite files, without spawning processes."""
    evidence = tmp_path / "evidence"
    attempt = evidence / "receipts/runs/run-id/attempts/attempt-id"
    attempt.mkdir(parents=True)
    monkeypatch.setenv("PLACE_EVIDENCE", str(evidence))
    (evidence / "candidate.json").write_text(
        json.dumps({"lock_sha256": "lock", "revision": "candidate"})
    )
    receipt = {
        "success": True,
        "status": "PASS",
        "executed": 1,
        "custody_complete": failure != "incomplete_custody",
        "revision": "other" if failure == "revision_mismatch" else "candidate",
        "lock_sha256": "other" if failure == "lock_mismatch" else "lock",
    }
    selected = {
        "unaccounted": ["lost"] if failure == "unaccounted" else [],
        "executed": []
        if failure == "missing_replay"
        else [
            "test_regional_layer_acquisition.py::test_captured_sources_rebuild_exact_layer_bytes"
        ],
    }
    (attempt / "selection.json").write_text(json.dumps(selected))
    for name in ("junit.xml", "stdout.log", "stderr.log"):
        if failure != "missing_junit" or name != "junit.xml":
            (attempt / name).write_text("retained")
    receipt["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in attempt.iterdir()
        if p.is_file()
    }
    (attempt / "receipt.json").write_text(json.dumps(receipt))
    if failure == "altered_artifact":
        (attempt / "stdout.log").write_text("changed after receipt")
    cases = [
        "test_parent_deadline_stops_and_reaps_real_worker[drip]",
        "test_parent_deadline_stops_and_reaps_real_worker[headers]",
        "test_batch_deadline_preserves_existing_artifacts_after_prior_download",
    ]
    events = [
        {
            "test": case,
            "workers": [
                {
                    "returncode": None if failure == "unreaped" else 0,
                    "stdout_closed": True,
                    "stderr_closed": True,
                }
            ],
        }
        for case in cases
        if failure != "missing_drip" or "[drip]" not in case
    ]
    (evidence / "workers.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    if failure == "duplicate_attempt":
        second = attempt.parent / "second"
        second.mkdir()
        (second / "receipt.json").write_text(json.dumps(receipt))
    code = compile(
        step("Require fresh canonical custody and real worker evidence")["run"],
        "workflow-custody",
        "exec",
    )
    if failure == "none":
        exec(code, {})
    else:
        with pytest.raises(AssertionError):
            exec(code, {})
