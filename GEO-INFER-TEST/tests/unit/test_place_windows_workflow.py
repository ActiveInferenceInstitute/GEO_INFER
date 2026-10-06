"""Structural acceptance for the dedicated native Windows PLACE lane.

These checks validate orchestration and custody structure. The native lane runs
the real minimal-profile negative control and both full acceptance files.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
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


def embedded_python(entry):
    if entry.get("shell") == "python":
        return entry["run"]
    assert entry["shell"] == "pwsh"
    assert "'@ | .venv\\Scripts\\python.exe -I -" in entry["run"]
    assert "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }" in entry["run"]
    return entry["run"].split("@'\n", 1)[1].split("\n'@", 1)[0]


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
    for required in (
        "importlib.util.find_spec(module) is None",
        "geo-infer-data",
        "sqlalchemy",
        "optional_absence",
        "assert absent",
    ):
        assert required in isolated
    assert "finally:" in isolated and '"runtime.json"' in isolated


def test_acceptance_uses_both_whole_files_and_fresh_canonical_runner():
    acceptance = step("Run canonical worker, batch and exact offline replay acceptance")
    code = embedded_python(acceptance)
    tree = ast.parse(code)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    runs = [
        n
        for n in calls
        if isinstance(n.func, ast.Attribute)
        and ast.unparse(n.func) == "execution.run_command"
    ]
    assert len(runs) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in runs[0].keywords}
    assert keywords == {
        "timeout": "300",
        "cwd": "root",
        "env_overrides": "{'PYTHONPATH': str(evidence)}",
    }
    assert "execution.pytest_base_args()" in ast.unparse(runs[0].args[0])
    assert "--junitxml=" in code and "execution.run_results_dir()" in code
    paths = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "paths"
            for target in n.targets
        )
    )
    assert [n.right.value for n in paths.value.elts] == [
        "GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py",
        "GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py",
    ]
    for required in (
        'execution.RESULTS_DIR = evidence / "receipts"',
        "execution.PROJECT_ROOT.resolve() == root",
        "execution.SuiteReport()",
        "report.add(execution.run_command(",
        "execution.write_summary(report, show_failures=True)",
        "raise SystemExit(0 if report.success else 1)",
    ):
        assert required in code
    assert "-I" not in ast.unparse(runs[0].args[0])
    assert "--collect-only" not in code and "-k" not in code and "-n" not in code
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
        "deselected",
        'selection["collected"]',
        'selection["selected"]',
        "skipped",
        "isolation-negative/expected-failure.json",
        "test_captured_sources_rebuild_exact_layer_bytes",
        "test_parent_deadline_stops_and_reaps_real_worker[drip]",
        "test_parent_deadline_stops_and_reaps_real_worker[headers]",
        "test_batch_deadline_preserves_existing_artifacts_after_prior_download",
    ):
        assert required in custody


def test_embedded_python_and_observer_parse_without_executing_imports_or_processes():
    for entry in workflow()["jobs"]["place-download"]["steps"]:
        if entry.get("shell") in {"python", "pwsh"}:
            ast.parse(embedded_python(entry))
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
    preparation = step("Prepare loopback guard and real worker observations")["run"]
    assert "PYTEST_PLUGINS=pytest_asyncio.plugin,place_windows_observer" in preparation
    assert "PYTHONPATH={evidence}" in preparation
    assert "os.name =" not in observer and "killpg =" not in observer


def test_native_negative_control_uses_real_builder_and_separate_failed_receipts():
    """The exact missing-dependency reproduction belongs to the minimal Windows profile."""
    control = step("Require canonical optional sibling poisoning negative control")
    code = embedded_python(control)
    names = [s.get("name") for s in workflow()["jobs"]["place-download"]["steps"]]
    assert names.index(control["name"]) < names.index(
        "Run canonical worker, batch and exact offline replay acceptance"
    )
    assert control["timeout-minutes"] == 2
    assert (
        control["env"]
        == step("Run canonical worker, batch and exact offline replay acceptance")[
            "env"
        ]
    )
    tree = ast.parse(code)
    runs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "execution.run_command"
    ]
    assert len(runs) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in runs[0].keywords}
    assert keywords == {
        "timeout": "60",
        "cwd": "root",
        "env_overrides": "{'PLACE_EVIDENCE': str(negative)}",
    }
    for required in (
        "importlib.metadata.distribution(distribution)",
        "importlib.metadata.PackageNotFoundError",
        "importlib.util.find_spec(module) is None",
        '(("geo-infer-data", "geo_infer_data"),',
        '("sqlalchemy", "sqlalchemy"))',
        "execution.build_subprocess_env()",
        'root / "GEO-INFER-DATA/src" in builder_paths',
        "assert evidence in builder_paths",
        'execution.RESULTS_DIR = negative / "receipts"',
        "execution.pytest_base_args()",
        "--junitxml=",
        "timeout=60",
        "execution.write_summary(report, show_failures=True)",
        'result.status == "FAIL"',
        "result.returncode not in (None, 0)",
        "result.executed == 0",
        "Error importing plugin",
        "place_windows_observer",
        "No module named 'sqlalchemy'",
        "geo_infer_data",
        "module_bridge.py",
        "GEO-INFER-DATA/src/geo_infer_data/core/pipeline.py",
        'assert not receipt["success"]',
        'receipt["artifacts"][name]',
        'assert not (negative / "workers.jsonl").exists()',
        'receipt["custody_complete"]',
        'receipt["lock_sha256"]',
        '"expected-failure.json"',
        '"preconditions.json"',
    ):
        assert required in code
    paths = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "paths"
            for target in n.targets
        )
    )
    assert [n.right.value for n in paths.value.elts] == [
        "GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py",
        "GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py",
    ]
    # No overriding PYTHONPATH, simulated imports, reduced selection, or direct subprocess bypass.
    assert "--collect-only" not in code and "subprocess.run" not in code
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "pytest"
        and n.func.attr in {"skip", "xfail"}
        for n in ast.walk(tree)
    )


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
        "missing_negative_control",
        "deselected",
        "inventory_loss",
        "missing_worker_file",
        "skipped",
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
    if failure != "missing_negative_control":
        negative = evidence / "isolation-negative"
        negative.mkdir()
        (negative / "expected-failure.json").write_text(
            '{"status": "EXPECTED_FAILURE"}'
        )
    (evidence / "candidate.json").write_text(
        json.dumps({"lock_sha256": "lock", "revision": "candidate"})
    )
    receipt = {
        "success": True,
        "status": "PASS",
        "executed": 2,
        "custody_complete": failure != "incomplete_custody",
        "revision": "other" if failure == "revision_mismatch" else "candidate",
        "lock_sha256": "other" if failure == "lock_mismatch" else "lock",
    }
    nodes = [
        "GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py::test_worker_ignores_parent_pythonpath",
        "GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py::test_captured_sources_rebuild_exact_layer_bytes",
    ]
    if failure == "missing_replay":
        nodes[1] = nodes[1].replace(
            "test_captured_sources_rebuild_exact_layer_bytes", "test_other"
        )
    if failure == "missing_worker_file":
        nodes[0] = nodes[0].replace(
            "test_regional_download_worker.py", "test_regional_layer_acquisition.py"
        )
    selected = {
        "unaccounted": ["lost"] if failure == "unaccounted" else [],
        "deselected": ["removed"] if failure == "deselected" else [],
        "collected": nodes + (["lost"] if failure == "inventory_loss" else []),
        "selected": nodes,
        "executed": nodes,
    }
    (attempt / "selection.json").write_text(json.dumps(selected))
    if failure != "missing_junit":
        skipped = '<skipped type="expected-failure"/>' if failure == "skipped" else ""
        (attempt / "junit.xml").write_text(
            f'<testsuites><testsuite><testcase name="worker">{skipped}</testcase>'
            '<testcase name="replay"/></testsuite></testsuites>'
        )
    for name in ("stdout.log", "stderr.log"):
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
