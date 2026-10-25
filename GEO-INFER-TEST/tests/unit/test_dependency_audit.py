"""Fail-closed universal dependency-audit inventory and tool-receipt contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from geo_infer_test import dependency_audit as audit

pytestmark = pytest.mark.unit


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A two-member resolved fixture includes mutually exclusive version forks."""
    (tmp_path / "GEO-INFER-TEST").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="geo-infer-framework"\nversion="0.4.0"\n'
    )
    (tmp_path / "GEO-INFER-TEST/pyproject.toml").write_text(
        '[project]\nname="geo-infer-test"\nversion="0.4.0"\n'
    )
    (tmp_path / "uv.toml").write_text('required-version="==0.13.1"\n')
    (tmp_path / ".python-version").write_text("3.12\n")
    (tmp_path / "uv.lock").write_text(
        "version=1\n"
        '[[package]]\nname="geo-infer-framework"\nversion="0.4.0"\nsource={virtual="."}\n'
        '[[package]]\nname="geo-infer-test"\nversion="0.4.0"\nsource={editable="GEO-INFER-TEST"}\n'
        '[[package]]\nname="example"\nversion="1.0"\nsource={registry="https://pypi.org/simple"}\n'
        "resolution-markers=[\"python_version < '3.12'\"]\n"
        '[[package]]\nname="example"\nversion="2.0"\nsource={registry="https://pypi.org/simple"}\n'
        "resolution-markers=[\"python_version >= '3.12'\"]\n"
    )
    return tmp_path


def test_inventory_keeps_every_platform_version_and_authenticates_locals(workspace):
    """Neither the host marker nor a local editable hides a registry version."""
    pairs, local, hashes = audit.locked_inventory(workspace)
    assert pairs == [("example", "1.0"), ("example", "2.0")]
    assert len(local) == 2
    assert "uv.lock" in hashes and "GEO-INFER-TEST/pyproject.toml" in hashes
    batches = audit.requirement_batches(pairs + [("other", "3.0")])
    assert len(batches) == 2
    assert sorted(pair for batch in batches for pair in batch) == sorted(
        pairs + [("other", "3.0")]
    )


@pytest.mark.parametrize(
    "old,new",
    [
        (
            'registry="https://pypi.org/simple"',
            'registry="https://private.invalid/simple"',
        ),
        ('editable="GEO-INFER-TEST"', 'editable="../unreviewed"'),
        ('name="example"', 'name="new\\noption"'),
    ],
)
def test_inventory_refuses_unsupported_or_forged_sources(workspace, old, new):
    """An unsupported source never becomes an excluded or missing audit row."""
    lock = workspace / "uv.lock"
    lock.write_text(lock.read_text().replace(old, new))
    with pytest.raises(ValueError):
        audit.locked_inventory(workspace)


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        [{"name": "other", "version": "1.0", "vulns": []}],
        [{"name": "example", "version": "1.0", "vulns": []}] * 2,
        [
            {
                "name": "example",
                "version": "1.0",
                "vulns": [],
                "skip_reason": "not on PyPI",
            }
        ],
        [{"name": "example", "version": "1.0"}],
    ],
)
def test_receipt_refuses_missing_extra_duplicate_skipped_and_unknown(dependencies):
    """A nominal zero exit cannot certify an incomplete package roster."""
    raw = json.dumps({"dependencies": dependencies}).encode()
    with pytest.raises(ValueError):
        audit.validate_response(raw, [("example", "1.0")], 0)


def test_receipt_retains_unfixed_advisory_without_ignore_or_exit_waiver():
    """Every advisory remains a finding even when no fixed version is available."""
    raw = json.dumps(
        {
            "dependencies": [
                {
                    "name": "Example",
                    "version": "1.0",
                    "vulns": [{"id": "PYSEC-fixture", "fix_versions": []}],
                }
            ]
        }
    ).encode()
    findings = audit.validate_response(raw, [("example", "1.0")], 1)
    assert findings[0]["id"] == "PYSEC-fixture"
    assert findings[0]["fix_versions"] == []
    with pytest.raises(ValueError):
        audit.validate_response(raw, [("example", "1.0")], 0)


@pytest.mark.parametrize(
    "raw",
    [
        b"not JSON",
        b'{"dependencies":[],"dependencies":[]}',
        b'{"error":"network unavailable"}',
    ],
)
def test_receipt_refuses_malformed_or_error_payload(raw):
    """Malformed transport data cannot be interpreted as no vulnerabilities."""
    with pytest.raises(ValueError):
        audit.validate_response(raw, [("example", "1.0")], 0)


@pytest.mark.parametrize("final_custody_timeout", [False, True])
def test_network_tool_failure_retains_evidence_and_fails(
    workspace, monkeypatch, final_custody_timeout
):
    """Real failing children and final custody timeouts always retain evidence."""
    custody_calls = []

    def receipt(**kwargs):
        custody_calls.append(kwargs["timeout"])
        assert 0 < kwargs["timeout"] <= 30
        if final_custody_timeout and len(custody_calls) == 2:
            raise subprocess.TimeoutExpired(
                ["receipt source hashing"], kwargs["timeout"]
            )
        return {"custody_complete": True, "revision": "fixture"}

    monkeypatch.setattr(audit, "runtime_receipt", receipt)
    tool = workspace / "audit-tool.py"
    tool.write_text(
        "import sys\n"
        "if '--version' in sys.argv:\n"
        " print('uv 0.13.1 fixture' if 'uv-tool' in sys.argv[0] else 'pip-audit 2.10.1')\n"
        "elif 'lock' not in sys.argv:\n"
        " print('network unavailable',file=sys.stderr);sys.exit(2)\n"
    )
    # Executable wrappers exercise the shared owned-process harness.
    uv = workspace / "uv-tool"
    auditor = workspace / "pip-audit"
    for target in [uv, auditor]:
        target.write_text(f"#!{sys.executable}\n" + tool.read_text())
        target.chmod(0o755)
    result = audit.audit(
        workspace, workspace / "results", uv=str(uv), auditor=str(auditor), timeout=30
    )
    assert result["status"] == "failed" and result["errors"]
    (receipt,) = (workspace / "results").glob("*/receipt.json")
    assert "network unavailable" in (receipt.parent / "audit-00.stderr").read_text()
    assert len(custody_calls) == 2 and custody_calls[1] < custody_calls[0]
    if final_custody_timeout:
        assert any("Final custody unavailable" in error for error in result["errors"])
        assert result.get("source_and_lock_bookends_equal") is not True
    else:
        assert result["source_and_lock_bookends_equal"] is True
    assert result.get("all_package_versions_audited") is not True


def test_missing_controller_dependency_retains_failed_receipt(workspace, monkeypatch):
    """An unavailable controller prerequisite cannot prevent failure evidence."""

    def unavailable(**kwargs):
        raise ModuleNotFoundError("No module named 'psutil'")

    monkeypatch.setattr(audit, "runtime_receipt", unavailable)
    result = audit.audit(
        workspace, workspace / "results", uv="unused", auditor="unused", timeout=30
    )
    assert result["status"] == "failed"
    assert result.get("all_package_versions_audited") is not True
    (receipt,) = (workspace / "results").glob("*/receipt.json")
    saved = json.loads(receipt.read_text())
    assert any("No module named 'psutil'" in error for error in saved["errors"])
    assert any("Final custody unavailable" in error for error in saved["errors"])


def test_exhausted_deadline_does_not_start_commands_or_final_custody(
    workspace, monkeypatch
):
    """A consumed attempt budget never starts another Git or advisory process."""
    custody_calls = []

    def delayed_receipt(**kwargs):
        custody_calls.append(kwargs["timeout"])
        assert 0 < kwargs["timeout"] <= 0.001
        time.sleep(0.01)
        return {"custody_complete": True, "revision": "fixture"}

    def unexpected_command(*args, **kwargs):
        pytest.fail("A child process started after the attempt deadline")

    monkeypatch.setattr(audit, "runtime_receipt", delayed_receipt)
    monkeypatch.setattr(audit, "run_process", unexpected_command)
    result = audit.audit(
        workspace, workspace / "results", uv="unused", auditor="unused", timeout=0.001
    )
    assert len(custody_calls) == 1
    assert result["status"] == "failed"
    assert result.get("all_package_versions_audited") is not True
    (receipt,) = (workspace / "results").glob("*/receipt.json")
    assert any(
        "deadline exhausted" in error
        for error in json.loads(receipt.read_text())["errors"]
    )
