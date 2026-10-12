"""Scanner-free custody tests using bounded local producer-shaped fixtures."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from geo_infer_test import research_evidence as evidence

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle(tmp_path):
    """Write the five shapes emitted by the manuscript producer."""
    root = tmp_path / "output"
    stamp = {
        "schema_version": "geo-infer-manuscript-evidence/v1",
        "source_commit": "18188fc49",
        "source_hash": "6a99e23964fd05fe",
    }
    payloads = [
        dict(
            stamp,
            verification_source_commit="abcdef123",
            verification_source_hash="abcdef1234567890",
            verification_measured_elsewhere=True,
            verification_failures=["unit-tests"],
        ),
        {
            "schema_version": stamp["schema_version"],
            "commit": "2fdd67ff4",
            "source_hash": "75403e258fb10483",
            "modules": [],
        },
        dict(
            stamp,
            full_validation_requested=True,
            results=[
                {
                    "name": "unit-tests",
                    "status": "failed",
                    "return_code": 1,
                    "command": "/private/command",
                    "output_tail": str(tmp_path),
                    "receipt": "/private/receipt",
                },
                {"name": "compile", "status": "passed", "return_code": 0},
            ],
        ),
        {
            "RESEARCH_COMMIT": stamp["source_commit"],
            "RESEARCH_SOURCE_HASH": stamp["source_hash"],
            "VERIFICATION_STATUS": "1 passed, 1 failed",
        },
        dict(stamp, schema_version="geo-infer-manuscript-figures/v1", figures=[]),
    ]
    for relative, payload in zip(evidence.SOURCE_PATHS, payloads):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_custody_hashes_actual_bytes_and_retains_failed_full_tier(bundle, tmp_path):
    """Custody success must not become a claim that failed verification passed."""
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "success"
    assert receipt["upload_source_paths"] == list(evidence.SOURCE_PATHS)
    for row in receipt["files"]:
        assert (
            row["sha256"]
            == hashlib.sha256((bundle / row["path"]).read_bytes()).hexdigest()
        )
    verification = receipt["files"][2]["producer_verification"]
    assert verification["full_validation_requested"] is True
    assert verification["has_failure_evidence"] is True
    assert verification["results"][0] == {
        "name": "unit-tests",
        "status": "failed",
        "return_code": 1,
    }
    assert verification["producer_success_established"] is False
    assert receipt["claims"] == {
        "producer_success_established": False,
        "freshness_established": False,
    }
    assert (
        receipt["files"][0]["provenance"]["reported_verification_measured_elsewhere"]
        is True
    )
    assert receipt["files"][1]["provenance"]["reported_source_commit"] == "2fdd67ff4"
    destination = tmp_path / "custody.json"
    evidence.write_research_receipt(receipt, destination, output_root=bundle)
    encoded = destination.read_bytes()
    assert len(encoded) <= evidence.MAX_RECEIPT_BYTES
    assert str(tmp_path).encode() not in encoded
    assert b"/private/" not in encoded
    assert json.loads(encoded) == receipt


def test_missing_one_disables_all_source_uploads_and_cli_retains_receipt(
    bundle, tmp_path
):
    """CLI failures still write the separately retained receipt."""
    (bundle / evidence.SOURCE_PATHS[0]).unlink()
    destination = tmp_path / "receipt.json"
    script = Path(__file__).resolve().parents[2] / "validate_research_evidence.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(bundle),
            "--receipt-path",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 1
    receipt = json.loads(destination.read_text())
    assert receipt["files"][0]["errors"] == ["missing"]
    assert receipt["upload_source_paths"] == []
    assert str(tmp_path) not in completed.stdout + completed.stderr


def test_oversized_source_never_read(bundle, monkeypatch):
    """Oversized individual bytes must never reach the parser or hash."""
    path = bundle / evidence.SOURCE_PATHS[0]
    with path.open("wb") as stream:
        stream.truncate(evidence.MAX_FILE_BYTES + 1)
    original_read = os.read
    target_inode = path.stat().st_ino

    def guarded_read(fd, length):
        assert os.fstat(fd).st_ino != target_inode
        return original_read(fd, length)

    monkeypatch.setattr(evidence.os, "read", guarded_read)
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "failure"
    assert receipt["files"][0]["errors"] == ["file_size_limit"]
    assert "sha256" not in receipt["files"][0]
    assert receipt["upload_source_paths"] == []


def test_total_cap_is_enforced_before_any_read(bundle, monkeypatch):
    """Five individually permitted files can exceed the aggregate budget."""
    for relative in evidence.SOURCE_PATHS:
        with (bundle / relative).open("wb") as stream:
            stream.truncate(evidence.MAX_FILE_BYTES)

    def forbidden_read(*args):
        pytest.fail("aggregate oversize bytes read")

    monkeypatch.setattr(evidence.os, "read", forbidden_read)
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["errors"] == ["total_size_limit"]
    assert receipt["upload_source_paths"] == []


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[]",
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e9999}',
        b'{"a":1,"a":2}',
        b'{"a":"\xff"}',
        b"{} trailing",
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_strict_json_rejects_invalid_or_nonfinite_sources(bundle, raw):
    """No permissive JSON parsing or exception containing source text."""
    (bundle / evidence.SOURCE_PATHS[0]).write_bytes(raw)
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["files"][0]["errors"] == ["invalid_json_or_evidence_shape"]
    assert receipt["outcome"] == "failure"
    assert receipt["upload_source_paths"] == []


@pytest.mark.parametrize("component", ["file", "directory", "root"])
def test_symlink_escape_is_rejected(bundle, tmp_path, component):
    """Reject symlinks at every level, including a symlink within the root."""
    if component == "file":
        path = bundle / evidence.SOURCE_PATHS[0]
        target = tmp_path / "private.json"
        target.write_text('{"private":true}')
        path.unlink()
        path.symlink_to(target)
    elif component == "directory":
        original = bundle / "data"
        target = tmp_path / "data"
        original.rename(target)
        original.symlink_to(target, target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(bundle, target_is_directory=True)
        bundle = alias
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "failure"
    assert receipt["upload_source_paths"] == []
    assert "private" not in json.dumps(receipt)


def test_nonregular_source_rejected_without_read(bundle):
    """A directory is not a retained JSON file."""
    path = bundle / evidence.SOURCE_PATHS[0]
    path.unlink()
    path.mkdir()
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["files"][0]["errors"] == ["not_regular_file"]
    assert receipt["upload_source_paths"] == []


def test_growth_during_read_fails_closed(bundle, monkeypatch):
    """Bound reads to the preflight size and reject concurrent mutation."""
    path = bundle / evidence.SOURCE_PATHS[0]
    target_inode = path.stat().st_ino
    original_read = os.read
    mutated = False

    def growing_read(fd, length):
        nonlocal mutated
        if os.fstat(fd).st_ino == target_inode and not mutated:
            with path.open("ab") as stream:
                stream.write(b" " * 1024)
            mutated = True
        return original_read(fd, length)

    monkeypatch.setattr(evidence.os, "read", growing_read)
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["files"][0]["errors"] == ["changed_during_read"]
    assert receipt["upload_source_paths"] == []


@pytest.mark.parametrize("component", ["root", "data", "figures", "file"])
@pytest.mark.parametrize("replacement", ["copy", "symlink", "missing"])
@pytest.mark.parametrize("read_at", ["first", "last"])
def test_path_replacement_during_read_disables_upload(
    bundle, tmp_path, monkeypatch, component, replacement, read_at
):
    """Current upload names must still identify the held, hashed bundle."""
    trigger = bundle / evidence.SOURCE_PATHS[0 if read_at == "first" else -1]
    trigger_identity = (trigger.stat().st_dev, trigger.stat().st_ino)
    path = (
        bundle
        if component == "root"
        else bundle / evidence.SOURCE_PATHS[0]
        if component == "file"
        else bundle / component
    )
    displaced = tmp_path / "private-displaced-source"
    original_read = os.read
    replaced = False

    def replacing_read(fd, length):
        nonlocal replaced
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == trigger_identity and not replaced:
            path.rename(displaced)
            if replacement == "symlink":
                path.symlink_to(displaced, target_is_directory=component != "file")
            elif replacement == "copy":
                if component == "file":
                    shutil.copyfile(displaced, path)
                    changed_source = path
                else:
                    shutil.copytree(displaced, path)
                    changed_source = (
                        path / evidence.SOURCE_PATHS[0]
                        if component == "root"
                        else path
                        / (
                            "figure_registry.json"
                            if component == "figures"
                            else "research_manifest.json"
                        )
                    )
                changed_source.write_bytes(b'{"private-replacement":true}')
            replaced = True
        return original_read(fd, length)

    monkeypatch.setattr(evidence.os, "read", replacing_read)
    receipt = evidence.validate_research_evidence(bundle)
    assert replaced
    assert receipt["outcome"] == "failure"
    assert receipt["errors"] == ["source_paths_changed"]
    assert receipt["upload_source_paths"] == []
    encoded = json.dumps(receipt)
    assert str(tmp_path) not in encoded
    assert "private" not in encoded


def test_fifo_source_fails_closed_without_read(bundle, monkeypatch):
    """Opening an unconnected FIFO must neither block nor read its contents."""
    path = bundle / evidence.SOURCE_PATHS[0]
    path.unlink()
    os.mkfifo(path)
    target = path.stat()
    original_read = os.read

    def guarded_read(fd, length):
        info = os.fstat(fd)
        assert (info.st_dev, info.st_ino) != (target.st_dev, target.st_ino)
        return original_read(fd, length)

    monkeypatch.setattr(evidence.os, "read", guarded_read)
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "failure"
    assert receipt["files"][0]["errors"] == ["not_regular_file"]
    assert receipt["upload_source_paths"] == []


def test_empty_verification_does_not_assert_producer_success(bundle):
    """An unmeasured producer record can be retained without acceptance claims."""
    path = bundle / evidence.SOURCE_PATHS[2]
    payload = json.loads(path.read_text())
    payload["results"] = []
    payload["full_validation_requested"] = False
    path.write_text(json.dumps(payload))
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "success"
    assert receipt["files"][2]["producer_verification"]["results"] == []
    assert receipt["claims"]["producer_success_established"] is False


def test_unbounded_verification_fails_without_receipt_growth(bundle):
    """Never silently truncate verification tiers to fit a receipt."""
    path = bundle / evidence.SOURCE_PATHS[2]
    payload = json.loads(path.read_text())
    payload["results"] *= evidence.MAX_RESULTS
    path.write_text(json.dumps(payload))
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "failure"
    assert len(json.dumps(receipt).encode()) < evidence.MAX_RECEIPT_BYTES


def test_private_identity_is_not_reflected(bundle):
    """Malformed provenance strings cannot publish a local filesystem path."""
    path = bundle / evidence.SOURCE_PATHS[0]
    payload = json.loads(path.read_text())
    payload["source_commit"] = "/Users/private/repository"
    path.write_text(json.dumps(payload))
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["files"][0]["provenance"]["reported_source_commit"] is None
    assert "/Users/" not in json.dumps(receipt)


@pytest.mark.parametrize("unsafe", ["source", "symlink"])
def test_receipt_cannot_overwrite_source_or_follow_symlink(bundle, tmp_path, unsafe):
    """The receipt writer preserves source files and external destinations."""
    source = bundle / evidence.SOURCE_PATHS[0]
    before = source.read_bytes()
    destination = source
    if unsafe == "symlink":
        destination = tmp_path / "receipt-link.json"
        destination.symlink_to(source)
    with pytest.raises(ValueError):
        evidence.write_research_receipt(
            evidence.validate_research_evidence(bundle), destination, output_root=bundle
        )
    assert source.read_bytes() == before


def test_successful_cli_exit(bundle, tmp_path):
    """Successful exit authorizes only custody upload, even with producer failures."""
    destination = tmp_path / "receipt.json"
    assert (
        evidence.main(
            ["--output-root", str(bundle), "--receipt-path", str(destination)]
        )
        == 0
    )
    assert json.loads(destination.read_text())["files"][2]["producer_verification"][
        "has_failure_evidence"
    ]


def test_maximum_verification_projection_fits_receipt_budget(bundle, tmp_path):
    """The maximum accepted tier inventory fits the independently retained receipt."""
    path = bundle / evidence.SOURCE_PATHS[2]
    payload = json.loads(path.read_text())
    payload["results"] = [
        {"name": "a" * 64, "status": "timed_out", "return_code": -(2**31)}
        for _ in range(evidence.MAX_RESULTS)
    ]
    path.write_text(json.dumps(payload))
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "success"
    destination = tmp_path / "receipt.json"
    evidence.write_research_receipt(receipt, destination, output_root=bundle)
    assert destination.stat().st_size <= evidence.MAX_RECEIPT_BYTES
    assert (
        len(
            json.loads(destination.read_text())["files"][2]["producer_verification"][
                "results"
            ]
        )
        == evidence.MAX_RESULTS
    )


def test_unavailable_root_still_writes_failure_receipt(tmp_path):
    """A missing output directory cannot suppress the custody failure receipt."""
    destination = tmp_path / "receipt.json"
    assert (
        evidence.main(
            [
                "--output-root",
                str(tmp_path / "missing"),
                "--receipt-path",
                str(destination),
            ]
        )
        == 1
    )
    receipt = json.loads(destination.read_text())
    assert receipt["errors"] == ["unsafe_or_unavailable_output_root"]
    assert receipt["upload_source_paths"] == []


def test_receipt_write_error_is_nonzero_and_sanitized(bundle, tmp_path, capsys):
    """Operational receipt failures must not print private destination paths."""
    assert (
        evidence.main(
            [
                "--output-root",
                str(bundle),
                "--receipt-path",
                str(tmp_path / "unavailable" / "receipt.json"),
            ]
        )
        == 1
    )
    assert str(tmp_path) not in capsys.readouterr().out


@pytest.mark.parametrize("status", ["timeout", "failed"])
def test_absent_exit_status_preserves_producer_failure_evidence(bundle, status):
    """Timeouts and failed launches have no observed exit code to invent."""
    path = bundle / "data/research_verification.json"
    payload = json.loads(path.read_text())
    payload["results"][0].update(status=status, return_code=None)
    path.write_text(json.dumps(payload))
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "success"
    verification = receipt["files"][2]["producer_verification"]
    assert verification["results"][0]["return_code"] is None
    assert verification["results"][0]["status"] == status
    assert verification["has_failure_evidence"]
    assert not verification["producer_success_established"]


def test_missing_filesystem_primitives_fail_without_traceback_or_private_paths(
    bundle, tmp_path, monkeypatch, capsys
):
    """Unsupported platforms cannot silently weaken no-follow operations."""
    destination = tmp_path / "receipt.json"
    with monkeypatch.context() as patch:
        patch.delattr(os, "O_DIRECTORY")
        receipt = evidence.validate_research_evidence(bundle)
        assert receipt["errors"] == ["unsupported_filesystem_primitives"]
        assert receipt["upload_source_paths"] == []
        assert (
            evidence.main(
                ["--output-root", str(bundle), "--receipt-path", str(destination)]
            )
            == 1
        )
    assert not destination.exists()
    assert str(tmp_path) not in capsys.readouterr().out


def test_aliased_output_root_cannot_authorize_overwriting_real_source(bundle, tmp_path):
    """A rejected ancestor alias still denotes the same protected source names."""
    alias = tmp_path / "aliased-output"
    alias.symlink_to(bundle, target_is_directory=True)
    source = bundle / evidence.SOURCE_PATHS[0]
    original = source.read_bytes()
    receipt = evidence.validate_research_evidence(alias)
    assert receipt["outcome"] == "failure"
    with pytest.raises(ValueError, match="overlaps source allowlist"):
        evidence.write_research_receipt(receipt, source, output_root=alias)
    assert source.read_bytes() == original


@pytest.mark.parametrize("component", ["data", "figures", "file"])
def test_aliased_source_component_preserves_real_source(bundle, tmp_path, component):
    """Even rejected source aliases must be protected from receipt replacement."""
    relative = evidence.SOURCE_PATHS[-1 if component == "figures" else 0]
    source = bundle / relative
    original = source.read_bytes()
    target = tmp_path / "real-source"
    if component == "file":
        source.rename(target)
        source.symlink_to(target)
        destination = target
    else:
        directory = bundle / component
        directory.rename(target)
        directory.symlink_to(target, target_is_directory=True)
        destination = target / source.name
    receipt = evidence.validate_research_evidence(bundle)
    assert receipt["outcome"] == "failure"
    with pytest.raises(ValueError, match="overlaps source allowlist"):
        evidence.write_research_receipt(receipt, destination, output_root=bundle)
    assert destination.read_bytes() == original
