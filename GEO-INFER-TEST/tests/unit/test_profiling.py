"""Real profiler subprocesses, completion custody and finite failure controls."""

import hashlib
import json
import pstats
from pathlib import Path
import sys

import pytest

from geo_infer_test.profiling import profile_command

pytestmark = pytest.mark.unit


def test_tracing_profile_executes_real_target_and_seals_all_artifacts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    script = source / "calculate.py"
    script.write_text(
        'import sys\ndef calculate():\n return sum(i*i for i in range(10000))\nassert sys.argv[1:]==["forwarded"]\nprint(calculate())\n'
    )
    output = tmp_path / "profile"
    result = profile_command(
        script=script, args=["forwarded"], output=output, cwd=source, timeout=30
    )
    assert result["success"], result
    assert result["completion"]["success"] and result["summary"]["total_calls"] > 0
    assert any(
        key[2] == "calculate"
        for key in pstats.Stats(str(output / "profile.pstats")).stats
    )
    assert "333283335000" in (output / "stdout.log").read_text()
    for name, digest in result["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / "receipt.json").read_text()) == result
    assert result["runtime"]["version_info"][:2] == list(sys.version_info[:2])


def test_failed_target_does_not_pass_because_profile_was_written(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text('raise ValueError("real target failure")\n')
    result = profile_command(
        script=script, output=tmp_path / "failure", cwd=tmp_path, timeout=30
    )
    assert not result["success"] and result["status"] == "failed"
    assert Path(result["receipt"]).is_file()
    assert "real target failure" in (tmp_path / "failure/stderr.log").read_text()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("preserved")
    script = tmp_path / "target.py"
    script.write_text('raise AssertionError("must not run")')
    with pytest.raises(FileExistsError):
        profile_command(script=script, output=output)
    assert marker.read_text() == "preserved" and sorted(
        p.name for p in output.iterdir()
    ) == ["marker"]


@pytest.mark.parametrize(
    "options",
    [
        {"engine": "missing"},
        {"timeout": float("inf")},
        {"timeout": True},
        {"sampling_rate": 0},
        {"limit": 201},
        {"args": "not-an-array"},
        {"env": {"key": 3}},
        {"engine": "tracing", "mode": "cpu"},
    ],
)
def test_invalid_profile_options_fail_before_output_creation(tmp_path, options):
    script = tmp_path / "target.py"
    script.write_text('print("unused")')
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        profile_command(script=script, output=output, **options)
    assert not output.exists()


def test_real_timeout_retains_failed_receipt(tmp_path):
    script = tmp_path / "wait.py"
    script.write_text("import time;time.sleep(30)")
    result = profile_command(
        script=script, output=tmp_path / "timed", cwd=tmp_path, timeout=5
    )
    assert not result["success"] and any(
        "Timeout" in reason for reason in result["failures"]
    )
    assert Path(result["receipt"]).is_file()


def test_git_custody_uses_checkout_root_when_target_cwd_is_nested(tmp_path):
    import subprocess

    root = tmp_path / "repository"
    child = root / "nested"
    child.mkdir(parents=True)
    script = child / "calculate.py"
    script.write_text("print(sum(range(100)))\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=10)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, timeout=10)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "target",
        ],
        check=True,
        timeout=10,
    )
    script.write_text(script.read_text() + "# dirty source must be hashed\n")
    result = profile_command(
        script=script, cwd=child, output=tmp_path / "nested-profile", timeout=30
    )
    assert result["success"] and result["source_bound"] and result["source_unchanged"]
    assert result["source_root"] == str(root.resolve())
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    assert result["source_before"]["dirty_sha256"]["nested/calculate.py"] == expected
    assert result["source_after"]["dirty_sha256"]["nested/calculate.py"] == expected
