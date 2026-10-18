"""Real profiler subprocesses, completion custody and finite failure controls."""

import hashlib
import json
import os
import pstats
from pathlib import Path
import sys

import pytest

from geo_infer_test.process import run_process
from geo_infer_test.profiling import (
    _sampling_summary,
    main,
    profile_command,
    summarize_profile,
)

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
    assert result["process_evidence"]["target_returncode"] == result["returncode"]
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


@pytest.fixture
def saved_profile(tmp_path):
    script = tmp_path / "calculate.py"
    script.write_text(
        "from pathlib import Path\n"
        "with Path('executions').open('a') as output: output.write('one\\n')\n"
        "def calculate():\n"
        " result=0\n"
        " for value in range(50000): result+=value*value\n"
        " return result\n"
        "def wrapper(): return sum(calculate() for _ in range(3))\n"
        "print(wrapper())\n"
    )
    result = profile_command(
        script=script, cwd=tmp_path, output=tmp_path / "profile", timeout=30
    )
    assert result["success"], result
    return Path(result["receipt"])


def test_saved_report_verifies_native_call_graph_without_rerunning_or_writing(
    saved_profile,
):
    folder = saved_profile.parent
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    report = summarize_profile(
        saved_profile,
        limit=200,
        sort_by="own",
        expected_receipt_sha256=hashlib.sha256(before["receipt.json"]).hexdigest(),
    )
    assert report["success"] and report["status"] == "passed"
    assert (
        report["receipt_sha256"] == hashlib.sha256(before["receipt.json"]).hexdigest()
    )
    assert report["artifacts_verified"] == len(before) - 1
    assert report["summary"]["ranking"] == "own"
    rows = report["summary"]["hotspots"]
    assert [r["own_seconds"] for r in rows] == sorted(
        (r["own_seconds"] for r in rows), reverse=True
    )
    calculate = next(r for r in rows if r["function"] == "calculate")
    assert calculate["calls"] == 3
    assert calculate["own_percent"] == pytest.approx(
        100 * calculate["own_seconds"] / report["summary"]["instrumented_seconds"]
    )
    edges = report["summary"]["call_graph"]["edges"]
    assert any(
        edge["callee"]["function"] == "calculate" and edge["calls"] == 3
        for edge in edges
    )
    bounded = summarize_profile(folder, limit=2, sort_by="calls")
    assert len(bounded["summary"]["hotspots"]) == 2
    assert len(bounded["summary"]["call_graph"]["edges"]) <= 2
    assert (folder.parent / "executions").read_text() == "one\n"
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("artifact", ["profile.pstats", "target.json", "stdout.log"])
def test_report_rejects_any_changed_artifact_before_profile_parsing(
    saved_profile, artifact
):
    path = saved_profile.parent / artifact
    # Invalid marshal bytes specifically prove the integrity failure precedes
    # native profile parsing. Ancillary artifact changes also invalidate it.
    path.write_bytes(b"changed and not a native profile")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        summarize_profile(saved_profile)


def test_report_rejects_artifacts_outside_its_own_receipt_directory(saved_profile):
    data = json.loads(saved_profile.read_text())
    outside = saved_profile.parent.parent / "outside.pstats"
    outside.write_bytes((saved_profile.parent / "profile.pstats").read_bytes())
    data["artifacts"]["../outside.pstats"] = hashlib.sha256(
        outside.read_bytes()
    ).hexdigest()
    saved_profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe|Unsealed"):
        summarize_profile(saved_profile)


def test_report_receipt_hash_detects_manifest_edits(saved_profile):
    original = hashlib.sha256(saved_profile.read_bytes()).hexdigest()
    payload = json.loads(saved_profile.read_text())
    payload["runtime"]["platform"] = "changed platform metadata"
    saved_profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independently retained hash"):
        summarize_profile(saved_profile, expected_receipt_sha256=original)


def test_report_rejects_completion_and_recomputed_digest_disagreement(saved_profile):
    payload = json.loads(saved_profile.read_text())
    path = saved_profile.parent / "completion.json"
    completion = json.loads(path.read_text())
    completion["token"] = "different target"
    path.write_text(json.dumps(completion))
    payload["artifacts"]["completion.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    saved_profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Sealed completion"):
        summarize_profile(saved_profile)


def test_report_rejects_missing_successful_completion_artifact(saved_profile):
    payload = json.loads(saved_profile.read_text())
    del payload["artifacts"]["completion.json"]
    (saved_profile.parent / "completion.json").unlink()
    saved_profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lacks sealed target/completion"):
        summarize_profile(saved_profile)


def test_report_rejects_inconsistent_source_custody_claim(saved_profile):
    payload = json.loads(saved_profile.read_text())
    assert payload["source_bound"] is False
    payload["source_bound"] = True
    saved_profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source custody"):
        summarize_profile(saved_profile)


def test_report_rejects_symlinked_profile_even_with_matching_digest(saved_profile):
    profile = saved_profile.parent / "profile.pstats"
    outside = saved_profile.parent.parent / "outside.pstats"
    profile.replace(outside)
    profile.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe profile artifact"):
        summarize_profile(saved_profile)


def test_sampling_report_aggregates_collapsed_labels_and_weight_shares(tmp_path):
    # Native collapsed output uses basenames, so distinct source frames can
    # serialize to the same label. Count unique labels, retaining all weight.
    artifact = tmp_path / "profile.collapsed"
    artifact.write_text(
        "tid:7;worker.py:calculate:10 2\ntid:7;worker.py:calculate:10 3\ntid:8;worker.py:wait:20 2\n"
    )
    summary = _sampling_summary(artifact, 1)
    assert summary["thread_samples"] == 7 and summary["unique_stacks"] == 2
    assert summary["hotspots"] == [
        dict(
            stack="tid:7;worker.py:calculate:10",
            thread_samples=5,
            sample_percent=100 * 5 / 7,
        )
    ]
    assert "not function call counts" in summary["measurement"]


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": True},
        {"limit": 201},
        {"max_artifact_bytes": 0},
        {"sort_by": "samples"},
    ],
)
def test_report_rejects_invalid_limits_and_engine_rankings(saved_profile, options):
    with pytest.raises(ValueError):
        summarize_profile(saved_profile, **options)


def test_report_cli_preserves_profile_and_rejects_execution_options(
    saved_profile, capsys
):
    assert (
        main(["--report", str(saved_profile), "--sort-by", "calls", "--limit", "3"])
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["ranking"] == "calls"
    assert len(output["summary"]["hotspots"]) == 3
    with pytest.raises(SystemExit) as rejected:
        main(
            [
                "--report",
                str(saved_profile),
                "--output",
                str(saved_profile.parent / "unused"),
            ]
        )
    assert rejected.value.code == 2
    assert not (saved_profile.parent / "unused").exists()
    assert (saved_profile.parent.parent / "executions").read_text() == "one\n"


def test_report_keeps_actual_failed_target_failed(tmp_path):
    target = tmp_path / "fail.py"
    target.write_text("raise ValueError('retained target failure')\n")
    original = profile_command(
        script=target, cwd=tmp_path, output=tmp_path / "failed", timeout=30
    )
    assert original["success"] is False
    report = summarize_profile(original["receipt"])
    assert report["success"] is False and report["status"] == "failed"
    assert report["failures"] == original["failures"]
    assert report["summary"]["total_calls"] > 0
    assert main(["--report", original["receipt"]]) == 1


def test_native_controller_interruption_drains_children_and_seals_failed_receipt(
    tmp_path,
):
    target = tmp_path / "wait.py"
    target.write_text(
        "import json,os,time\nfrom pathlib import Path\n"
        "Path('started.tmp').write_text(json.dumps({'pid':os.getpid()}))\n"
        "Path('started.tmp').replace('started.json')\n"
        "print('owned target started',flush=True)\ntime.sleep(30)\n"
    )
    controller = tmp_path / "controller.py"
    controller.write_text(
        "import _thread,hashlib,json,threading,time\n"
        "from pathlib import Path\nimport psutil\n"
        "from geo_infer_test.profiling import profile_command\n"
        "def interrupt_started():\n"
        " deadline=time.monotonic()+15\n"
        " while not Path('started.json').is_file():\n"
        "  if time.monotonic()>deadline: return\n"
        "  time.sleep(.01)\n"
        " _thread.interrupt_main()\n"
        "thread=threading.Thread(target=interrupt_started,daemon=True)\nthread.start()\n"
        "try:\n profile_command(script='wait.py',cwd=Path.cwd(),output='profile',timeout=20)\n"
        "except KeyboardInterrupt:\n"
        " thread.join(timeout=5)\n assert not thread.is_alive()\n"
        " record=json.loads(Path('profile/receipt.json').read_text())\n"
        " assert record['success'] is False and record['status']=='interrupted',record\n"
        " assert any('KeyboardInterrupt' in item for item in record['failures'])\n"
        " assert record['process_evidence']['cleanup_seconds']>=0\n"
        " assert 'owned target started' in Path('profile/stdout.log').read_text()\n"
        " assert not psutil.pid_exists(json.loads(Path('started.json').read_text())['pid'])\n"
        " for name,digest in record['artifacts'].items():\n"
        "  assert hashlib.sha256((Path('profile')/name).read_bytes()).hexdigest()==digest\n"
        " print('interruption propagated after owned cleanup and failed receipt')\n"
        "else:\n raise AssertionError('native interruption was swallowed')\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    result = run_process(
        [sys.executable, str(controller)], cwd=tmp_path, env=environment, timeout=40
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "interruption propagated after owned cleanup and failed receipt"
        in result.stdout
    )
