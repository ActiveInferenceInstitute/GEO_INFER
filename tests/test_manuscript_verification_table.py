"""Regression tests for the published per-group verification table.

Before this table existed the manuscript described its verification record in
prose and left the per-command outcomes inside a JSON file no reader of the PDF
opens, and ``generate`` aborted on the first failing group — which made the
``N passed, M failed`` summary branch unreachable.  These tests pin both: every
defined group gets a row, and a failed group is published rather than fatal.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest
import psutil

from geo_infer_test import execution


@pytest.fixture
def execution_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate receipts and metadata while retaining actual process execution.

    Git source custody has its own real-checkout regression suite. These tests
    exercise actual subprocesses with a small execution budget, so metadata is
    explicitly supplied rather than consuming that budget on another Git scan.
    """
    monkeypatch.setattr(execution, "RESULTS_DIR", tmp_path / ".geo-infer-test-results")
    monkeypatch.setattr(
        execution, "runtime_receipt", lambda **_kwargs: {"custody_complete": True}
    )
    return tmp_path


def _attempt(root: Path, result) -> tuple[Path, dict]:
    """Read a relative receipt and independently bind every retained artifact."""
    assert result.receipt is not None
    assert not Path(result.receipt).is_absolute()
    receipt = root / result.receipt
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    for filename, digest in recorded["artifacts"].items():
        assert (
            hashlib.sha256((receipt.parent / filename).read_bytes()).hexdigest()
            == digest
        )
    assert recorded["cwd"] == str(root)
    return receipt, recorded


def _result(generator: ModuleType, name: str, status: str, code: int):
    return generator.VerificationResult(
        name=name,
        command=f"run {name}",
        status=status,
        return_code=code,
        duration_seconds=1.5,
        output_tail="",
    )


class TestVerificationTable:
    """Every defined command group appears, run or not."""

    def test_empty_record_prints_every_defined_group_as_not_run(
        self, generator: ModuleType
    ) -> None:
        table = generator._verification_table([], full_validation=False)
        rows = [line for line in table.splitlines() if line.startswith("| `")]
        assert len(rows) == len(generator.VERIFICATION_COMMANDS)
        assert all("not run" in row for row in rows)
        for name, _command in generator.VERIFICATION_COMMANDS:
            assert f"| `{name}` |" in table

    def test_full_validation_adds_the_second_tier(self, generator: ModuleType) -> None:
        table = generator._verification_table([], full_validation=True)
        rows = [line for line in table.splitlines() if line.startswith("| `")]
        assert len(rows) == len(generator.VERIFICATION_COMMANDS) + len(
            generator.FULL_VALIDATION_COMMANDS
        )

    def test_a_failed_group_publishes_its_return_code(
        self, generator: ModuleType
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        table = generator._verification_table(
            [_result(generator, name, "failed", 1)], full_validation=False
        )
        row = next(
            line for line in table.splitlines() if line.startswith(f"| `{name}`")
        )
        assert "failed" in row
        assert "| 1 |" in row

    def test_build_variables_publishes_a_table_naming_every_defined_group(
        self, generator: ModuleType, repo_inventory, figure_specs
    ) -> None:
        variables = generator.build_variables(
            repo_inventory,
            figure_specs,
            generator.VerificationRecord.unmeasured(),
        )
        table = variables["VERIFICATION_TABLE"]
        assert table.strip()
        for name, _command in generator.VERIFICATION_COMMANDS:
            assert f"| `{name}` |" in table

    def test_the_token_is_authored_into_the_manuscript(self, repo_root) -> None:
        authored = (
            repo_root / "manuscript" / "04_artifacts_and_evidence.md"
        ).read_text(encoding="utf-8")
        assert "{{VERIFICATION_TABLE}}" in authored

    def test_generate_refuses_a_failed_group_when_publishing(
        self, generator: ModuleType, generatable_checkout: Path, monkeypatch
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        monkeypatch.setattr(
            generator,
            "run_verification",
            lambda _root, **_kwargs: (_result(generator, name, "failed", 1),),
        )
        with pytest.raises(RuntimeError, match="verification record contains"):
            generator.generate(generatable_checkout, verify=True, publication=True)

    def test_a_non_publication_build_publishes_the_failure(
        self, generator: ModuleType, generatable_checkout: Path, monkeypatch
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        monkeypatch.setattr(
            generator,
            "run_verification",
            lambda _root, **_kwargs: (_result(generator, name, "failed", 1),),
        )
        manifest = generator.generate(generatable_checkout, verify=True)
        assert manifest["verification_failures"] == [name]
        variables = json.loads(
            (
                generatable_checkout / "output" / "data" / "manuscript_variables.json"
            ).read_text(encoding="utf-8")
        )
        assert variables["VERIFICATION_FAIL_COUNT"] == "1"
        next(
            line
            for line in variables["VERIFICATION_TABLE"].splitlines()
            if line.startswith(f"| `{name}`")
        )

    def test_summary_reports_mixed_outcomes(self, generator: ModuleType) -> None:
        names = [name for name, _ in generator.VERIFICATION_COMMANDS]
        results = [
            _result(generator, names[0], "passed", 0),
            _result(generator, names[1], "failed", 1),
        ]
        summary, passed, failed, unrun = generator._verification_summary(
            results, full_validation=False
        )
        assert (passed, failed) == (1, 1)
        assert summary == "1 passed, 1 failed"
        assert unrun == len(names) - 2


class TestVerificationRecordReuse:
    """A stored record is reused only when it still describes the tree."""

    def _write(self, generator: ModuleType, root, payload) -> None:
        data = root / "output" / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "research_verification.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )

    def test_missing_record_is_not_reused(
        self, generator: ModuleType, tmp_path, repo_inventory
    ) -> None:
        assert (
            generator.load_matching_verification(
                tmp_path, repo_inventory, full_validation=False
            )
            is None
        )

    def test_matching_record_is_reused(
        self, generator: ModuleType, tmp_path, repo_inventory
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        self._write(
            generator,
            tmp_path,
            {
                "schema_version": generator.RESEARCH_SCHEMA,
                "full_validation_requested": False,
                "source_commit": repo_inventory.commit,
                "source_hash": repo_inventory.source_hash,
                "results": [
                    {
                        "name": name,
                        "command": "run it",
                        "status": "passed",
                        "return_code": 0,
                        "duration_seconds": 1.0,
                        "output_tail": "",
                    }
                ],
            },
        )
        reused = generator.load_matching_verification(
            tmp_path, repo_inventory, full_validation=False
        )
        assert reused is not None
        assert [result.name for result in reused] == [name]

    def test_a_different_source_hash_is_not_reused(
        self, generator: ModuleType, tmp_path, repo_inventory
    ) -> None:
        self._write(
            generator,
            tmp_path,
            {
                "schema_version": generator.RESEARCH_SCHEMA,
                "full_validation_requested": False,
                "source_commit": repo_inventory.commit,
                "source_hash": "0" * 16,
                "results": [
                    {
                        "name": "compile",
                        "command": "run it",
                        "status": "passed",
                        "return_code": 0,
                        "duration_seconds": 1.0,
                        "output_tail": "",
                    }
                ],
            },
        )
        assert (
            generator.load_matching_verification(
                tmp_path, repo_inventory, full_validation=False
            )
            is None
        )

    def test_an_empty_record_is_not_reused(
        self, generator: ModuleType, tmp_path, repo_inventory
    ) -> None:
        self._write(
            generator,
            tmp_path,
            {
                "schema_version": generator.RESEARCH_SCHEMA,
                "full_validation_requested": False,
                "source_commit": repo_inventory.commit,
                "source_hash": repo_inventory.source_hash,
                "results": [],
            },
        )
        assert (
            generator.load_matching_verification(
                tmp_path, repo_inventory, full_validation=False
            )
            is None
        )


class TestVerificationResolutionTier:
    """A build that measures nothing must not discard a record that stands.

    The reuse lookup used to sit inside ``if verify:``.  Every default render —
    the only render the supported pipeline performs — therefore replaced a
    measured record with an empty one and republished ``not run``.  These tests
    drive the decision seam itself rather than asserting on source text.
    """

    def _store(self, generator: ModuleType, root: Path, inventory, results) -> None:
        data = root / "output" / "data"
        data.mkdir(parents=True, exist_ok=True)
        record = generator.VerificationRecord(
            results=tuple(results),
            source_commit=inventory.commit,
            source_hash=inventory.source_hash,
            full_validation_requested=False,
        )
        (data / "research_verification.json").write_text(
            json.dumps(generator._verification_payload(record)), encoding="utf-8"
        )

    def _result(self, generator: ModuleType, name: str):
        return generator.VerificationResult(
            name=name,
            command="run it",
            status="passed",
            return_code=0,
            duration_seconds=1.0,
            output_tail="",
        )

    def test_a_non_verifying_build_republishes_a_record_that_still_stands(
        self, generator: ModuleType, tmp_path: Path, repo_inventory
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        stored = (self._result(generator, name),)
        self._store(generator, tmp_path, repo_inventory, stored)
        resolved = generator.resolve_verification(
            tmp_path,
            repo_inventory,
            verify=False,
            full_validation=False,
            reuse_verification=True,
        )
        assert [result.name for result in resolved.results] == [name]
        assert resolved.measured_elsewhere is False
        status, passed, failed, unrun = generator._verification_summary(
            resolved.results, full_validation=False
        )
        assert (status, passed, failed) == ("1 passed", 1, 0)
        assert unrun == len(generator.VERIFICATION_COMMANDS) - 1

    def test_a_record_describing_another_tree_is_carried_not_deleted(
        self, generator: ModuleType, tmp_path: Path, repo_inventory
    ) -> None:
        # Discarding it was the defect.  The record is the only copy of a
        # measurement that costs minutes, and the build replacing it measured
        # nothing, so it is republished with its own provenance instead.
        other = dataclasses.replace(repo_inventory, source_hash="0" * 16)
        name = generator.VERIFICATION_COMMANDS[0][0]
        self._store(generator, tmp_path, other, (self._result(generator, name),))
        resolved = generator.resolve_verification(
            tmp_path,
            repo_inventory,
            verify=False,
            full_validation=False,
            reuse_verification=True,
        )
        assert [result.name for result in resolved.results] == [name]
        assert resolved.measured_elsewhere is True
        assert resolved.source_hash == "0" * 16

    def test_reuse_is_the_default_for_the_command_line(
        self, generator: ModuleType
    ) -> None:
        # ``--rerun-verification`` is the opt-out.  A default invocation that
        # opted out of reuse is what deleted the record in the first place.
        assert generator._parse_args([]).rerun_verification is False
        assert generator._parse_args(["--rerun-verification"]).rerun_verification

    def test_a_non_verifying_build_runs_no_command(
        self, generator: ModuleType, tmp_path: Path, repo_inventory, monkeypatch
    ) -> None:
        def _fail(*_args, **_kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("a non-verifying build ran a verification command")

        monkeypatch.setattr(generator, "run_verification", _fail)
        resolved = generator.resolve_verification(
            tmp_path,
            repo_inventory,
            verify=False,
            full_validation=False,
            reuse_verification=True,
        )
        assert resolved.results == ()
        assert resolved.source_commit == repo_inventory.commit


class TestRunVerification:
    """``run_verification`` maps a real execution onto the record directly.

    Every other test fakes ``run_verification`` wholesale, so nothing pinned
    the mapping a real execution feeds it: exit code to status, combined
    output to the bounded tail the record stores, and the fact that a failing
    group does not stop the remaining groups from running.
    """

    @staticmethod
    def _fake_run(
        monkeypatch: pytest.MonkeyPatch,
        generator: ModuleType,
        outcomes: list[tuple[int, str, str]],
    ) -> None:
        """Drive shared execution from a queue of (code, stdout, stderr).

        The last outcome repeats when the queue empties, so a single entry
        fakes a uniform run and a pair fakes a divergence between the first
        group and everything after it.
        """
        pending = list(outcomes)

        def _run(command, name, timeout, **_kwargs):
            code, stdout, stderr = pending.pop(0) if len(pending) > 1 else pending[0]
            return execution.CommandResult(
                name=name,
                success=code == 0,
                duration=1.5,
                command=command,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                returncode=code,
            )

        monkeypatch.setattr(execution, "run_command", _run)

    def test_run_verification_maps_an_exit_code_to_a_bounded_failed_tail(
        self, generator: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        for directory in ("src", "examples"):
            (tmp_path / "GEO-INFER-FIXTURE" / directory).mkdir(parents=True)
        (tmp_path / "manuscript").mkdir()
        self._fake_run(monkeypatch, generator, [(3, "x" * 2500, "")])
        results = generator.run_verification(tmp_path)
        assert len(results) == len(generator.VERIFICATION_COMMANDS)
        first = results[0]
        assert first.status == "failed"
        assert first.return_code == 3
        assert 0 < len(first.output_tail) <= 2000

    def test_run_verification_continues_after_a_failing_first_group(
        self, generator: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        for directory in ("src", "examples"):
            (tmp_path / "GEO-INFER-FIXTURE" / directory).mkdir(parents=True)
        (tmp_path / "manuscript").mkdir()
        self._fake_run(
            monkeypatch, generator, [(1, "boom", "traceback"), (0, "ok", "")]
        )
        results = generator.run_verification(tmp_path)
        assert len(results) == len(generator.VERIFICATION_COMMANDS)
        assert results[0].status == "failed"
        assert results[0].return_code == 1
        assert "boom" in results[0].output_tail
        assert results[1].status == "passed"
        assert results[1].return_code == 0

    def test_interruption_retains_failure_and_leaves_later_groups_not_run(
        self, generator: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            generator,
            "VERIFICATION_COMMANDS",
            (("cancelled", "true"), ("must-not-launch", "true")),
        )
        admitted = []

        def interrupted(command, name, timeout, **_kwargs):
            admitted.append(name)
            assert name == "cancelled", "interruption admitted another group"
            return execution.CommandResult(
                name,
                False,
                1.0,
                command,
                status="INTERRUPTED",
                stderr="interrupted; cleanup receipt retained",
                returncode=None,
                receipt=str(tmp_path / "attempt" / "receipt.json"),
            )

        monkeypatch.setattr(execution, "run_command", interrupted)
        results = generator.run_verification(tmp_path)
        assert admitted == ["cancelled"]
        assert len(results) == 1 and results[0].status == "failed"
        assert results[0].receipt == "attempt/receipt.json"
        assert "cleanup receipt retained" in results[0].output_tail
        table = generator._verification_table(results, full_validation=False)
        assert "| `must-not-launch` | `true` | not run |" in table

    def test_compilation_expands_only_its_paths_against_execution_root(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        for module in ("GEO-INFER-Z", "GEO-INFER-A"):
            for directory in ("src", "examples"):
                (tmp_path / module / directory).mkdir(parents=True)
        (tmp_path / "manuscript").mkdir()
        argv = generator._verification_argv(
            "python -m compileall -q GEO-INFER-*/src GEO-INFER-*/examples manuscript",
            tmp_path,
        )
        assert argv == [
            "python",
            "-m",
            "compileall",
            "-q",
            "GEO-INFER-A/src",
            "GEO-INFER-Z/src",
            "GEO-INFER-A/examples",
            "GEO-INFER-Z/examples",
            "manuscript",
        ]
        literal = "print('$(false); * | &')"
        assert generator._verification_argv(
            shlex.join([sys.executable, "-c", literal]), tmp_path
        ) == [sys.executable, "-c", literal]

    def test_unmatched_compile_pattern_cannot_be_a_passing_empty_selection(
        self, generator: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            generator,
            "VERIFICATION_COMMANDS",
            (("compile", "python -m compileall -q GEO-INFER-*/src"),),
        )

        def forbidden(*_args, **_kwargs):
            pytest.fail("an unmatched compile selection launched a command")

        monkeypatch.setattr(execution, "run_command", forbidden)
        with pytest.raises(ValueError, match="matched no paths"):
            generator.run_verification(tmp_path)

    @pytest.mark.parametrize("paths", ["missing.py", ""])
    def test_missing_or_implicit_compile_paths_are_rejected(
        self, generator: ModuleType, tmp_path: Path, paths
    ) -> None:
        with pytest.raises(ValueError, match="does not exist|explicit path"):
            generator._verification_argv(f"python -m compileall -q {paths}", tmp_path)

    def test_compile_option_values_remain_literal(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        source = tmp_path / "valid.py"
        source.write_text("x = 1\n", encoding="utf-8")
        argv = [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "-x",
            ".*skip.*",
            "-d",
            "virtual/nonexistent",
            "valid.py",
        ]
        assert generator._verification_argv(shlex.join(argv), tmp_path) == argv

    @pytest.mark.parametrize(
        "commands",
        [
            (),
            (("same", "true"), ("same", "false")),
            ((" ", "true"),),
            (("empty", " "),),
            (("bad-quotes", "python '"),),
        ],
    )
    def test_invalid_selection_fails_before_any_execution(
        self, generator: ModuleType, tmp_path: Path, monkeypatch, commands
    ) -> None:
        def forbidden(*_args, **_kwargs):
            pytest.fail("invalid declarations launched a command")

        monkeypatch.setattr(generator, "VERIFICATION_COMMANDS", commands)
        monkeypatch.setattr(execution, "run_command", forbidden)
        with pytest.raises(ValueError):
            generator.run_verification(tmp_path)

    @pytest.mark.parametrize(
        "shared_status", ["FAIL", "METADATA_ERROR", "INTERRUPTED", "EMPTY"]
    )
    def test_unaccepted_shared_statuses_keep_diagnostics_and_fail(
        self, generator: ModuleType, tmp_path: Path, monkeypatch, shared_status
    ) -> None:
        monkeypatch.setattr(generator, "VERIFICATION_COMMANDS", (("one", "true"),))
        monkeypatch.setattr(
            execution,
            "run_command",
            lambda command, name, timeout, **_kwargs: execution.CommandResult(
                name,
                shared_status == "EMPTY",
                2.123456,
                command,
                stderr="cleanup inspection failed; retained diagnostic",
                returncode=0,
                status=shared_status,
            ),
        )
        (result,) = generator.run_verification(tmp_path)
        assert result.status == "failed"
        assert result.duration_seconds == 2.123
        assert "cleanup inspection failed" in result.output_tail
        assert "terminated" not in result.output_tail


class TestVerificationAttemptEvidence:
    """Actual commands retain complete logs and never overwrite prior attempts."""

    def test_failure_continuation_cwd_literal_arguments_and_immutable_logs(
        self, generator: ModuleType, execution_attempts: Path, monkeypatch
    ) -> None:
        root = execution_attempts
        invocation_log = root / "invocations.txt"
        marker = "literal $(false); * & |"
        script = (
            "import pathlib,sys; "
            "pathlib.Path('invocations.txt').open('a').write(sys.argv[1]+'\\n'); "
            "print('STDOUT-BEGIN'+ 'x'*6000 + 'STDOUT-END'); "
            "print('STDERR-BEGIN'+ 'y'*6000 + 'STDERR-END',file=sys.stderr); "
            "print(sys.argv[2]); sys.exit(int(sys.argv[3]))"
        )
        commands = tuple(
            (name, shlex.join([sys.executable, "-c", script, name, marker, str(code)]))
            for name, code in (("fails", 7), ("follows", 0))
        )
        monkeypatch.setattr(generator, "VERIFICATION_COMMANDS", commands)
        monkeypatch.setattr(generator, "VERIFICATION_TIMEOUT_SECONDS", 10)
        first = generator.run_verification(root)
        assert [result.status for result in first] == ["failed", "passed"]
        assert [result.return_code for result in first] == [7, 0]
        initial_bytes = {}
        for result in first:
            receipt, recorded = _attempt(root, result)
            assert recorded["command"] == shlex.split(result.command)
            stdout = (receipt.parent / "stdout.log").read_text(encoding="utf-8")
            stderr = (receipt.parent / "stderr.log").read_text(encoding="utf-8")
            assert "STDOUT-BEGIN" in stdout and "STDOUT-END" in stdout
            assert "STDERR-BEGIN" in stderr and "STDERR-END" in stderr
            assert marker in stdout
            assert len(result.output_tail) <= 2000
            initial_bytes.update(
                {path: path.read_bytes() for path in receipt.parent.iterdir()}
            )
        second = generator.run_verification(root)
        assert {result.receipt for result in first}.isdisjoint(
            result.receipt for result in second
        )
        assert all(
            path.read_bytes() == before for path, before in initial_bytes.items()
        )
        for result in second:
            _attempt(root, result)
        assert invocation_log.read_text(encoding="utf-8").splitlines() == [
            "fails",
            "follows",
            "fails",
            "follows",
        ]
        assert not (root / "output").exists()

    def test_missing_executable_is_recorded_and_later_group_runs(
        self, generator: ModuleType, execution_attempts: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            generator,
            "VERIFICATION_COMMANDS",
            (
                (
                    "missing",
                    shlex.join([str(execution_attempts / "missing executable")]),
                ),
                ("follows", shlex.join([sys.executable, "-c", "print('completed')"])),
            ),
        )
        monkeypatch.setattr(generator, "VERIFICATION_TIMEOUT_SECONDS", 10)
        failed, passed = generator.run_verification(execution_attempts)
        assert failed.status == "failed" and failed.return_code is None
        assert "FileNotFoundError" in failed.output_tail
        assert passed.status == "passed" and "completed" in passed.output_tail
        _attempt(execution_attempts, failed)
        _attempt(execution_attempts, passed)

    def test_legacy_result_and_new_receipt_reference_round_trip(
        self, generator: ModuleType
    ) -> None:
        legacy = {
            "name": "one",
            "command": "true",
            "status": "passed",
            "return_code": 0,
            "duration_seconds": 1.5,
            "output_tail": "ok",
        }
        assert generator.VerificationResult(**legacy).receipt is None
        old_record = generator.VerificationRecord(
            results=(generator.VerificationResult(**legacy),),
            source_commit="old",
            source_hash="old",
            full_validation_requested=False,
        )
        assert generator._verification_payload(old_record)["results"] == [legacy]
        current = generator.VerificationResult(
            **legacy, receipt=".geo-infer-test-results/runs/one/receipt.json"
        )
        decoded = json.loads(json.dumps(dataclasses.asdict(current)))
        assert generator.VerificationResult(**decoded) == current
        current_record = dataclasses.replace(old_record, results=(current,))
        assert generator._verification_payload(current_record)["results"] == [decoded]

    @pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan")])
    def test_invalid_deadline_creates_no_attempt(
        self, generator: ModuleType, execution_attempts: Path, monkeypatch, deadline
    ) -> None:
        monkeypatch.setattr(generator, "VERIFICATION_COMMANDS", (("one", "true"),))
        monkeypatch.setattr(generator, "VERIFICATION_TIMEOUT_SECONDS", deadline)
        with pytest.raises(ValueError, match="finite and positive"):
            generator.run_verification(execution_attempts)
        assert not (execution_attempts / ".geo-infer-test-results").exists()


class TestTierAxisOfReuse:
    """The reuse lookup is exact at the record's tier; a mismatch re-measures.

    The chosen rule, deliberately: a record is reused only when it was
    measured at the tier being requested.  A default-tier record never
    satisfies a full-validation request — its results cover the narrower
    tier, and accepting it would publish the wider tier's groups as evidence
    they never produced.  A full-validation record never satisfies a
    default-tier request — the caller that wants the narrower tier either
    re-measures or widens (which ``resolve_verification`` does).  A tier
    mismatch returns ``None``, and the caller runs the commands.
    """

    def _write(self, generator: ModuleType, root, payload) -> None:
        data = root / "output" / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "research_verification.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    @staticmethod
    def _entry(name: str) -> dict:
        return {
            "name": name,
            "command": "run it",
            "status": "passed",
            "return_code": 0,
            "duration_seconds": 1.0,
            "output_tail": "",
        }

    def test_a_full_tier_record_is_not_reused_for_a_default_request(
        self, generator: ModuleType, tmp_path: Path, repo_inventory
    ) -> None:
        name = generator.FULL_VALIDATION_COMMANDS[0][0]
        self._write(
            generator,
            tmp_path,
            {
                "schema_version": generator.RESEARCH_SCHEMA,
                "full_validation_requested": True,
                "source_commit": repo_inventory.commit,
                "source_hash": repo_inventory.source_hash,
                "results": [self._entry(name)],
            },
        )
        assert (
            generator.load_matching_verification(
                tmp_path, repo_inventory, full_validation=False
            )
            is None
        )

    def test_a_default_tier_record_is_not_reused_for_a_full_request(
        self, generator: ModuleType, tmp_path: Path, repo_inventory
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]
        self._write(
            generator,
            tmp_path,
            {
                "schema_version": generator.RESEARCH_SCHEMA,
                "full_validation_requested": False,
                "source_commit": repo_inventory.commit,
                "source_hash": repo_inventory.source_hash,
                "results": [self._entry(name)],
            },
        )
        assert (
            generator.load_matching_verification(
                tmp_path, repo_inventory, full_validation=True
            )
            is None
        )


class TestRunVerificationTimeout:
    """A hung validator is recorded as timed out rather than stalling the run.

    The verification runner is unbounded at the CI level (the manuscript job
    has no per-command ceiling of its own), so the runner itself must bound
    every command.  A command that exceeds the envelope is published as a
    ``timeout`` result with the elapsed duration, and the publication gate
    treats it exactly like a failure — the record must never show a hung
    group as a pass or as silently absent.
    """

    def test_a_hanging_command_times_out_and_is_recorded(
        self, generator: ModuleType, execution_attempts: Path, monkeypatch
    ) -> None:
        commands = (
            (
                "hangs",
                shlex.join([sys.executable, "-c", "import time; time.sleep(30)"]),
            ),
            ("answers", shlex.join([sys.executable, "-c", "print('ok')"])),
        )
        monkeypatch.setattr(generator, "VERIFICATION_COMMANDS", commands)
        monkeypatch.setattr(generator, "VERIFICATION_TIMEOUT_SECONDS", 2)
        results = generator.run_verification(execution_attempts)
        assert len(results) == 2
        assert results[0].status == "timeout"
        assert results[0].return_code is None
        assert results[0].duration_seconds >= 2
        assert "timed out" in results[0].output_tail.lower()
        assert results[1].status == "passed"
        _attempt(execution_attempts, results[0])
        _attempt(execution_attempts, results[1])

    def test_a_timeout_result_counts_as_a_failure_in_the_summary(
        self, generator: ModuleType
    ) -> None:
        results = [
            generator.VerificationResult(
                name=generator.VERIFICATION_COMMANDS[0][0],
                command="true",
                status="timeout",
                return_code=None,
                duration_seconds=2.0,
                output_tail="",
            )
        ]
        summary, passed, failed, _unrun = generator._verification_summary(
            results, full_validation=False
        )
        assert failed == 1
        assert passed == 0
        assert "failed" in summary

    def test_timeout_reaps_parent_and_detached_child_and_retains_output(
        self, generator: ModuleType, execution_attempts: Path, monkeypatch
    ) -> None:
        root = execution_attempts
        identities = root / "processes.json"
        child = root / "child.py"
        child.write_text(
            "import time\nprint('DETACHED-CHILD-READY', flush=True)\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        parent = root / "parent.py"
        parent.write_text(
            "import json,os,pathlib,subprocess,sys,time,psutil\n"
            "child=subprocess.Popen([sys.executable, 'child.py'],start_new_session=os.name=='posix')\n"
            "processes=[psutil.Process(os.getpid()),psutil.Process(child.pid)]\n"
            "pathlib.Path('processes.json').write_text(json.dumps([[p.pid,p.create_time()] for p in processes]))\n"
            "print('TIMEOUT-STDOUT-BEGIN'+'x'*6000+'TIMEOUT-STDOUT-END',flush=True)\n"
            "print('TIMEOUT-STDERR-BEGIN'+'y'*6000+'TIMEOUT-STDERR-END',file=sys.stderr,flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            generator,
            "VERIFICATION_COMMANDS",
            (
                ("hangs", shlex.join([sys.executable, str(parent)])),
                (
                    "follows",
                    shlex.join([sys.executable, "-c", "print('after-timeout')"]),
                ),
            ),
        )
        monkeypatch.setattr(generator, "VERIFICATION_TIMEOUT_SECONDS", 2)
        try:
            failed, passed = generator.run_verification(root)
            assert failed.status == "timeout" and failed.return_code is None
            assert passed.status == "passed" and "after-timeout" in passed.output_tail
            assert identities.is_file(), (
                "timeout must reach the actual process-tree fixture"
            )
            for pid, created in json.loads(identities.read_text(encoding="utf-8")):
                try:
                    process = psutil.Process(pid)
                    assert (
                        process.create_time() != created
                        or process.status() == psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    pass
            receipt, recorded = _attempt(root, failed)
            assert recorded["status"] == "TIMEOUT"
            stdout = (receipt.parent / "stdout.log").read_text(encoding="utf-8")
            stderr = (receipt.parent / "stderr.log").read_text(encoding="utf-8")
            assert "TIMEOUT-STDOUT-BEGIN" in stdout and "TIMEOUT-STDOUT-END" in stdout
            assert "DETACHED-CHILD-READY" in stdout
            assert "TIMEOUT-STDERR-BEGIN" in stderr and "TIMEOUT-STDERR-END" in stderr
            assert "Timed out" in stderr
            _attempt(root, passed)
            assert not (root / "output").exists()
        finally:
            if identities.is_file():
                for pid, created in json.loads(identities.read_text(encoding="utf-8")):
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == created:
                            process.kill()
                            process.wait(timeout=2)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass

    def test_a_timed_out_group_refuses_publication(
        self, generator: ModuleType, generatable_checkout: Path, monkeypatch
    ) -> None:
        name = generator.VERIFICATION_COMMANDS[0][0]

        def _run(_root, *, full_validation):
            return (
                generator.VerificationResult(
                    name=name,
                    command="sleep 30",
                    status="timeout",
                    return_code=None,
                    duration_seconds=2.0,
                    output_tail="timed out",
                ),
            )

        monkeypatch.setattr(generator, "run_verification", _run)
        with pytest.raises(RuntimeError, match=name):
            generator.generate(generatable_checkout, verify=True, publication=True)
