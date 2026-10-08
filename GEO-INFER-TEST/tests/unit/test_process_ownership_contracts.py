"""Process ownership remains strict for live or uninspectable candidates."""

import os
import subprocess
import traceback
from types import SimpleNamespace

import psutil
import pytest


from geo_infer_test import process as module


TOKEN = "a" * 32


@pytest.fixture(autouse=True)
def unobservable_kernel_exit(monkeypatch):
    """Fake PIDs must never read an unrelated real task's kernel flags."""
    monkeypatch.setattr(module, "_kernel_exiting", lambda pid: None)
    monkeypatch.setattr(module, "_kernel_denial_facts", lambda pid: None)


class BoundaryProcess:
    pid = 4321

    def __init__(self, *, environment=None, running=True, status=psutil.STATUS_RUNNING):
        self.environment = environment
        self.running = running
        self.state = status
        self.calls = []

    def environ(self):
        self.calls.append("environ")
        if isinstance(self.environment, BaseException):
            raise self.environment
        return self.environment or {}

    def is_running(self):
        self.calls.append("is_running")
        if isinstance(self.running, BaseException):
            raise self.running
        return self.running

    def status(self):
        self.calls.append("status")
        if isinstance(self.state, BaseException):
            raise self.state
        return self.state

    def create_time(self):
        self.calls.append("create_time")
        return 123.0


def census():
    return module._DescendantCensus(9999, TOKEN)


@pytest.mark.parametrize(
    "value,expected",
    [
        (TOKEN, True),
        ("", False),
        ("b" * 32, False),
        ("b" * 32 + ":" + TOKEN, True),
        (TOKEN + "0", False),
        ("0" + TOKEN, False),
    ],
)
def test_readable_environment_preserves_exact_token(value, expected):
    child = BoundaryProcess(
        environment={module._OWNERSHIP_ENV: value},
        status=RuntimeError("must not inspect"),
    )
    assert census()._owned(child) is expected
    assert child.calls == ["environ"]


def test_environment_no_such_process_is_positive_exit():
    child = BoundaryProcess(environment=psutil.NoSuchProcess(4321))
    assert census()._owned(child) is False
    assert child.calls == ["environ"]


@pytest.mark.parametrize(
    "running,state,expected_calls",
    [
        (False, RuntimeError("must not inspect reused PID"), ["environ", "is_running"]),
        (True, psutil.STATUS_ZOMBIE, ["environ", "is_running", "status"]),
        (
            psutil.NoSuchProcess(4321),
            RuntimeError("must not inspect exited PID"),
            ["environ", "is_running"],
        ),
        (True, psutil.NoSuchProcess(4321), ["environ", "is_running", "status"]),
    ],
)
def test_denied_environment_excludes_only_positive_stopped_identity(
    running, state, expected_calls
):
    child = BoundaryProcess(
        environment=psutil.AccessDenied(4321), running=running, status=state
    )
    assert census()._owned(child) is False
    assert child.calls == expected_calls


@pytest.mark.parametrize(
    "state",
    [
        psutil.STATUS_RUNNING,
        psutil.STATUS_SLEEPING,
        psutil.STATUS_STOPPED,
        psutil.STATUS_TRACING_STOP,
        psutil.STATUS_DEAD,
        "unknown",
    ],
)
def test_denied_live_or_unknown_state_remains_fatal(state):
    error = psutil.AccessDenied(4321)
    child = BoundaryProcess(environment=error, status=state)
    with pytest.raises(psutil.AccessDenied) as caught:
        census()._owned(child)
    assert caught.value is error
    assert child.calls == ["environ", "is_running", "status"]


@pytest.mark.parametrize(
    "step,error",
    [
        ("running", psutil.AccessDenied(4321)),
        ("running", RuntimeError("identity unavailable")),
        ("status", psutil.AccessDenied(4321)),
        ("status", RuntimeError("state unavailable")),
    ],
)
def test_identity_or_status_inspection_failure_remains_fatal(step, error):
    denied = psutil.AccessDenied(4321)
    child = BoundaryProcess(
        environment=denied,
        running=error if step == "running" else True,
        status=error if step == "status" else psutil.STATUS_ZOMBIE,
    )
    with pytest.raises(type(error)) as caught:
        census()._owned(child)
    assert caught.value is error
    assert caught.value.__context__ is denied


def native_listing(child):
    return subprocess.CompletedProcess(
        ["ps"],
        0,
        f"{child.pid} command {module._OWNERSHIP_ENV}={TOKEN} PRIVATE_SENTINEL=do-not-retain\n",
        "",
    )


def posix_boundary(monkeypatch, child):
    owned = census()
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", environ=os.environ))
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: native_listing(child)
    )
    monkeypatch.setattr(owned.psutil, "Process", lambda pid: child)
    return owned


@pytest.mark.parametrize(
    "field,expected",
    [
        (f"{module._OWNERSHIP_ENV}={TOKEN}", True),
        (f"{module._OWNERSHIP_ENV}={'b' * 32}:{TOKEN}", True),
        (f"{module._OWNERSHIP_ENV}={TOKEN}0", False),
        (f"{module._OWNERSHIP_ENV}=0{TOKEN}", False),
        (f"OTHER={TOKEN}", False),
        (f"PREFIX_{module._OWNERSHIP_ENV}={TOKEN}", False),
        (f"{module._OWNERSHIP_ENV}={TOKEN}INVALID", False),
        (f"{module._OWNERSHIP_ENV}={'b' * 32}", False),
    ],
)
def test_native_long_listing_retains_only_exact_owned_candidates(
    monkeypatch, field, expected
):
    child = BoundaryProcess(environment={module._OWNERSHIP_ENV: TOKEN})
    owned = posix_boundary(monkeypatch, child)
    inspected = []

    def identity(pid):
        inspected.append(pid)
        return child

    monkeypatch.setattr(owned.psutil, "Process", identity)
    # Public synthetic rows exercise long negative input and token lookalikes;
    # native text remains only a filter before actual environment verification.
    listing = "1 command PUBLIC=" + "x" * 100_000 + "\n"
    listing += f"{child.pid} command {field}\n"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(["ps"], 0, listing, ""),
    )
    owned.refresh(timeout=2)
    assert inspected == ([child.pid] if expected else [])
    assert list(owned.processes) == ([child.pid] if expected else [])


@pytest.mark.parametrize("chain", [TOKEN, "b" * 32 + ":" + TOKEN])
def test_native_observer_has_no_target_tokens_and_preserves_parent_environment(
    monkeypatch, chain
):
    """Outer owners must never claim their nested, synchronous native observer."""
    monkeypatch.setenv(module._OWNERSHIP_ENV, chain)
    monkeypatch.setenv("GEO_INFER_OBSERVER_CONTROL", "preserved")
    environment_before = os.environ.copy()
    calls = []

    def inspect_scanner(command, **kwargs):
        assert command == ["ps", "axeww", "-o", "pid=,command="]
        assert module._OWNERSHIP_ENV not in kwargs["env"]
        assert kwargs["env"] == {
            key: value
            for key, value in environment_before.items()
            if key != module._OWNERSHIP_ENV
        }
        assert 0 < kwargs["timeout"] <= module._CENSUS_BUDGET_SECONDS
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", environ=os.environ))
    monkeypatch.setattr(module.subprocess, "run", inspect_scanner)
    census().refresh(timeout=1)
    assert len(calls) == 1
    assert os.environ == environment_before


def test_positive_zombie_native_candidate_is_excluded(monkeypatch):
    child = BoundaryProcess(
        environment=psutil.AccessDenied(4321), status=psutil.STATUS_ZOMBIE
    )
    owned = posix_boundary(monkeypatch, child)
    owned.refresh(timeout=1)
    assert owned.processes == {}
    assert child.calls == ["create_time", "environ", "is_running", "status"]


@pytest.mark.parametrize(
    "state",
    [
        psutil.STATUS_RUNNING,
        psutil.AccessDenied(4321),
        RuntimeError("state unavailable"),
    ],
)
def test_positive_native_candidate_with_denied_live_or_unknown_state_fails_closed(
    monkeypatch, state
):
    child = BoundaryProcess(environment=psutil.AccessDenied(4321), status=state)
    owned = posix_boundary(monkeypatch, child)
    with pytest.raises(
        type(state) if isinstance(state, BaseException) else psutil.AccessDenied
    ) as caught:
        owned.refresh(timeout=1)
    assert "PRIVATE_SENTINEL" not in str(caught.value)
    assert "do-not-retain" not in str(caught.value)
    assert owned.processes == {}


@pytest.mark.parametrize("phase", ["identity", "environment", "status"])
def test_native_denial_retains_failure_identity_with_sanitized_phase(
    monkeypatch, phase
):
    error = psutil.AccessDenied(4321, name="PRIVATE_PROCESS", msg="PRIVATE_MESSAGE")
    child = BoundaryProcess(
        environment=error if phase == "environment" else psutil.AccessDenied(4321),
        status=error if phase == "status" else psutil.STATUS_RUNNING,
    )
    if phase == "identity":

        def denied_identity():
            raise error

        child.create_time = denied_identity
    owned = posix_boundary(monkeypatch, child)
    with pytest.raises(psutil.AccessDenied) as caught:
        owned.refresh(timeout=1)
    assert caught.value is error
    assert error.pid == 4321
    assert error.name is None
    assert error.msg == f"Owned PID 4321: {phase} inspection denied"
    assert "PRIVATE_PROCESS" not in str(error)
    assert "PRIVATE_MESSAGE" not in str(error)
    assert owned.processes == {}


@pytest.mark.parametrize("phase", ["identity", "status", "unknown-status"])
def test_distinct_inspection_failure_preserves_sanitized_exception_chain(phase):
    native_cause = PermissionError("native environment read denied")
    denied = psutil.AccessDenied(
        4321, name="PRIVATE_ENVIRONMENT_NAME", msg="PRIVATE_ENVIRONMENT_MESSAGE"
    )
    denied.__cause__ = native_cause
    inspection_error = (
        RuntimeError("state unavailable")
        if phase == "unknown-status"
        else psutil.AccessDenied(
            4321, name="PRIVATE_INSPECTION_NAME", msg="PRIVATE_INSPECTION_MESSAGE"
        )
    )
    child = BoundaryProcess(
        environment=denied,
        running=inspection_error if phase == "identity" else True,
        status=inspection_error,
    )
    with pytest.raises(type(inspection_error)) as caught:
        census()._owned(child)
    assert caught.value is inspection_error
    assert inspection_error.__context__ is denied
    assert denied.__cause__ is native_cause
    assert denied.pid == 4321
    assert denied.name is None
    assert denied.msg == "Owned PID 4321: environment inspection denied"
    formatted = "".join(traceback.format_exception(caught.value))
    assert "PRIVATE_ENVIRONMENT_NAME" not in formatted
    assert "PRIVATE_ENVIRONMENT_MESSAGE" not in formatted
    assert "PRIVATE_INSPECTION_NAME" not in formatted
    assert "PRIVATE_INSPECTION_MESSAGE" not in formatted
    assert "Owned PID 4321: environment inspection denied" in formatted


def test_readable_detached_live_candidate_is_retained(monkeypatch):
    child = BoundaryProcess(environment={module._OWNERSHIP_ENV: TOKEN})
    owned = posix_boundary(monkeypatch, child)
    owned.refresh(timeout=1)
    assert owned.processes == {4321: child}
    assert owned.live() == [4321]


def test_zombie_positive_status_cannot_extend_refresh_deadline(monkeypatch):
    now = [0.0]
    child = BoundaryProcess(
        environment=psutil.AccessDenied(4321), status=psutil.STATUS_ZOMBIE
    )
    original_status = child.status

    def late_status():
        now[0] = 2.0
        return original_status()

    child.status = late_status
    owned = posix_boundary(monkeypatch, child)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with pytest.raises(subprocess.TimeoutExpired):
        owned.refresh(timeout=1)
    assert owned.processes == {}


def test_successful_owned_cleanup_still_kills_and_waits_identity(monkeypatch):
    child = BoundaryProcess(environment={module._OWNERSHIP_ENV: TOKEN})
    calls = []
    child.kill = lambda: (
        calls.append("kill"),
        setattr(child, "state", psutil.STATUS_ZOMBIE),
    )
    owned = census()
    owned.processes[child.pid] = child
    monkeypatch.setattr(
        owned.psutil,
        "wait_procs",
        lambda children, timeout: calls.append(("wait", tuple(children), timeout)),
    )
    owned.kill()
    assert calls == ["kill", ("wait", (child,), module._DESCENDANT_REAP_SECONDS)]
    assert owned.live() == []


def test_arbitrary_environment_error_is_not_reclassified():
    error = RuntimeError("environment failure")
    child = BoundaryProcess(environment=error, status=psutil.STATUS_ZOMBIE)
    with pytest.raises(RuntimeError) as caught:
        census()._owned(child)
    assert caught.value is error
    assert child.calls == ["environ"]


def observation_clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    return now, sleeps


@pytest.mark.parametrize("exit_kind", ["zombie", "gone", "reused"])
def test_exiting_denied_identity_requires_positive_stop(monkeypatch, exit_kind):
    now, sleeps = observation_clock(monkeypatch)
    child = BoundaryProcess(environment=psutil.AccessDenied(4321))

    def running():
        if exit_kind == "reused" and now[0] >= 0.003:
            return False
        return True

    def status():
        if now[0] >= 0.003:
            if exit_kind == "gone":
                raise psutil.NoSuchProcess(child.pid)
            return psutil.STATUS_ZOMBIE
        return psutil.STATUS_RUNNING

    child.is_running = running
    child.status = status
    assert census()._owned(child, deadline=1) is False
    assert now[0] == pytest.approx(0.003)
    assert sleeps and child.calls == ["environ"]


@pytest.mark.parametrize("deadline", [0.003, 10])
def test_persistent_live_denial_respects_existing_deadline_and_grace(
    monkeypatch, deadline
):
    now, sleeps = observation_clock(monkeypatch)
    error = psutil.AccessDenied(4321)
    child = BoundaryProcess(environment=error)
    with pytest.raises(psutil.AccessDenied) as caught:
        census()._owned(child, deadline=deadline)
    assert caught.value is error
    assert now[0] == pytest.approx(min(deadline, module._EXIT_OBSERVATION_SECONDS))
    assert sleeps and max(sleeps) <= 0.001
    assert child.calls.count("environ") == 1


def test_unknown_status_during_exit_observation_stays_fatal(monkeypatch):
    now, _ = observation_clock(monkeypatch)
    child = BoundaryProcess(environment=psutil.AccessDenied(4321))

    def status():
        if now[0] >= 0.002:
            raise RuntimeError("status inspection failed")
        return psutil.STATUS_RUNNING

    child.status = status
    with pytest.raises(RuntimeError, match="status inspection failed"):
        census()._owned(child, deadline=1)
    assert now[0] == pytest.approx(0.002)


def test_posix_refresh_observes_exit_without_extending_deadline(monkeypatch):
    now, _ = observation_clock(monkeypatch)
    child = BoundaryProcess(environment=psutil.AccessDenied(4321))
    owned = posix_boundary(monkeypatch, child)
    with pytest.raises(psutil.AccessDenied):
        owned.refresh(timeout=0.004)
    assert now[0] == pytest.approx(0.004)
    assert owned.processes == {}


def test_real_psutil_cached_create_time_mismatch_excludes_reused_identity(monkeypatch):
    """Exercise real psutil identity comparison without killing a foreign PID."""
    identity = psutil.Process(os.getpid())
    actual_creation = identity.create_time()
    # Represent a retained older identity for this currently live numeric PID.
    # The kernel PID itself is not reused or modified by this private oracle.
    identity._ident = (identity.pid, actual_creation - 1.0)

    def denied():
        raise psutil.AccessDenied(identity.pid)

    def must_not_inspect_status():
        raise AssertionError("Reused identity must short-circuit status inspection")

    monkeypatch.setattr(identity, "environ", denied)
    monkeypatch.setattr(identity, "status", must_not_inspect_status)
    assert census()._owned(identity) is False
    assert identity.is_running() is False
    current = psutil.Process(os.getpid())
    assert current.is_running() is True and current.create_time() == actual_creation


def _stat_line(*, comm: str, flags: int) -> str:
    # pid (comm) state ppid pgrp session tty_nr tpgid flags ...
    return f"4321 ({comm}) R 1 1 1 0 -1 {flags} 0 0 0 0\n"


@pytest.mark.parametrize(
    "comm,flags,expected",
    [
        ("python3", 0x00400100, False),
        ("python3", 0x00400104, True),
        # A task may rename itself; only the last ")" delimits comm.
        ("x) R 1 1 1 0 -1 4 (y", 0x00400100, False),
        ("x) R 1 1 1 0 -1 0 (y", 0x00400104, True),
    ],
)
def test_kernel_stat_exit_flag_is_parsed_after_the_final_comm_delimiter(
    comm, flags, expected
):
    assert module._stat_reports_exiting(_stat_line(comm=comm, flags=flags)) is expected


@pytest.mark.parametrize("text", ["", "4321 (python3", "4321 (python3) R 1 1"])
def test_unparseable_kernel_stat_is_unknown_not_exited(text):
    assert module._stat_reports_exiting(text) is None


def test_denied_environment_of_kernel_exiting_task_is_positive_exit(monkeypatch):
    """Linux releases an exiting task's mm before it becomes a zombie.

    During that window /proc/<pid>/environ is denied while status still
    reads as running. PF_EXITING is the kernel's positive exit observation.
    """
    now, _ = observation_clock(monkeypatch)
    child = BoundaryProcess(environment=psutil.AccessDenied(4321))
    observed = []

    def exiting(pid):
        observed.append(pid)
        return now[0] >= 0.003

    monkeypatch.setattr(module, "_kernel_exiting", exiting)
    assert census()._owned(child, deadline=1) is False
    assert now[0] == pytest.approx(0.003)
    assert set(observed) == {4321}
    assert child.calls.count("environ") == 1


@pytest.mark.parametrize("flag", [False, None])
def test_live_or_unobservable_kernel_flags_keep_denial_fatal(monkeypatch, flag):
    now, _ = observation_clock(monkeypatch)
    error = psutil.AccessDenied(4321)
    child = BoundaryProcess(environment=error)
    monkeypatch.setattr(module, "_kernel_exiting", lambda pid: flag)
    with pytest.raises(psutil.AccessDenied) as caught:
        census()._owned(child, deadline=1)
    assert caught.value is error
    assert now[0] == pytest.approx(module._EXIT_OBSERVATION_SECONDS)


def test_real_kernel_observation_of_this_live_process(monkeypatch):
    """Linux procfs reports this live task as not exiting; elsewhere unknown."""
    monkeypatch.undo()
    procfs = os.path.exists(f"/proc/{os.getpid()}/stat")
    assert module._kernel_exiting(os.getpid()) is (False if procfs else None)


def test_missing_procfs_entry_is_unobservable(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(module, "_PROCFS", tmp_path)
    assert module._kernel_exiting(4321) is None


def _fake_task(root, pid, *, stat, status, environ_mode=0o400, attr=None):
    task = root / str(pid)
    task.mkdir()
    (task / "stat").write_text(stat)
    (task / "status").write_text(status)
    (task / "environ").write_bytes(b"")
    (task / "environ").chmod(environ_mode)
    if attr is not None:
        (task / "attr").mkdir()
        (task / "attr" / "current").write_text(attr)
    return task


def test_denial_facts_are_content_free_kernel_observations(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(module, "_PROCFS", tmp_path)
    task = _fake_task(
        tmp_path,
        4321,
        stat=_stat_line(comm="PRIVATE_COMM", flags=0x00400100),
        status=(
            "Name:\tPRIVATE_NAME\n"
            "Uid:\t1000\t1000\t1000\t1000\n"
            "Threads:\t3\nNoNewPrivs:\t0\nSeccomp:\t2\n"
        ),
        attr="/usr/bin/PRIVATE_PROFILE (enforce)\n",
    )
    owner = os.stat(task / "environ").st_uid
    facts = module._kernel_denial_facts(4321, uid=owner)
    assert facts == (
        "state=R flags=0x400100 environ_owner=self "
        f"uid_match={owner == 1000} "
        "threads=3 no_new_privs=0 seccomp=2 lsm=confined(enforce)"
    )
    assert "PRIVATE" not in facts


def test_denial_facts_report_foreign_environ_owner_and_unconfined(
    tmp_path, monkeypatch
):
    monkeypatch.undo()
    monkeypatch.setattr(module, "_PROCFS", tmp_path)
    task = _fake_task(
        tmp_path,
        4321,
        stat=_stat_line(comm="x", flags=0x4),
        status="Uid:\t7\t7\t7\t7\n",
        attr="unconfined\n",
    )
    owner = os.stat(task / "environ").st_uid
    facts = module._kernel_denial_facts(4321, uid=owner + 1)
    assert facts == (
        f"state=R flags=0x4 environ_owner={owner} uid_match={owner + 1 == 7} "
        "lsm=unconfined"
    )


def test_denial_facts_need_posix_credentials(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(module, "_PROCFS", tmp_path)
    monkeypatch.setattr(module, "os", SimpleNamespace(stat=os.stat))
    assert module._kernel_denial_facts(4321) is None


def test_missing_task_has_no_denial_facts(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(module, "_PROCFS", tmp_path)
    assert module._kernel_denial_facts(4321, uid=1000) is None


def test_persistent_denial_retains_kernel_facts_in_sanitized_message(monkeypatch):
    now, _ = observation_clock(monkeypatch)
    error = psutil.AccessDenied(4321, name="PRIVATE_NAME", msg="PRIVATE_MESSAGE")
    child = BoundaryProcess(environment=error)
    observed_at = []

    def facts(pid):
        observed_at.append(now[0])
        return "state=S flags=0x0"

    monkeypatch.setattr(module, "_kernel_denial_facts", facts)
    with pytest.raises(psutil.AccessDenied) as caught:
        census()._owned(child, deadline=1)
    assert caught.value is error
    assert error.msg == (
        "Owned PID 4321: environment inspection denied (state=S flags=0x0)"
    )
    # Observed at the denial, before a short-lived task can disappear.
    assert observed_at == [0.0]
    assert "PRIVATE" not in str(error)
