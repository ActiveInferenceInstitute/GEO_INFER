"""Deadline-bounded subprocesses with descendant cleanup."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
import uuid
import re
from typing import Any, cast

_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_DESCENDANTS: dict[subprocess.Popen[str], _DescendantCensus] = {}
_PROCESS_LOCKS: dict[subprocess.Popen[str], Any] = {}
_ACTIVE_LOCK = threading.Lock()
_CANCELLED = threading.Event()


_OWNERSHIP_ENV = "GEO_INFER_PROCESS_TOKENS"
_CENSUS_BUDGET_SECONDS = 5
_DESCENDANT_REAP_SECONDS = 2
_EXIT_OBSERVATION_SECONDS = 0.05
_PROCFS = Path("/proc")
# include/linux/sched.h: set by do_exit() before exit_mm() releases the
# address space, and retained until the task is reaped.
_PF_EXITING = 0x00000004


class OwnedProcessLeakError(subprocess.SubprocessError):
    """A nominally successful command retained owned background processes."""

    def __init__(self, pids: list[int], stdout: str, stderr: str):
        super().__init__(f"Command left owned background processes running: {pids}")
        self.output = stdout
        self.stderr = stderr


class ProcessCensusError(subprocess.SubprocessError):
    """Ownership inspection failed independently of the target deadline."""


def _annotate_ownership_denial(error: Any, *, pid: int, phase: str) -> None:
    """Retain failure type and identity without process names or environment."""
    error.name = None
    error.msg = f"Owned PID {pid}: {phase} inspection denied"


def _stat_reports_exiting(text: str) -> bool | None:
    """Return PF_EXITING from one /proc/<pid>/stat line, or None if unparseable."""
    # comm is caller-controlled and may contain ")" or spaces; the last ")"
    # ends it. The remaining fields start at state (3); flags is field 9.
    _, delimiter, rest = text.rpartition(")")
    fields = rest.split()
    if not delimiter or len(fields) < 7:
        return None
    try:
        flags = int(fields[6])
    except ValueError:
        return None
    return bool(flags & _PF_EXITING)


def _kernel_exiting(pid: int) -> bool | None:
    """Observe Linux PF_EXITING for ``pid``; None where it cannot be observed.

    After exit_mm() procfs reassigns the task's 0400 ``environ`` inode to
    root, so it is denied while status is not yet zombie; the 0444 ``stat``
    inode stays world-readable.
    """
    try:
        text = (_PROCFS / str(pid) / "stat").read_text(encoding="latin-1")
    except OSError:
        return None
    return _stat_reports_exiting(text)


def _kernel_denial_facts(
    pid: int, *, uid: int | None = None, gid: int | None = None
) -> str | None:
    """Summarize why ``pid``'s environment may be denied, without its content.

    Only kernel state is reported: run state and flags, whether procfs made
    ``environ`` root-owned (a non-dumpable or mm-less task), credential
    equality, seccomp/no_new_privs and whether an LSM confines the task. Names,
    command lines, environment and LSM profile names are never read into it.
    """
    if uid is None:
        getuid = getattr(os, "getuid", None)
        if getuid is None:
            return None
        uid = getuid()
    if gid is None:
        getgid = getattr(os, "getgid", None)
        gid = getgid() if getgid is not None else None
    task = _PROCFS / str(pid)
    try:
        stat = (task / "stat").read_text(encoding="latin-1")
        status = (task / "status").read_text(encoding="latin-1")
    except OSError:
        return None
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 7 or not fields[6].isdigit():
        return None
    lines = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    try:
        owner = os.stat(task / "environ").st_uid
    except OSError:
        owner_fact = "unavailable"
    else:
        owner_fact = "self" if owner == uid else str(owner)
    uids = lines.get("Uid", "").split()
    gids = lines.get("Gid", "").split()
    gid_match = (
        str(all(value == str(gid) for value in gids))
        if gid is not None
        and len(gids) == 4
        and all(value.isdecimal() for value in gids)
        else "unavailable"
    )
    facts = [
        f"state={fields[0]}",
        f"flags={int(fields[6]):#x}",
        f"environ_owner={owner_fact}",
        f"uid_match={bool(uids) and all(value == str(uid) for value in uids)}",
        f"gid_match={gid_match}",
    ]
    for key, label in (
        ("Threads", "threads"),
        ("NoNewPrivs", "no_new_privs"),
        ("Seccomp", "seccomp"),
    ):
        value = lines.get(key, "").strip()
        if value.isdigit():
            facts.append(f"{label}={value}")
    try:
        label_text = (task / "attr" / "current").read_text(encoding="latin-1").strip()
    except OSError:
        label_text = ""
    if label_text:
        mode = label_text.rpartition("(")[2].rstrip(")") if "(" in label_text else ""
        facts.append(
            "lsm=unconfined"
            if label_text == "unconfined"
            else f"lsm=confined({mode})"
            if mode in {"enforce", "complain", "kill"}
            else "lsm=confined"
        )
    return " ".join(facts)


class _DescendantCensus:
    """Retain launch-token ownership and identities across orphaning.

    Tokens are inherited and nested commands append their own token. This
    closes the immediate-detachment race that a PPID census cannot detect.
    Native census output is internal, discarded after matching our token,
    and never becomes command output or receipt data. Deliberately stripping
    the launch environment is outside this cooperative execution contract.
    """

    def __init__(self, pid: int, token: str):
        import psutil

        self.psutil = psutil
        self.pid = pid
        self.token = token
        self.processes: dict[int, Any] = {}
        self.lock = threading.RLock()

    def _owned(self, process: Any, *, deadline: float | None = None) -> bool:
        try:
            return self.token in process.environ().get(_OWNERSHIP_ENV, "").split(":")
        except self.psutil.NoSuchProcess:
            return False
        except self.psutil.AccessDenied as error:
            # A later identity/status failure retains this exception as its
            # context. Sanitize it before those inspections can raise.
            _annotate_ownership_denial(error, pid=process.pid, phase="environment")
            # Observe the denied state now: a short-lived task may be gone
            # by the time the bounded observation below ends.
            facts = _kernel_denial_facts(process.pid)
            # Linux can remove an exiting task's environment before marking
            # it as a zombie. Observe that same identity for at most 50 ms,
            # within the caller's existing census deadline. Only a positive
            # exit/zombie/PF_EXITING result excludes it; persistent denial or
            # an unknown status remains fatal. No environment read or command is retried.
            stop = (
                min(deadline, time.monotonic() + _EXIT_OBSERVATION_SECONDS)
                if deadline is not None
                else time.monotonic()
            )
            while True:
                phase = "identity"
                try:
                    if not process.is_running():
                        return False
                    phase = "status"
                    if process.status() == self.psutil.STATUS_ZOMBIE:
                        return False
                    # Between exit_mm() and exit_notify() the environment is
                    # denied but the task is not yet a zombie; under load
                    # that window can outlast the observation budget.
                    if _kernel_exiting(process.pid) is True:
                        return False
                except self.psutil.NoSuchProcess:
                    return False
                except self.psutil.AccessDenied as inspection_error:
                    _annotate_ownership_denial(
                        inspection_error, pid=process.pid, phase=phase
                    )
                    raise
                remaining = stop - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.001, remaining))
            if facts:
                error.msg = f"{error.msg} ({facts})"
            raise

    def refresh(self, *, timeout: float) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Census timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        if not self.lock.acquire(timeout=timeout):
            raise subprocess.TimeoutExpired("owned process census", timeout)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("owned process census", timeout)
            if os.name == "posix":
                # One native scan avoids per-process environment reads for
                # unrelated PIDs. Only matches receive identity verification.
                # This synchronous observer must not become an owned target
                # of an outer, nested census while its short-lived PID exits.
                # Ordinary child launches retain the complete token chain.
                scanner_env = os.environ.copy()
                scanner_env.pop(_OWNERSHIP_ENV, None)
                try:
                    listing = subprocess.run(
                        ["ps", "axeww", "-o", "pid=,command="],
                        capture_output=True,
                        text=True,
                        errors="replace",
                        env=scanner_env,
                        timeout=min(_CENSUS_BUDGET_SECONDS, remaining),
                        check=True,
                    )
                except subprocess.SubprocessError as exc:
                    # Never attach the internal environment census to logs.
                    cast(Any, exc).output = None
                    cast(Any, exc).stderr = None
                    if (
                        isinstance(exc, subprocess.TimeoutExpired)
                        and time.monotonic() < deadline
                    ):
                        raise ProcessCensusError(
                            f"Owned process census exceeded its {min(_CENSUS_BUDGET_SECONDS, timeout):g}s inspection budget"
                        ) from exc
                    raise
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("owned process census", timeout)
                candidates = []
                pattern = re.compile(
                    r"(?:^|\s)" + _OWNERSHIP_ENV + r"=([0-9a-f:]+)(?:\s|$)"
                )
                for line in listing.stdout.splitlines():
                    # Exact membership below remains the ownership boundary;
                    # token-negative rows cannot contain an owned candidate.
                    if self.token not in line:
                        continue
                    match = pattern.search(line)
                    if match and self.token in match.group(1).split(":"):
                        candidates.append(int(line.split(None, 1)[0]))
                del listing
                for pid in candidates:
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired("owned process census", timeout)
                    if pid == self.pid:
                        continue
                    try:
                        try:
                            child = self.psutil.Process(pid)
                            child.create_time()
                        except self.psutil.AccessDenied as error:
                            _annotate_ownership_denial(error, pid=pid, phase="identity")
                            raise
                        if self._owned(child, deadline=deadline):
                            self.processes[pid] = child
                    except self.psutil.NoSuchProcess:
                        pass
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("owned process census", timeout)
                return
            for child in self.psutil.process_iter():
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("owned process census", timeout)
                if child.pid == self.pid:
                    continue
                try:
                    if self._owned(child):
                        child.create_time()
                        self.processes[child.pid] = child
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    # System-owned processes cannot belong to this launch.
                    continue
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("owned process census", timeout)
        finally:
            self.lock.release()

    def live(self) -> list[int]:
        with self.lock:
            result = []
            for child in self.processes.values():
                try:
                    if (
                        child.is_running()
                        and child.status() != self.psutil.STATUS_ZOMBIE
                    ):
                        result.append(child.pid)
                except self.psutil.NoSuchProcess:
                    pass
            return result

    def kill(self) -> None:
        failures = []
        with self.lock:
            children = tuple(self.processes.values())
            for child in reversed(children):
                try:
                    child.kill()
                except self.psutil.NoSuchProcess:
                    pass
                except Exception as exc:
                    failures.append(exc)
            # Sending SIGKILL does not establish that an orphan has stopped.
            # Output pipes may close before its process status changes, so wait
            # for the owned identities within a separate, finite cleanup budget.
            if children:
                try:
                    self.psutil.wait_procs(children, timeout=_DESCENDANT_REAP_SECONDS)
                except Exception as exc:
                    failures.append(exc)
                # A non-child zombie has stopped even if its platform reaper
                # has not removed the PID yet; ownership liveness excludes it.
                try:
                    survivors = self.live()
                except Exception as exc:
                    failures.append(exc)
                else:
                    if survivors:
                        failures.append(
                            RuntimeError(
                                f"Owned descendants survived cleanup: {survivors}"
                            )
                        )
        if failures:
            raise ExceptionGroup("Owned descendant cleanup failed", failures)


def reset_process_cancellation() -> None:
    """Start a new fleet only after the previous workers were reaped."""
    _CANCELLED.clear()


def terminate_running_processes() -> None:
    """Cancel processes owned by this execution engine on fleet interruption."""
    _CANCELLED.set()
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    failures = []
    for process in processes:
        try:
            terminate_tree(process)
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise ExceptionGroup("Fleet process cleanup failed", failures)


def terminate_tree(process: subprocess.Popen[str]) -> None:
    """Terminate the child session, including descendants holding output pipes."""
    with _ACTIVE_LOCK:
        census = _DESCENDANTS.get(process)
    # Retained descendants must be terminated even when the original parent
    # has exited and the native tree tool can no longer discover its children.
    try:
        # Stop the positively owned, unreaped primary before the slower token
        # census, so it cannot continue launching work during cleanup.
        _terminate_native_tree(process)
    finally:
        if census is not None:
            try:
                # The last pre-deadline snapshot may precede an immediate
                # spawn-and-detach. Cleanup gets one bounded ownership scan
                # after the deadline, independently of the original PPID.
                census.refresh(timeout=_CENSUS_BUDGET_SECONDS)
            finally:
                census.kill()


def _terminate_native_tree(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_LOCK:
        process_lock = _PROCESS_LOCKS.get(process, threading.RLock())
    with process_lock:
        _terminate_unreaped_native_tree(process)


def _terminate_unreaped_native_tree(process: subprocess.Popen[str]) -> None:
    # Popen.poll() establishes that this unreaped child still owns the PID.
    # A reaped parent's numeric PID/group can already belong to another task.
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    else:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if result.returncode and process.poll() is None:
                raise RuntimeError("Process-tree cleanup failed on Windows")
        finally:
            if process.poll() is None:
                process.kill()


def _capture_after_cleanup(
    process: subprocess.Popen[str], process_lock: Any
) -> tuple[str, str]:
    """Drain target output even when ownership cleanup itself reports failure."""
    diagnostics = []
    try:
        terminate_tree(process)
    except Exception as exc:
        diagnostics.append(f"Process cleanup failed: {type(exc).__name__}: {exc}")
    try:
        with process_lock:
            stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        raw_stdout, raw_stderr = exc.output or "", exc.stderr or ""
        stdout = (
            raw_stdout.decode(errors="replace")
            if isinstance(raw_stdout, bytes)
            else raw_stdout
        )
        stderr = (
            raw_stderr.decode(errors="replace")
            if isinstance(raw_stderr, bytes)
            else raw_stderr
        )
        diagnostics.append("Target output pipes remained open after bounded cleanup")
    if diagnostics:
        stderr += "\n" + "\n".join(diagnostics)
    return stdout, stderr


def run_process(
    command: list[str],
    *,
    timeout: float,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one attempt under a monotonic deadline and reap its process tree.

    Captured output survives timeout and interruption. There are no retries;
    a caller must explicitly retain and account for any additional attempt.
    Text logs decode UTF-8, replacing invalid bytes so diagnostics survive.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Process timeout must be finite and positive")
    started = time.monotonic()
    deadline = started + timeout
    # Constant-size, content-free evidence separates target observation from
    # ownership inspection and post-deadline cleanup. An observed zero exit
    # never overrides a failed census or an exhausted attempt deadline.
    evidence: dict[str, Any] = {
        "target_completed_seconds": None,
        "target_returncode": None,
        "census_seconds": 0.0,
        "census_calls": 0,
        "cleanup_seconds": 0.0,
        "failure_phase": None,
    }
    phase = "launch"

    def refresh_census() -> None:
        nonlocal phase
        phase = "census"
        scan_started = time.monotonic()
        evidence["census_calls"] += 1
        try:
            remaining = deadline - scan_started
            if remaining <= 0:
                raise subprocess.TimeoutExpired("owned process census", timeout)
            census.refresh(timeout=remaining)
        finally:
            evidence["census_seconds"] += time.monotonic() - scan_started

    def capture_failure(error: BaseException) -> None:
        evidence["failure_phase"] = phase
        cleanup_started = time.monotonic()
        try:
            stdout, stderr = _capture_after_cleanup(process, process_lock)
            cast(Any, error).output = stdout
            cast(Any, error).stderr = stderr
        finally:
            evidence["cleanup_seconds"] = time.monotonic() - cleanup_started
            evidence["process_seconds"] = time.monotonic() - started
            cast(Any, error).process_evidence = evidence

    token = uuid.uuid4().hex
    child_env = dict(os.environ if env is None else env)
    inherited = [
        value
        for value in child_env.get(_OWNERSHIP_ENV, "").split(":")
        if re.fullmatch("[0-9a-f]{32}", value)
    ]
    child_env[_OWNERSHIP_ENV] = ":".join([*inherited, token])
    # Construct ownership infrastructure before spawning, so cancellation can
    # never observe a launched process without its census registry entry.
    census = _DescendantCensus(0, token)
    # Spawn and registration share the cancellation snapshot lock. No process
    # can start between a cancellation snapshot and its registry insertion.
    with _ACTIVE_LOCK:
        if _CANCELLED.is_set():
            raise InterruptedError(
                "Execution fleet was cancelled before process launch"
            )
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name == "posix",
        )
        census.pid = process.pid
        _ACTIVE_PROCESSES.add(process)
        _DESCENDANTS[process] = census
        process_lock = threading.RLock()
        _PROCESS_LOCKS[process] = process_lock
    with process:
        try:
            while True:
                phase = "target"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    with process_lock:
                        stdout, stderr = process.communicate(
                            # Bound scan frequency while retaining ownership
                            # identities between pipe polls.
                            timeout=min(0.25, remaining)
                        )
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    refresh_census()
                    continue
                evidence["target_completed_seconds"] = time.monotonic() - started
                evidence["target_returncode"] = process.returncode
                # Census failures propagate once; they are never mistaken for
                # an incomplete pipe poll or retried inside the target loop.
                refresh_census()
                phase = "ownership"
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("owned process liveness", timeout)
                leaks = census.live()
                # A completed observation does not establish timely ownership
                # verification. Keep the original attempt deadline even when
                # the target's zero exit has already been observed.
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("owned process liveness", timeout)
                if leaks:
                    census.kill()
                    raise OwnedProcessLeakError(leaks, stdout, stderr)
                break
        except BaseException as exc:
            capture_failure(exc)
            raise
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.discard(process)
                _DESCENDANTS.pop(process, None)
                _PROCESS_LOCKS.pop(process, None)
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
    evidence["process_seconds"] = time.monotonic() - started
    cast(Any, result).process_evidence = evidence
    if check:
        result.check_returncode()
    return result
