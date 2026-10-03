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


class OwnedProcessLeakError(subprocess.SubprocessError):
    """A nominally successful command retained owned background processes."""

    def __init__(self, pids: list[int], stdout: str, stderr: str):
        super().__init__(f"Command left owned background processes running: {pids}")
        self.output = stdout
        self.stderr = stderr


class ProcessCensusError(subprocess.SubprocessError):
    """Ownership inspection failed independently of the target deadline."""


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

    def _owned(self, process: Any) -> bool:
        try:
            return self.token in process.environ().get(_OWNERSHIP_ENV, "").split(":")
        except self.psutil.NoSuchProcess:
            return False

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
                try:
                    listing = subprocess.run(
                        ["ps", "eww", "-axo", "pid=,command="],
                        capture_output=True,
                        text=True,
                        errors="replace",
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
                        child = self.psutil.Process(pid)
                        child.create_time()
                        if self._owned(child):
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
    deadline = time.monotonic() + timeout
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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    with process_lock:
                        stdout, stderr = process.communicate(
                            # A native ownership census reads the complete
                            # process listing. Bound its frequency under module
                            # concurrency while still scanning immediately on
                            # exit and during deadline cleanup.
                            timeout=min(0.25, remaining)
                        )
                    census.refresh(timeout=max(0.001, deadline - time.monotonic()))
                    leaks = census.live()
                    if leaks:
                        census.kill()
                        raise OwnedProcessLeakError(leaks, stdout, stderr)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    # Tiny commands need one final ownership scan, while a
                    # running command retains identities between pipe polls.
                    census.refresh(timeout=deadline - time.monotonic())
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _capture_after_cleanup(process, process_lock)
            cast(Any, exc).output = stdout
            cast(Any, exc).stderr = stderr
            raise
        except BaseException as exc:
            stdout, stderr = _capture_after_cleanup(process, process_lock)
            cast(Any, exc).output = stdout
            cast(Any, exc).stderr = stderr
            raise
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.discard(process)
                _DESCENDANTS.pop(process, None)
                _PROCESS_LOCKS.pop(process, None)
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
    if check:
        result.check_returncode()
    return result
