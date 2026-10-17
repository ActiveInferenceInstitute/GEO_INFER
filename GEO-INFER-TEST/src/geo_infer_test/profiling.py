"""Owned, deadline-bounded tracing and Python 3.15 sampling profiles.

Targets execute with the local user's authority. No attachment to unrelated
processes, privilege elevation, implicit fallback, or automatic retry is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import shutil
import sys
import time
from typing import Any
from uuid import uuid4

from .execution import runtime_receipt
from .process import run_process

_RUNTIME = """import json,platform,sys,sysconfig
jit=getattr(sys,'_jit',None)
print(json.dumps(dict(version=sys.version,version_info=list(sys.version_info[:3]),executable=sys.executable,platform=platform.platform(),implementation=platform.python_implementation(),jit_available=bool(jit and jit.is_available()),jit_enabled=bool(jit and jit.is_enabled()),gil_disabled=bool(sysconfig.get_config_var('Py_GIL_DISABLED')))))
"""
_WORKER = """import json,os,runpy,sys,time
from pathlib import Path
folder=Path(__file__).resolve().parent
config=json.loads((folder/'target.json').read_text())
start=time.perf_counter();success=False
try:
    sys.argv=[config['module'] or config['script'],*config['args']]
    sys.path[0]=config['cwd'] if config['module'] else str(Path(config['script']).parent)
    if config['module']:
        runpy.run_module(config['module'],run_name='__main__',alter_sys=True)
    else:
        runpy.run_path(config['script'],run_name='__main__')
    success=True
except SystemExit as error:
    success=error.code is None or error.code == 0
    if not success:raise
finally:
    (folder/'completion.json').write_text(json.dumps(dict(token=config['token'],success=success,pid=os.getpid(),python=sys.version,executable=sys.executable,seconds=time.perf_counter()-start)))
"""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracing_summary(path: Path, limit: int) -> dict[str, Any]:
    stats = pstats.Stats(str(path))
    rows = []
    for (filename, line, function), (
        primitive,
        calls,
        own,
        cumulative,
        _,
    ) in stats.stats.items():
        rows.append(
            dict(
                file=filename,
                line=line,
                function=function,
                primitive_calls=primitive,
                calls=calls,
                own_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    if not rows or not stats.total_calls:
        raise ValueError("Tracing profile contains no executed functions")
    rows.sort(
        key=lambda row: (
            -row["cumulative_seconds"],
            row["file"],
            row["line"],
            row["function"],
        )
    )
    return dict(
        measurement="instrumented function time",
        total_calls=stats.total_calls,
        primitive_calls=stats.prim_calls,
        instrumented_seconds=stats.total_tt,
        hotspots=rows[:limit],
    )


def _sampling_summary(path: Path, limit: int) -> dict[str, Any]:
    stacks = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            stack, count = line.rstrip().rsplit(" ", 1)
            count = int(count)
            if count <= 0 or not stack:
                raise ValueError("Invalid collapsed sample weight")
            stacks.append(dict(stack=stack, thread_samples=count))
    if not stacks:
        raise ValueError("Sampling profile contains no samples")
    stacks.sort(key=lambda row: (-row["thread_samples"], row["stack"]))
    return dict(
        measurement="sampled thread stacks; not function call counts",
        thread_samples=sum(row["thread_samples"] for row in stacks),
        unique_stacks=len(stacks),
        hotspots=stacks[:limit],
    )


def profile_command(
    *,
    output: str | Path,
    script: str | Path | None = None,
    module: str | None = None,
    args: list[str] | None = None,
    python: str | Path | None = None,
    cwd: str | Path | None = None,
    engine: str = "tracing",
    mode: str = "wall",
    sampling_rate: int = 1000,
    timeout: float = 60,
    limit: int = 30,
    max_artifact_bytes: int = 128 * 1024 * 1024,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Profile one actual script/module and retain an immutable receipt directory.

    Sampling requires Python 3.15 and OS memory-inspection permission. Tracing
    works on qualified older interpreters too. Explicit requests never fall
    back. Check result.success: a failed target, timeout, empty profile or source
    drift fails qualification even if the profiler itself returns zero. Profiles
    include instrumentation overhead and are diagnostic, not benchmark claims.
    """
    if bool(script) == bool(module):
        raise ValueError("Supply exactly one of script or module")
    if module is not None and (
        not isinstance(module, str)
        or not all(p.isidentifier() for p in module.split("."))
    ):
        raise ValueError("module must be a dotted Python module name")
    if engine not in {"tracing", "sampling"} or mode not in {
        "wall",
        "cpu",
        "gil",
        "exception",
    }:
        raise ValueError("Invalid profiling engine or sampling mode")
    if engine == "tracing" and mode != "wall":
        raise ValueError(
            "Tracing measures instrumented wall time; sampling is required for cpu/gil/exception modes"
        )
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    for name, value, maximum in [
        ("sampling_rate", sampling_rate, 100000),
        ("limit", limit, 200),
        ("max_artifact_bytes", max_artifact_bytes, 2**40),
    ]:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    if args is not None and (
        not isinstance(args, list) or any(not isinstance(v, str) for v in args)
    ):
        raise ValueError("args must be an array of strings")
    if env is not None and (
        not isinstance(env, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
    ):
        raise ValueError("env must map strings to strings")
    directory = Path(cwd or Path.cwd()).resolve()
    if not directory.is_dir():
        raise ValueError("cwd must be an existing directory")
    target = (directory / script).resolve() if script is not None else None
    if target is not None and not target.is_file():
        raise ValueError("script must be an existing file")
    executable = shutil.which(str(python or sys.executable))
    if executable is None:
        raise ValueError("Python executable was not found")
    provided_output = Path(output)
    if provided_output.exists() or provided_output.is_symlink():
        raise FileExistsError(provided_output)
    destination = provided_output.resolve()
    environment = {**os.environ, **(env or {})}
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Profile deadline exceeded")
        return seconds

    probe = run_process(
        [executable, "-c", _RUNTIME],
        cwd=directory,
        env=environment,
        timeout=min(10, remaining()),
    )
    if probe.returncode:
        raise RuntimeError("Python runtime probe failed: " + probe.stderr[-2000:])
    runtime = json.loads(probe.stdout)
    if engine == "sampling" and runtime["version_info"][:2] < [3, 15]:
        raise ValueError("Sampling requires Python 3.15 or later")
    root_probe = run_process(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "--show-toplevel"],
        cwd=directory,
        env=environment,
        timeout=min(5, remaining()),
    )
    custody_root = (
        Path(root_probe.stdout.strip()).resolve()
        if root_probe.returncode == 0
        else directory
    )
    before = runtime_receipt(root=custody_root, timeout=min(10, remaining()))
    if before.get("custody_complete") and destination.is_relative_to(custody_root):
        ignored = run_process(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "check-ignore",
                "--quiet",
                "--no-index",
                str(destination / "receipt.json"),
            ],
            cwd=custody_root,
            env=environment,
            timeout=min(5, remaining()),
        )
        if ignored.returncode:
            raise ValueError(
                "Use an output directory outside the checkout or Git-ignored, e.g. .geo-infer-test-results/profiles/name"
            )
    source_hash = _hash(target) if target is not None else None
    destination.mkdir(parents=True, exist_ok=False)
    token = uuid4().hex
    configuration = dict(
        token=token,
        module=module,
        script=str(target) if target else None,
        args=args or [],
        cwd=str(directory),
    )
    (destination / "target.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    worker = destination / "profile_target.py"
    worker.write_text(_WORKER, encoding="utf-8")
    artifact = destination / (
        "profile.collapsed" if engine == "sampling" else "profile.pstats"
    )
    if engine == "sampling":
        command = [
            executable,
            "-m",
            "profiling.sampling",
            "run",
            "--collapsed",
            "--native",
            "--all-threads",
            "--mode",
            mode,
            "-r",
            str(sampling_rate),
            "-o",
            str(artifact),
            str(worker),
        ]
    else:
        backend = (
            "profiling.tracing"
            if runtime["version_info"][:2] >= [3, 15]
            else "cProfile"
        )
        command = [executable, "-m", backend, "-o", str(artifact), str(worker)]
    receipt = dict(
        schema="geo-infer-profile/1",
        controller=dict(python=sys.version, executable=sys.executable),
        runtime=runtime,
        engine=engine,
        mode=mode,
        sampling_rate=sampling_rate if engine == "sampling" else None,
        command=command,
        target=configuration,
        source_before=before,
        source_root=str(custody_root),
        timeout=timeout,
        success=False,
        status="failed",
        failures=[],
    )
    start = time.monotonic()
    stdout = stderr = ""
    try:
        result = run_process(
            command, cwd=directory, env=environment, timeout=remaining()
        )
        stdout, stderr = result.stdout, result.stderr
        receipt["returncode"] = result.returncode
        if result.returncode:
            receipt["failures"].append(f"Profiler exited {result.returncode}")
        completion = json.loads((destination / "completion.json").read_text())
        if (
            completion.get("token") != token
            or not completion.get("success")
            or type(completion.get("pid")) is not int
            or completion["pid"] <= 0
        ):
            raise ValueError("Target completion receipt failed validation")
        receipt["completion"] = completion
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size > max_artifact_bytes
        ):
            raise ValueError("Missing, unsafe or oversized profiling artifact")
        receipt["summary"] = (
            _sampling_summary if engine == "sampling" else _tracing_summary
        )(artifact, limit)
    except (Exception, SystemExit) as error:
        stdout = getattr(error, "output", None) or stdout
        stderr = getattr(error, "stderr", None) or stderr
        receipt["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - start
        (destination / "stdout.log").write_text(
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout,
            encoding="utf-8",
        )
        (destination / "stderr.log").write_text(
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr,
            encoding="utf-8",
        )
        try:
            after = runtime_receipt(root=custody_root, timeout=10)
        except Exception as error:
            after = {
                "custody_complete": False,
                "error": f"{type(error).__name__}: {error}",
            }
        if before.get("custody_complete") and not after.get("custody_complete"):
            receipt["failures"].append("Final source custody could not be established")
        receipt["source_after"] = after
        bound = bool(before.get("custody_complete") and after.get("custody_complete"))
        receipt["source_bound"] = bound
        receipt["source_unchanged"] = (
            all(
                before.get(k) == after.get(k)
                for k in ("revision", "dirty", "dirty_sha256", "lock_sha256")
            )
            if bound
            else None
        )
        try:
            target_changed = bool(
                target and (not target.is_file() or _hash(target) != source_hash)
            )
        except OSError as error:
            target_changed = True
            receipt["failures"].append(f"Target source audit failed: {error}")
        if receipt["source_unchanged"] is False or target_changed:
            receipt["failures"].append(
                "Target or checkout source changed during profiling"
            )
        receipt["target_source_sha256"] = source_hash
        receipt["artifacts"] = {
            p.name: _hash(p)
            for p in sorted(destination.iterdir())
            if p.is_file() and not p.is_symlink()
        }
        receipt["success"] = not receipt["failures"]
        receipt["status"] = "passed" if receipt["success"] else "failed"
        receipt["receipt"] = str(destination / "receipt.json")
        (destination / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Machine-readable profile CLI; code runs once and failures retain evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--script", type=Path)
    selection.add_argument("--module")
    parser.add_argument("--python")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["tracing", "sampling"], default="tracing")
    parser.add_argument(
        "--mode", choices=["wall", "cpu", "gil", "exception"], default="wall"
    )
    parser.add_argument("--sampling-rate", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = vars(parser.parse_args(argv))
    options["args"] = (
        options["args"][1:] if options["args"][:1] == ["--"] else options["args"]
    )
    try:
        result = profile_command(**options)
    except (Exception, SystemExit) as error:
        print(
            json.dumps(
                dict(
                    success=False,
                    error=dict(type=type(error).__name__, message=str(error)),
                )
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
