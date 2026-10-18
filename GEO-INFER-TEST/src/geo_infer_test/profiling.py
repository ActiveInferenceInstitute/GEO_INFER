"""Owned, deadline-bounded tracing and Python 3.15 sampling profiles.

Targets execute with the local user's authority. No attachment to unrelated
processes, privilege elevation, implicit fallback, or automatic retry is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import marshal
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


def _tracing_summary(
    path: Path,
    limit: int,
    *,
    sort_by: str = "cumulative",
    verified_data: bytes | None = None,
    call_graph: bool = False,
) -> dict[str, Any]:
    if verified_data is None:
        stats = pstats.Stats(str(path))
    else:
        # Load the exact bytes whose sealed hash was checked, rather than
        # reopening a path that could have changed between verification/read.
        stats = pstats.Stats()
        stats.stats = marshal.loads(verified_data)
        stats.get_top_level_stats()
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
                own_percent=100 * own / stats.total_tt if stats.total_tt else 0.0,
            )
        )
    if not rows or not stats.total_calls:
        raise ValueError("Tracing profile contains no executed functions")
    metric = {
        "cumulative": "cumulative_seconds",
        "own": "own_seconds",
        "calls": "calls",
    }[sort_by]
    hotspots = heapq.nsmallest(
        limit,
        rows,
        key=lambda row: (
            -row[metric],
            row["file"],
            row["line"],
            row["function"],
        ),
    )
    summary = dict(
        measurement="instrumented function time",
        ranking=sort_by,
        function_count=len(rows),
        total_calls=stats.total_calls,
        primitive_calls=stats.prim_calls,
        instrumented_seconds=stats.total_tt,
        hotspots=hotspots,
    )
    if call_graph:
        edges = []
        for row in hotspots:
            callee = (row["file"], row["line"], row["function"])
            for caller, measurements in stats.stats[callee][4].items():
                # Native tracing uses (primitive, total, own, cumulative).
                # The older profile module may store an integer call count.
                if isinstance(measurements, tuple):
                    primitive, calls, own, cumulative = measurements
                else:
                    primitive = calls = measurements
                    own = cumulative = None
                edges.append(
                    dict(
                        caller=dict(zip(("file", "line", "function"), caller)),
                        callee=dict(zip(("file", "line", "function"), callee)),
                        primitive_calls=primitive,
                        calls=calls,
                        own_seconds=own,
                        cumulative_seconds=cumulative,
                    )
                )
        summary["call_graph"] = dict(
            measurement="caller edges into reported hotspots; cumulative times overlap",
            total_candidate_edges=len(edges),
            edges=heapq.nsmallest(
                limit,
                edges,
                key=lambda edge: (
                    -(edge["cumulative_seconds"] or 0),
                    edge["caller"]["file"],
                    edge["caller"]["line"],
                    edge["callee"]["file"],
                    edge["callee"]["line"],
                ),
            ),
        )
    return summary


def _sampling_summary(
    path: Path, limit: int, *, verified_data: bytes | None = None
) -> dict[str, Any]:
    stacks: Counter[str] = Counter()
    if verified_data is not None:
        lines = verified_data.decode("utf-8").splitlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stack, count = line.rstrip().rsplit(" ", 1)
        count = int(count)
        if count <= 0 or not stack:
            raise ValueError("Invalid collapsed sample weight")
        stacks[stack] += count
    if not stacks:
        raise ValueError("Sampling profile contains no samples")
    total = sum(stacks.values())
    return dict(
        measurement="sampled thread stacks; not function call counts",
        ranking="samples",
        thread_samples=total,
        unique_stacks=len(stacks),
        hotspots=[
            dict(stack=stack, thread_samples=count, sample_percent=100 * count / total)
            for stack, count in heapq.nsmallest(
                limit, stacks.items(), key=lambda row: (-row[1], row[0])
            )
        ],
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
    interrupted = False
    try:
        result = run_process(
            command, cwd=directory, env=environment, timeout=remaining()
        )
        stdout, stderr = result.stdout, result.stderr
        receipt["process_evidence"] = getattr(result, "process_evidence", {})
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
    except BaseException as error:
        stdout = getattr(error, "output", None) or stdout
        stderr = getattr(error, "stderr", None) or stderr
        receipt["process_evidence"] = getattr(
            error, "process_evidence", receipt.get("process_evidence", {})
        )
        receipt["failures"].append(f"{type(error).__name__}: {error}")
        interrupted = not isinstance(error, (Exception, SystemExit))
        if interrupted:
            # run_process has already drained and reaped its owned children.
            # Retain a failed receipt before preserving native interruption.
            raise
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
        receipt["status"] = (
            "interrupted"
            if interrupted
            else "passed"
            if receipt["success"]
            else "failed"
        )
        receipt["receipt"] = str(destination / "receipt.json")
        (destination / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return receipt


def summarize_profile(
    receipt: str | Path,
    *,
    limit: int = 30,
    sort_by: str = "auto",
    max_artifact_bytes: int = 128 * 1024 * 1024,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a trusted local profile receipt without executing its target again.

    Every listed artifact must be a regular file inside the receipt directory
    with its original SHA256. Verified tracing bytes are parsed only after all
    hashes match. This is integrity checking for receipts produced by this API,
    not a sandbox for arbitrary profiling files or attacker-authored manifests.
    Supply ``expected_receipt_sha256`` from independently retained evidence to
    authenticate the manifest too; an unanchored local manifest is trusted input.
    A report preserves its original success/failure and historical source
    custody; reading old evidence does not qualify the current checkout.
    Failed or interrupted attempts can end before a profile is written. Their
    verified reports retain the failure evidence with ``profile_available``
    false and ``summary`` None; successful attempts always require a profile.

    Tracing ranks by cumulative time, own time or calls and returns at most
    ``limit`` caller edges into its hotspots. Cumulative times overlap and
    cannot be summed as percentages. Sampling ranks collapsed stack weights;
    its percentages describe sampled thread stacks, never function call counts.
    """
    for name, value, maximum in [
        ("limit", limit, 200),
        ("max_artifact_bytes", max_artifact_bytes, 2**40),
    ]:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    provided = Path(receipt)
    if provided.is_dir():
        provided = provided / "receipt.json"
    if provided.is_symlink() or not provided.is_file():
        raise ValueError("receipt must be an existing regular receipt file")
    receipt_path = provided.resolve()
    with receipt_path.open("rb") as stream:
        raw_receipt = stream.read(max_artifact_bytes + 1)
    if len(raw_receipt) > max_artifact_bytes:
        raise ValueError("Profile receipt exceeds max_artifact_bytes")
    if expected_receipt_sha256 is not None and (
        not isinstance(expected_receipt_sha256, str)
        or hashlib.sha256(raw_receipt).hexdigest() != expected_receipt_sha256
    ):
        raise ValueError(
            "Profile receipt SHA256 does not match independently retained hash"
        )
    payload = json.loads(raw_receipt)
    if (
        payload.get("schema") != "geo-infer-profile/1"
        or type(payload.get("success")) is not bool
        or payload.get("status") not in {"passed", "failed", "interrupted"}
        or payload["success"] != (payload.get("status") == "passed")
        or payload.get("engine") not in {"tracing", "sampling"}
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise ValueError("Invalid owned profiling receipt")
    if payload["success"] and (
        payload.get("status") != "passed"
        or type(payload.get("returncode")) is not int
        or payload["returncode"] != 0
        or payload.get("failures") != []
        or not payload.get("completion", {}).get("success")
        or "summary" not in payload
        or payload.get("source_unchanged") is False
    ):
        raise ValueError("Successful receipt lacks valid completion/source evidence")
    if not payload["success"] and (
        not isinstance(payload.get("failures"), list)
        or not payload["failures"]
        or any(
            not isinstance(reason, str) or not reason for reason in payload["failures"]
        )
    ):
        raise ValueError("Failed receipt lacks a failure disposition")
    engine = payload["engine"]
    ranking = (
        ("cumulative" if engine == "tracing" else "samples")
        if sort_by == "auto"
        else sort_by
    )
    if ranking not in (
        {"cumulative", "own", "calls"} if engine == "tracing" else {"samples"}
    ):
        raise ValueError("sort_by is incompatible with the profiling engine")
    artifact_name = "profile.pstats" if engine == "tracing" else "profile.collapsed"
    verified_data = None
    sealed_metadata = {}
    actual_files = {
        path.name
        for path in receipt_path.parent.iterdir()
        if path.is_file() or path.is_symlink()
    } - {receipt_path.name}
    if actual_files != set(payload["artifacts"]):
        raise ValueError("Unsealed or missing files in the profile artifact directory")
    for name, expected_hash in payload["artifacts"].items():
        if (
            not name
            or name in {".", "..", "receipt.json"}
            or "/" in name
            or "\\" in name
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError("Unsafe profile artifact manifest")
        artifact = receipt_path.parent / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("Missing or unsafe profile artifact: " + name)
        with artifact.open("rb") as stream:
            content = stream.read(max_artifact_bytes + 1)
        if len(content) > max_artifact_bytes:
            raise ValueError("Profile artifact exceeds max_artifact_bytes: " + name)
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Profile artifact hash mismatch: " + name)
        if name == artifact_name:
            verified_data = content
        elif name in {"completion.json", "target.json"}:
            sealed_metadata[name] = content
    if verified_data is None and payload["success"]:
        raise ValueError("Receipt has no sealed profiling artifact")
    if payload["success"]:
        required = {
            "completion.json",
            "target.json",
            "profile_target.py",
            "stdout.log",
            "stderr.log",
        }
        if not required.issubset(payload["artifacts"]):
            raise ValueError(
                "Successful receipt lacks sealed target/completion artifacts"
            )
        completion = json.loads(sealed_metadata["completion.json"])
        target = json.loads(sealed_metadata["target.json"])
        runtime = payload["runtime"]
        if (
            completion != payload["completion"]
            or target != payload.get("target")
            or completion.get("token") != target.get("token")
            or completion.get("success") is not True
            or type(completion.get("pid")) is not int
            or completion["pid"] <= 0
            or completion.get("python") != runtime.get("version")
            or completion.get("executable") != runtime.get("executable")
        ):
            raise ValueError(
                "Sealed completion does not match the owned target/runtime"
            )
        before, after = (
            payload.get("source_before", {}),
            payload.get("source_after", {}),
        )
        bound = bool(before.get("custody_complete") and after.get("custody_complete"))
        unchanged = (
            all(
                before.get(key) == after.get(key)
                for key in ("revision", "dirty", "dirty_sha256", "lock_sha256")
            )
            if bound
            else None
        )
        if (
            payload.get("source_bound") is not bound
            or payload.get("source_unchanged") is not unchanged
        ):
            raise ValueError(
                "Receipt source custody does not match its sealed disposition"
            )
    summary = (
        None
        if verified_data is None
        else (
            _tracing_summary(
                receipt_path.parent / artifact_name,
                limit,
                sort_by=ranking,
                verified_data=verified_data,
                call_graph=True,
            )
            if engine == "tracing"
            else _sampling_summary(
                receipt_path.parent / artifact_name, limit, verified_data=verified_data
            )
        )
    )
    return dict(
        schema="geo-infer-profile-report/1",
        receipt=str(receipt_path),
        receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),
        success=payload["success"],
        status=payload["status"],
        failures=payload.get("failures", []),
        returncode=payload.get("returncode"),
        completion=payload.get("completion"),
        elapsed_seconds=payload.get("elapsed_seconds"),
        timeout=payload.get("timeout"),
        profile_available=verified_data is not None,
        engine=engine,
        mode=payload.get("mode"),
        runtime=payload["runtime"],
        source_bound=payload.get("source_bound"),
        source_unchanged=payload.get("source_unchanged"),
        source_before=payload.get("source_before"),
        source_after=payload.get("source_after"),
        process_evidence=payload.get("process_evidence", {}),
        artifacts_verified=len(payload["artifacts"]),
        limit=limit,
        summary=summary,
    )


def main(argv: list[str] | None = None) -> int:
    """Machine-readable profile CLI; code runs once and failures retain evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--script", type=Path)
    selection.add_argument("--module")
    selection.add_argument(
        "--report",
        type=Path,
        help="read and verify a saved receipt without target execution",
    )
    parser.add_argument("--python")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=["tracing", "sampling"])
    parser.add_argument("--mode", choices=["wall", "cpu", "gil", "exception"])
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--sort-by",
        choices=["auto", "cumulative", "own", "calls", "samples"],
        default="auto",
    )
    parser.add_argument("--max-artifact-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument(
        "--receipt-sha256",
        help="independently retained receipt SHA256; requires --report",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = vars(parser.parse_args(argv))
    options["args"] = (
        options["args"][1:] if options["args"][:1] == ["--"] else options["args"]
    )
    report = options.pop("report")
    sort_by = options.pop("sort_by")
    expected_receipt_sha256 = options.pop("receipt_sha256")
    if report is not None:
        if (
            any(
                options[name] is not None
                for name in (
                    "python",
                    "cwd",
                    "output",
                    "engine",
                    "mode",
                    "sampling_rate",
                    "timeout",
                )
            )
            or options["args"]
        ):
            parser.error("--report cannot be combined with target execution options")
    elif (
        options["output"] is None
        or sort_by != "auto"
        or expected_receipt_sha256 is not None
    ):
        parser.error("target execution requires --output; --sort-by requires --report")
    else:
        for name, default in [
            ("engine", "tracing"),
            ("mode", "wall"),
            ("sampling_rate", 1000),
            ("timeout", 60),
        ]:
            if options[name] is None:
                options[name] = default
    try:
        result = (
            summarize_profile(
                report,
                limit=options["limit"],
                sort_by=sort_by,
                max_artifact_bytes=options["max_artifact_bytes"],
                expected_receipt_sha256=expected_receipt_sha256,
            )
            if report is not None
            else profile_command(**options)
        )
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
