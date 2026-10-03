#!/usr/bin/env python3
"""
Multi-package wheel build driver for the GEO-INFER monorepo.

ARCH-01: Builds a wheel for every ``GEO-INFER-*`` package, verifies that each
built wheel belongs to the correct ``geo-infer-*`` distribution namespace, and
(optionally) installs each wheel into an isolated virtual environment to smoke
test importability and configuration-resource discovery.

Intended to be invoked by ``.github/workflows/release.yml`` after the shared
uv workspace is synced. Archive-contract checks and installed-import checks
are separate so small fixture wheels can exercise failure paths independently.
"""

from __future__ import annotations

import argparse
import importlib.util
from email.parser import BytesParser
from fnmatch import fnmatchcase
import zipfile
import os
import math
import subprocess
import sys
import tempfile
import json
import hashlib
import time
import uuid
import shutil
from datetime import datetime, UTC
from dataclasses import dataclass, field
from pathlib import Path

from import_probe import run_import_probe, run_process
from validate_packaging import (
    distribution_name,
    module_dirs,
    parse_pyproject,
    wheel_filename_is_valid,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
runtime_receipt = importlib.import_module("geo_infer_test.execution").runtime_receipt
validate_wheel_receipt = importlib.import_module(
    "geo_infer_test.wheel_evidence"
).validate_wheel_receipt

# This CLI also runs in the stdlib-only build/import portability lane.
_profile_spec = importlib.util.spec_from_file_location(
    "geo_infer_test_wheel_profiles_standalone",
    Path(__file__).parent / "src" / "geo_infer_test" / "wheel_profiles.py",
)
assert _profile_spec is not None and _profile_spec.loader is not None
_profiles = importlib.util.module_from_spec(_profile_spec)
sys.modules[_profile_spec.name] = _profiles
_profile_spec.loader.exec_module(_profiles)
REQUIRED_PROFILES = _profiles.REQUIRED_PROFILES


@dataclass
class BuildResult:
    module: str
    distribution: str | None
    wheel: Path | None = None
    ok: bool = False
    error: str = ""
    returncode: int | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class BuildSummary:
    results: list[BuildResult] = field(default_factory=list)

    @property
    def namespaces_valid(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    @property
    def failures(self) -> list[BuildResult]:
        return [r for r in self.results if not r.ok]


def validate_wheel_contents(wheel: Path, module_dir: Path) -> None:
    """Compare built metadata and package resources with their source contracts."""
    config = parse_pyproject(module_dir)
    project = config.get("project", {})
    expected_name = project.get("name")
    expected_version = project.get("version")
    package_name = str(expected_name).replace("-", "_")
    source = module_dir / "src"
    package = source / package_name
    if not (package / "__init__.py").is_file():
        raise ValueError("Expected source package is missing its __init__.py")
    suffixes = {".py", ".json", ".geojson", ".yaml", ".yml", ".md", ".txt"}
    package_data = config.get("tool", {}).get("setuptools", {}).get("package-data", {})
    expected_resources = set()
    for path in package.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        include = path.suffix in suffixes
        for owner, patterns in package_data.items():
            owners = (
                [
                    package,
                    *[
                        parent
                        for parent in path.parents
                        if parent != package and package in parent.parents
                    ],
                ]
                if owner == "*"
                else [source / owner.replace(".", "/")]
            )
            for base in owners:
                if path.is_relative_to(base) and any(
                    fnmatchcase(path.relative_to(base).as_posix(), pattern)
                    for pattern in patterns
                ):
                    include = True
        if include:
            expected_resources.add(relative)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_paths = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(
                "Wheel must contain exactly one distribution metadata record"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata.get("Name", "").lower().replace("_", "-") != expected_name:
            raise ValueError("Wheel metadata name disagrees with pyproject.toml")
        if metadata.get("Version") != expected_version:
            raise ValueError("Wheel metadata version disagrees with pyproject.toml")
        missing = sorted(expected_resources - names)
        if missing:
            raise ValueError(
                "Wheel omits source package resources: " + ", ".join(missing)
            )
        for resource in expected_resources:
            if archive.read(resource) != (source / resource).read_bytes():
                raise ValueError("Wheel resource differs from source: " + resource)


def build_wheel(
    module_dir: Path,
    outdir: Path,
    python: list[str],
    *,
    evidence_dir: Path | None = None,
) -> BuildResult:
    """Build in a fresh directory so stale neighboring wheels cannot be selected."""
    distribution = distribution_name(parse_pyproject(module_dir))
    result = BuildResult(module=module_dir.name, distribution=distribution)
    if not distribution:
        result.error = "missing [project].name"
        return result
    outdir.mkdir(parents=True, exist_ok=True)
    evidence_root = evidence_dir or outdir
    attempt = evidence_root / "builds" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    started = time.monotonic()
    stdout, stderr = "", ""
    try:
        with tempfile.TemporaryDirectory(dir=outdir) as temporary:
            completed = run_process(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--python",
                    python[0],
                    "--out-dir",
                    temporary,
                    str(module_dir),
                ],
                check=True,
                timeout=300,
                cwd=Path.cwd(),
            )
            stdout, stderr = completed.stdout, completed.stderr
            result.returncode = completed.returncode
            wheels = list(Path(temporary).glob("*.whl"))
            if len(wheels) != 1 or not wheel_filename_is_valid(
                wheels[0].name, distribution
            ):
                result.error = (
                    "build did not produce exactly one wheel in the expected namespace"
                )
                return result
            validate_wheel_contents(wheels[0], module_dir)
            result.wheel = outdir / wheels[0].name
            wheels[0].replace(result.wheel)
            result.ok = True
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        stdout = getattr(exc, "output", None) or stdout
        stderr = getattr(exc, "stderr", None) or stderr
        result.returncode = getattr(exc, "returncode", None)
        detail = getattr(exc, "stderr", None) or str(exc)
        result.error = (
            detail.decode(errors="replace") if isinstance(detail, bytes) else detail
        )[-2000:]
    finally:
        result.duration_seconds = time.monotonic() - started
        for stream, content in (("stdout", stdout), ("stderr", stderr)):
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            artifact = attempt / f"{stream}.log"
            artifact.write_text(content or "", encoding="utf-8")
            result.artifacts[str(artifact.relative_to(evidence_root))] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
    return result


def verify_wheels(
    wheels: list[Path],
    python: list[str],
    *,
    import_timeout: float = 120,
    extras: tuple[str, ...] = (),
    probe_code: str = "",
    receipts: list[dict] | None = None,
    evidence_dir: Path | None = None,
) -> list[dict]:
    """Install wheels in a clean environment and execute actual import/resource probes."""
    if not wheels:
        raise ValueError("At least one wheel is required")
    if not math.isfinite(import_timeout) or import_timeout <= 0:
        raise ValueError("Import timeout must be finite and positive")
    collected = receipts if receipts is not None else []
    wheels = [wheel.resolve() for wheel in wheels]
    if extras and len(wheels) != 1:
        raise ValueError("Extra verification requires exactly one target wheel")
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
    ):
        env.pop(key, None)
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        env[key] = "1"
    with tempfile.TemporaryDirectory(prefix="geo-infer-wheel-check-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        run_process(
            ["uv", "venv", "--python", python[0], str(environment)],
            check=True,
            env=env,
            timeout=120,
            cwd=root,
        )
        executable = str(
            environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        repository = Path(__file__).resolve().parent.parent
        constraints = root / "constraints.txt"
        run_process(
            [
                "uv",
                "export",
                "--locked",
                "--all-packages",
                "--all-extras",
                "--no-emit-workspace",
                "--no-hashes",
                "--format",
                "requirements-txt",
                "--output-file",
                str(constraints),
            ],
            cwd=repository,
            env=env,
            check=True,
            timeout=120,
        )
        run_process(
            [
                "uv",
                "pip",
                "install",
                "--python",
                executable,
                "--constraint",
                str(constraints),
                "--find-links",
                str(wheels[0].parent),
                *[
                    str(wheel) + ("[" + ",".join(extras) + "]" if extras else "")
                    for wheel in wheels
                ],
            ],
            cwd=root,
            env=env,
            check=True,
            timeout=600,
        )
        probe = """import faulthandler, importlib, importlib.metadata, importlib.resources, json, pathlib, sys, sysconfig
faulthandler.dump_traceback_later(min(float(sys.argv[2]) / 2, 30))
name = sys.argv[1]
package = importlib.import_module(name)
site = pathlib.Path(sysconfig.get_paths()['purelib']).resolve()
origin = pathlib.Path(package.__file__).resolve()
assert origin.is_relative_to(site), (origin, site)
dist = importlib.metadata.distribution(name.replace('_','-'))
if hasattr(package, '__version__'):
    assert package.__version__ == dist.version, (package.__version__, dist.version)
resources = []
for path in dist.files or []:
    if str(path).startswith(name + '/') and path.suffix in ('.json','.geojson','.yaml','.yml'):
        target = pathlib.Path(dist.locate_file(path))
        assert target.is_file(), target
        target.read_bytes()
        resources.append(str(path))
# EXTRA_PROBE
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'package':name,'profile':sys.argv[3],'version':dist.version,'origin':str(origin),'resources':resources,'probe_token':sys.argv[-1],'status':'ok'}))
"""
        probe = probe.replace("# EXTRA_PROBE", probe_code)
        for wheel in wheels:
            package = wheel.name.split("-")[0]
            command = [
                executable,
                "-I",
                "-c",
                probe,
                package,
                str(import_timeout),
                ",".join(extras) or "base",
            ]
            started = time.monotonic()
            stdout = stderr = ""
            probe_receipt = {
                "package": package,
                "kind": "operation" if probe_code else "import",
                "profile": ",".join(extras) or "base",
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "probe_sha256": hashlib.sha256(probe_code.encode()).hexdigest(),
                "status": "failed",
                "returncode": None,
            }
            try:
                result = run_import_probe(
                    command, package=package, cwd=root, env=env, timeout=import_timeout
                )
                stdout, stderr = result.stdout, result.stderr
                completed = json.loads(stdout.strip().splitlines()[-1])
                if (
                    completed.get("package") != package
                    or completed.get("status") != "ok"
                ):
                    raise ValueError(
                        "Installed probe did not finish with its validated receipt"
                    )
                probe_receipt.update(completed, returncode=result.returncode)
            except BaseException as exc:
                stdout = getattr(exc, "output", None) or stdout
                stderr = getattr(exc, "stderr", None) or stderr
                probe_receipt["returncode"] = getattr(exc, "returncode", None)
                probe_receipt["diagnostics"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                probe_receipt["duration_seconds"] = time.monotonic() - started
                if evidence_dir is not None:
                    attempt = evidence_dir / uuid.uuid4().hex
                    attempt.mkdir(parents=True)
                    artifacts = {}
                    for name, content in (
                        ("stdout.log", stdout),
                        ("stderr.log", stderr),
                    ):
                        if isinstance(content, bytes):
                            content = content.decode(errors="replace")
                        path = attempt / name
                        path.write_text(content, encoding="utf-8")
                        artifacts[str(path.relative_to(evidence_dir.parent))] = (
                            hashlib.sha256(path.read_bytes()).hexdigest()
                        )
                    probe_receipt["artifacts"] = artifacts
                collected.append(probe_receipt)
            print(result.stdout.strip(), flush=True)
    return collected


def verify_profiles(
    wheels: list[Path],
    python: list[str],
    *,
    import_timeout: float = 120,
    receipts: list[dict] | None = None,
    evidence_dir: Path | None = None,
) -> None:
    """Verify each declared operation profile in a separate clean installation."""
    targets = {wheel.name.split("-")[0]: wheel for wheel in wheels}
    for profile in REQUIRED_PROFILES:
        if profile.package not in targets:
            raise ValueError("Missing wheel for required profile: " + profile.name)
        print("Verifying isolated profile " + profile.name, flush=True)
        verify_wheels(
            [targets[profile.package]],
            python,
            import_timeout=import_timeout,
            extras=profile.extras,
            probe_code=profile.code,
            receipts=receipts,
            evidence_dir=evidence_dir,
        )


def install_and_verify(wheel: Path, python: list[str]) -> None:
    """Verify one wheel with the same isolation and resource contract as a release."""
    verify_wheels([wheel], python)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate GEO-INFER wheels")
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument("--verify", action="store_true", help="isolated venv install")
    parser.add_argument(
        "--verify-extras",
        action="store_true",
        help="also verify required independent base/extra operation profiles",
    )
    parser.add_argument(
        "--import-timeout",
        type=float,
        default=120,
        help="Seconds allowed per installed package import (default 120)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.import_timeout) or args.import_timeout <= 0:
        parser.error("--import-timeout must be finite and positive")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    attempt = outdir / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    started = time.monotonic()
    summary = BuildSummary()
    receipts: list[dict] = []
    before: dict = {}
    after: dict = {}
    diagnostics = []
    failure_artifacts = {}
    returncode = 1
    modules = module_dirs()
    try:
        before = runtime_receipt(timeout=30)
        if not before.get("custody_complete"):
            raise OSError("Wheel source custody is unavailable")
        returncode = _perform_build(args, outdir, modules, summary, receipts, attempt)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostics.append(f"{type(exc).__name__}: {exc}")
        for stream, value in (
            ("stdout", getattr(exc, "output", "")),
            ("stderr", getattr(exc, "stderr", "")),
        ):
            content = value or ""
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            artifact = attempt / f"failure.{stream}.log"
            artifact.write_text(str(content), encoding="utf-8")
            failure_artifacts[artifact.name] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
        print("Wheel verification failed: " + diagnostics[-1], file=sys.stderr)
    finally:
        try:
            after = runtime_receipt(timeout=30)
            if not after.get("custody_complete"):
                raise OSError("Final wheel source custody is unavailable")
            if any(
                before.get(key) != after.get(key)
                for key in ("revision", "dirty", "dirty_sha256", "lock_sha256")
            ):
                raise ValueError("Source or lock changed during wheel verification")
        except Exception as exc:
            diagnostics.append(f"{type(exc).__name__}: {exc}")
            returncode = 1
        expected_imports = len(modules) if args.verify or args.verify_extras else 0
        expected_operations = len(REQUIRED_PROFILES) if args.verify_extras else 0
        completed_imports = sum(
            row["kind"] == "import" and row["status"] == "ok" for row in receipts
        )
        completed_operations = sum(
            row["kind"] == "operation" and row["status"] == "ok" for row in receipts
        )
        if returncode == 0 and (
            completed_imports != expected_imports
            or completed_operations != expected_operations
        ):
            diagnostics.append("Installed probe inventory is incomplete")
            returncode = 1
        record = {
            "schema_version": "geo-infer-wheel-receipt/1",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "passed" if returncode == 0 else "failed",
            "returncode": returncode,
            "duration_seconds": time.monotonic() - started,
            "source": before,
            "final_source": after,
            "selected": {
                "wheels": len(modules),
                "imports": expected_imports,
                "operations": expected_operations,
            },
            "executed": {
                "wheels": sum(result.ok for result in summary.results),
                "imports": completed_imports,
                "operations": completed_operations,
            },
            "diagnostics": diagnostics,
            "failure_artifacts": failure_artifacts,
            "wheels": [
                {
                    "module": result.module,
                    "status": "passed" if result.ok else "failed",
                    "diagnostics": result.error,
                    "returncode": result.returncode,
                    "duration_seconds": result.duration_seconds,
                    "artifacts": result.artifacts,
                    "artifact": str(result.wheel.relative_to(outdir))
                    if result.wheel
                    else None,
                    "sha256": hashlib.sha256(result.wheel.read_bytes()).hexdigest()
                    if result.wheel
                    else None,
                }
                for result in summary.results
            ],
            "probes": receipts,
        }
        pending = attempt / "receipt.pending.json"
        with pending.open("x", encoding="utf-8") as target:
            json.dump(record, target, indent=2, allow_nan=False)
            target.write("\n")
        if returncode == 0:
            try:
                validate_wheel_receipt(
                    pending,
                    modules=[module.name for module in modules],
                    profiles=[
                        (
                            profile.package,
                            ",".join(profile.extras) or "base",
                            hashlib.sha256(profile.code.encode()).hexdigest(),
                        )
                        for profile in REQUIRED_PROFILES
                    ],
                    revision=before["revision"],
                    require_imports=args.verify,
                    require_profiles=args.verify_extras,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append(f"Invalid retained wheel evidence: {exc}")
                returncode = 1
                record.update(status="failed", returncode=1)
                pending.unlink()
                with pending.open("x", encoding="utf-8") as target:
                    json.dump(record, target, indent=2, allow_nan=False)
                    target.write("\n")
        pending.rename(attempt / "receipt.json")
        print(f"Wheel attempt receipt: {attempt / 'receipt.json'}", flush=True)
    return returncode


def _perform_build(args, outdir, modules, summary, receipts, attempt) -> int:
    python = [sys.executable]
    for module_dir in modules:
        result = build_wheel(module_dir, outdir, python, evidence_dir=attempt)
        if result.ok and result.wheel is not None:
            owned = attempt / "wheels" / result.wheel.name
            owned.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.wheel, owned)
            result.wheel = owned
        summary.results.append(result)
    if (args.verify or args.verify_extras) and not summary.failures:
        verify_wheels(
            [result.wheel for result in summary.results if result.wheel is not None],
            python,
            import_timeout=args.import_timeout,
            receipts=receipts,
            evidence_dir=attempt / "probes",
        )
        if args.verify_extras:
            verify_profiles(
                [
                    result.wheel
                    for result in summary.results
                    if result.wheel is not None
                ],
                python,
                import_timeout=args.import_timeout,
                receipts=receipts,
                evidence_dir=attempt / "probes",
            )
    for result in summary.results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.module} -> {result.wheel or result.error}")
    if not summary.namespaces_valid:
        print("Namespace validation FAILED")
        return 1
    print("Namespace validation passed across all packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
