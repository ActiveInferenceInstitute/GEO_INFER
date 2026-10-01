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
from email.parser import BytesParser
from fnmatch import fnmatchcase
import zipfile
import os
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from import_probe import run_import_probe
from validate_packaging import (
    distribution_name,
    module_dirs,
    parse_pyproject,
    wheel_filename_is_valid,
)


@dataclass
class BuildResult:
    module: str
    distribution: str | None
    wheel: Path | None = None
    ok: bool = False
    error: str = ""


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


def build_wheel(module_dir: Path, outdir: Path, python: list[str]) -> BuildResult:
    """Build in a fresh directory so stale neighboring wheels cannot be selected."""
    distribution = distribution_name(parse_pyproject(module_dir))
    result = BuildResult(module=module_dir.name, distribution=distribution)
    if not distribution:
        result.error = "missing [project].name"
        return result
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=outdir) as temporary:
            subprocess.run(
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
                capture_output=True,
                timeout=300,
            )
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
        detail = getattr(exc, "stderr", None) or str(exc)
        result.error = (
            detail.decode(errors="replace") if isinstance(detail, bytes) else detail
        )[-2000:]
    return result


def verify_wheels(
    wheels: list[Path], python: list[str], *, import_timeout: float = 120
) -> None:
    """Install wheels in a clean environment and execute actual import/resource probes."""
    if not wheels:
        raise ValueError("At least one wheel is required")
    if not math.isfinite(import_timeout) or import_timeout <= 0:
        raise ValueError("Import timeout must be finite and positive")
    wheels = [wheel.resolve() for wheel in wheels]
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="geo-infer-wheel-check-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        subprocess.run(
            ["uv", "venv", "--python", python[0], str(environment)],
            check=True,
            capture_output=True,
            env=env,
            timeout=120,
        )
        executable = str(
            environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        repository = Path(__file__).resolve().parent.parent
        constraints = root / "constraints.txt"
        subprocess.run(
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
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
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
                *[str(wheel) for wheel in wheels],
            ],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
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
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'package':name,'version':dist.version,'origin':str(origin),'resources':resources,'probe_token':sys.argv[-1],'status':'ok'}))
"""
        for wheel in wheels:
            package = wheel.name.split("-")[0]
            result = run_import_probe(
                [executable, "-I", "-c", probe, package, str(import_timeout)],
                package=package,
                cwd=root,
                env=env,
                timeout=import_timeout,
            )
            print(result.stdout.strip(), flush=True)


def install_and_verify(wheel: Path, python: list[str]) -> None:
    """Verify one wheel with the same isolation and resource contract as a release."""
    verify_wheels([wheel], python)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate GEO-INFER wheels")
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument("--verify", action="store_true", help="isolated venv install")
    parser.add_argument(
        "--import-timeout",
        type=float,
        default=120,
        help="Seconds allowed per installed package import (default 120)",
    )
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    python = [sys.executable]
    summary = BuildSummary()
    for module_dir in module_dirs():
        result = build_wheel(module_dir, outdir, python)
        summary.results.append(result)
    if args.verify and not summary.failures:
        try:
            verify_wheels(
                [r.wheel for r in summary.results if r.wheel is not None],
                python,
                import_timeout=args.import_timeout,
            )
        except (
            OSError,
            ValueError,
            zipfile.BadZipFile,
            subprocess.SubprocessError,
        ) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            print(
                "Installed-wheel verification failed: "
                + (
                    detail.decode(errors="replace")
                    if isinstance(detail, bytes)
                    else detail
                )
            )
            return 1

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
