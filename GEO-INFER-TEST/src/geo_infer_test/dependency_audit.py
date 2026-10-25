"""Audit every registry version in the frozen universal workspace lock.

No target-package resolution or installation occurs. Platform, extra and group
markers never filter this inventory. Only authenticated local workspace members
are outside the PyPI advisory query; unknown sources and skipped packages fail.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import tomllib
import uuid

from .execution import runtime_receipt
from .process import run_process

AUDITOR_VERSION = "2.10.1"
UV_VERSION = "0.13.1"


def canonical_name(value: str) -> str:
    """Normalize a distribution name using the PyPI name equivalence rule."""
    return re.sub(r"[-_.]+", "-", value).lower()


def locked_inventory(root: Path) -> tuple[list[tuple[str, str]], list[dict], dict]:
    """Read all universal lock nodes; authenticate every excluded local node."""
    paths = [root / "pyproject.toml", *sorted(root.glob("GEO-INFER-*/pyproject.toml"))]
    metadata = {path: tomllib.loads(path.read_text(encoding="utf-8")) for path in paths}
    local = {
        canonical_name(data["project"]["name"]): (
            path.parent.relative_to(root).as_posix(),
            data["project"]["version"],
        )
        for path, data in metadata.items()
    }
    if len(local) != len(metadata):
        raise ValueError("Duplicate workspace distribution name")
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("Missing resolved lock package inventory")
    registry, excluded, seen_local = set(), [], set()
    for package in packages:
        name, version, source = (
            package.get("name"),
            package.get("version"),
            package.get("source"),
        )
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not isinstance(source, dict)
        ):
            raise ValueError("Incomplete resolved lock node")
        name = canonical_name(name)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version
        ):
            raise ValueError("Invalid pinned distribution identity")
        if source == {"registry": "https://pypi.org/simple"}:
            registry.add((name, version))
        elif set(source) in ({"editable"}, {"virtual"}):
            location = source.get("editable", source.get("virtual"))
            if (
                name not in local
                or (location, version) != local[name]
                or name in seen_local
            ):
                raise ValueError(
                    "Unrecognized or duplicate local workspace node: " + name
                )
            seen_local.add(name)
            excluded.append({"name": name, "version": version, "source": source})
        else:
            raise ValueError(
                "Unsupported dependency source requires separate audit: " + name
            )
    if seen_local != set(local) or not registry:
        raise ValueError("Incomplete local or registry lock inventory")
    paths += [root / "uv.lock", root / "uv.toml", root / ".python-version"]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return sorted(registry), sorted(excluded, key=lambda row: row["name"]), hashes


def requirement_batches(pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Separate universal version forks without dropping any name/version pair."""
    batches: list[list[tuple[str, str]]] = []
    for pair in pairs:
        for batch in batches:
            if all(name != pair[0] for name, _ in batch):
                batch.append(pair)
                break
        else:
            batches.append([pair])
    return batches


def validate_response(
    raw: bytes, expected: list[tuple[str, str]], returncode: int
) -> list[dict]:
    """Require complete exact tool coverage and preserve every advisory."""

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate audit JSON key")
            result[key] = value
        return result

    if len(raw) > 16 * 1024**2:
        raise ValueError("Audit JSON exceeds receipt budget")
    response = json.loads(raw, object_pairs_hook=unique)
    dependencies = response.get("dependencies") if isinstance(response, dict) else None
    if not isinstance(dependencies, list):
        raise ValueError("Audit JSON has no dependency list")
    observed, findings = [], []
    for row in dependencies:
        if not isinstance(row, dict) or row.get("skip_reason"):
            raise ValueError("Auditor skipped or did not identify a dependency")
        if (
            not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
            or not isinstance(row.get("vulns"), list)
        ):
            raise ValueError("Incomplete dependency audit row")
        pair = canonical_name(row["name"]), row["version"]
        observed.append(pair)
        for advisory in row["vulns"]:
            if (
                not isinstance(advisory, dict)
                or not isinstance(advisory.get("id"), str)
                or not advisory["id"]
            ):
                raise ValueError("Malformed advisory")
            findings.append({"name": pair[0], "version": pair[1], **advisory})
    if Counter(observed) != Counter(expected):
        raise ValueError(
            "Auditor package/version coverage differs from frozen inventory"
        )
    if returncode not in (0, 1) or returncode != int(bool(findings)):
        raise ValueError(
            "Auditor exit status disagrees with its complete advisory result"
        )
    return findings


def audit(root: Path, output: Path, *, uv: str, auditor: str, timeout: float) -> dict:
    """Run one finite fail-closed audit and always retain its actual evidence."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Audit timeout must be finite and positive")
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    attempt.mkdir()
    deadline = time.monotonic() + timeout
    report = {
        "schema": "geo-infer-dependency-audit-v1",
        "status": "failed",
        "scope": "All PyPI name/version nodes in the frozen universal lock, including all platform/Python forks, extras and groups. Known Python-package advisories only; not a source scan or native shared-library security claim.",
        "errors": [],
        "findings": [],
        "batches": [],
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONOPTIMIZE",
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
        }
    }
    env["PYTHONNOUSERSITE"] = "1"

    def command(label: str, argv: list[str]):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Dependency audit deadline exhausted")
        try:
            result = run_process(argv, timeout=remaining, cwd=root, env=env)
        except (OSError, subprocess.SubprocessError) as error:
            (attempt / (label + ".stdout")).write_text(
                str(getattr(error, "output", "") or ""), encoding="utf-8"
            )
            (attempt / (label + ".stderr")).write_text(
                str(getattr(error, "stderr", "") or ""), encoding="utf-8"
            )
            raise
        (attempt / (label + ".stdout")).write_text(result.stdout, encoding="utf-8")
        (attempt / (label + ".stderr")).write_text(result.stderr, encoding="utf-8")
        return result

    def custody():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Dependency audit deadline exhausted before source custody"
            )
        return runtime_receipt(root=root, timeout=remaining)

    cache = tempfile.TemporaryDirectory(prefix="geo-dependency-advisories-")
    try:
        before = custody()
        if not before["custody_complete"]:
            raise ValueError("Source custody unavailable")
        report["source_before"] = before
        pairs, excluded, hashes = locked_inventory(root)
        report.update(
            expected_pairs=[list(pair) for pair in pairs],
            expected_package_versions=len(pairs),
            excluded_local_workspace=excluded,
            input_sha256=hashes,
        )
        uv_result = command("uv-version", [uv, "--version"])
        tool_result = command("auditor-version", [auditor, "--version"])
        if uv_result.returncode or not uv_result.stdout.startswith(
            "uv " + UV_VERSION + " "
        ):
            raise ValueError("Exact uv " + UV_VERSION + " required")
        if (
            tool_result.returncode
            or tool_result.stdout.strip() != "pip-audit " + AUDITOR_VERSION
        ):
            raise ValueError("Exact pip-audit " + AUDITOR_VERSION + " required")
        report["tools"] = {
            "uv": uv_result.stdout.strip(),
            "pip-audit": tool_result.stdout.strip(),
        }
        check = command("frozen-lock-check", [uv, "lock", "--check", "--offline"])
        if check.returncode:
            raise ValueError("Frozen lock is not current with workspace metadata")
        for index, batch in enumerate(requirement_batches(pairs)):
            label = f"audit-{index:02d}"
            requirements = attempt / (label + ".txt")
            requirements.write_text(
                "".join(f"{name}=={version}\n" for name, version in batch),
                encoding="utf-8",
            )
            result_path = attempt / (label + ".json")
            argv = [
                auditor,
                "--strict",
                "--no-deps",
                "--disable-pip",
                "--cache-dir",
                cache.name,
                "--timeout",
                "30",
                "--vulnerability-service",
                "pypi",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--output",
                str(result_path),
                "--requirement",
                str(requirements),
            ]
            result = command(label, argv)
            if not result_path.is_file():
                raise ValueError("Auditor produced no JSON receipt")
            findings = validate_response(
                result_path.read_bytes(), batch, result.returncode
            )
            report["findings"].extend(findings)
            report["batches"].append(
                {
                    "requirements": requirements.name,
                    "result": result_path.name,
                    "returncode": result.returncode,
                    "package_versions": len(batch),
                    "findings": len(findings),
                }
            )
        if sum(row["package_versions"] for row in report["batches"]) != len(pairs):
            raise ValueError("Incomplete audit batch inventory")
        report["all_package_versions_audited"] = True
        if not report["findings"]:
            report["status"] = "passed"
    except (
        OSError,
        ImportError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        cache.cleanup()
        try:
            after = custody()
            report["source_after"] = after
            _, _, final_hashes = locked_inventory(root)
            equal = (
                report.get("source_before") == after
                and report.get("input_sha256") == final_hashes
            )
            report["source_and_lock_bookends_equal"] = equal
            if not equal:
                report["errors"].append("Source or lock changed during audit")
        except (
            OSError,
            ImportError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as error:
            report["errors"].append("Final custody unavailable: " + str(error))
        if report["errors"] or report["findings"]:
            report["status"] = "failed"
        report["evidence_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(attempt.iterdir())
            if path.is_file()
        }
        (attempt / "receipt.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "receipt": str(attempt / "receipt.json"),
                "status": report["status"],
                "package_versions": report.get("expected_package_versions"),
                "findings": len(report["findings"]),
                "errors": report["errors"],
            }
        )
    )
    return report


def main() -> int:
    """Run the standalone audit without installing the GEO scientific stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".geo-infer-test-results/dependency-audit")
    )
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--auditor", default="pip-audit")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    report = audit(
        args.repo,
        args.output.resolve(),
        uv=args.uv,
        auditor=args.auditor,
        timeout=args.timeout,
    )
    return int(report["status"] != "passed")
