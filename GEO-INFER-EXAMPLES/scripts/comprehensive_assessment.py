#!/usr/bin/env python3
"""Inventory integration examples without guessing whether they work.

This tool intentionally performs static discovery only. A directory containing
a script is a runnable candidate, not proof of implementation. Execution and
environment-dependent validation belong to the example's explicit test or
release gate.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"
NON_EXAMPLE_DIRS = {
    "config",
    "data",
    "docs",
    "logs",
    "scripts",
    "tests",
    "visualizations",
}
MODULE_PATTERN = re.compile(r"(?:GEO-INFER-|geo_infer_)([A-Za-z0-9_]+)")


def _module_names() -> set[str]:
    return {
        path.name.removeprefix("GEO-INFER-").upper()
        for path in REPO_ROOT.glob("GEO-INFER-*")
        if path.is_dir()
    }


def _python_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*.py")
        if ".venv" not in candidate.parts and "__pycache__" not in candidate.parts
    )


def discover_examples(root: Path = EXAMPLES_ROOT) -> list[Path]:
    """Discover example roots from executable content and documentation."""
    candidates: set[Path] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_dir() or path.name.lower() in NON_EXAMPLE_DIRS:
            continue
        if _python_files(path) and (
            (path / "README.md").exists()
            or any(
                child.name in {"run_example.py", "main.py"}
                for child in path.iterdir()
                if child.is_file()
            )
            or (path / "scripts").is_dir()
        ):
            candidates.add(path)
    # Keep the highest meaningful directory when a nested script directory
    # would otherwise be reported as a second example.
    return sorted(
        path
        for path in candidates
        if not any(parent in candidates for parent in path.parents)
    )


def _entrypoints(path: Path) -> list[str]:
    names = {
        "run_example.py",
        "main.py",
    }
    return sorted(
        str(candidate.relative_to(path))
        for candidate in _python_files(path)
        if candidate.name in names or candidate.parent.name == "scripts"
    )


def assess_example(path: Path, modules: set[str]) -> dict[str, Any]:
    text = "\n".join(
        candidate.read_text(encoding="utf-8", errors="ignore")
        for candidate in [path / "README.md"]
        if candidate.exists()
    )
    discovered_modules = sorted(
        {
            match.group(1).replace("_", "-").upper()
            for match in MODULE_PATTERN.finditer(text)
            if match.group(1).replace("_", "-").upper() in modules
        }
    )
    scripts = _entrypoints(path)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "name": path.name,
        "category": path.parent.name,
        "status": "runnable_candidate" if scripts else "documentation_only",
        "verification": "inventory_only",
        "readme": (path / "README.md").is_file(),
        "entrypoints": scripts,
        "config": any(
            candidate.exists()
            for candidate in (
                path / "config",
                path / "config.yaml",
                path / "config.json",
            )
        ),
        "tests": (path / "tests").is_dir()
        or any(candidate.name.startswith("test_") for candidate in _python_files(path)),
        "modules_identified": discovered_modules,
    }


def build_report(root: Path = EXAMPLES_ROOT) -> dict[str, Any]:
    modules = _module_names()
    assessments = [assess_example(path, modules) for path in discover_examples(root)]
    return {
        "schema_version": 2,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_of_truth": "filesystem discovery and README text only",
        "execution_performed": False,
        "summary": {
            "total_examples": len(assessments),
            "runnable_candidates": sum(
                item["status"] == "runnable_candidate" for item in assessments
            ),
            "documentation_only": sum(
                item["status"] == "documentation_only" for item in assessments
            ),
            "with_tests": sum(item["tests"] for item in assessments),
            "with_readme": sum(item["readme"] for item in assessments),
        },
        "examples": assessments,
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# GEO-INFER example inventory",
        "",
        f"Generated: `{report['timestamp']}`",
        "",
        "This is a static inventory. `runnable_candidate` means an entrypoint was found; it is not an execution or implementation claim.",
        "",
        "## Summary",
        "",
        f"- Example roots: {summary['total_examples']}",
        f"- Runnable candidates: {summary['runnable_candidates']}",
        f"- Documentation-only roots: {summary['documentation_only']}",
        f"- With README: {summary['with_readme']}",
        f"- With tests: {summary['with_tests']}",
        "",
        "## Examples",
        "",
        "| Category | Example | Status | Entrypoints | Tests | Modules |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["examples"]:
        lines.append(
            f"| {item['category']} | `{item['name']}` | {item['status']} | "
            f"{', '.join(item['entrypoints']) or 'none'} | "
            f"{'yes' if item['tests'] else 'no'} | "
            f"{', '.join(item['modules_identified']) or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def save_report(
    report: dict[str, Any], output: Path | None = None
) -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    basename = output or (
        EXAMPLES_ROOT.parent / "assessment_results" / f"inventory_{stamp}"
    )
    if basename.suffix:
        basename = basename.with_suffix("")
    basename.parent.mkdir(parents=True, exist_ok=True)
    json_path = basename.with_suffix(".json")
    markdown_path = basename.with_suffix(".md")
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {basename}")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Unique receipt basename")
    args = parser.parse_args()
    report = build_report()
    output = args.output
    if output is not None and not output.is_absolute():
        output = REPO_ROOT / output
    json_path, markdown_path = save_report(report, output)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
