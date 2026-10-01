#!/usr/bin/env python3
"""Generate a current, discovery-based GEO-INFER repository receipt.

The former assessor required YAML front matter and fixed README headings that
the repository no longer uses. This replacement reports observable repository
facts, keeps historical outputs untouched, and writes every receipt to a
unique dated path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_PREFIX = "GEO-INFER-"


def _python_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*.py")
        if ".venv" not in candidate.parts
        and "site-packages" not in candidate.parts
        and "__pycache__" not in candidate.parts
    )


def _signpost_gaps(module_path: Path) -> list[str]:
    """Find code/test directories missing both local repository signposts."""
    gaps: list[str] = []
    for root in sorted({path.parent for path in _python_files(module_path)}):
        if root == module_path or ".git" in root.parts:
            continue
        if (root / "README.md").exists() and (root / "AGENTS.md").exists():
            continue
        relative = root.relative_to(PROJECT_ROOT)
        gaps.append(str(relative))
    return gaps


class RepositoryAssessment:
    """Discover modules and emit current-state facts without status guessing."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules = {
            path.name: path
            for path in sorted(project_root.glob(f"{MODULE_PREFIX}*"))
            if path.is_dir()
        }

    def assess_module(self, module_name: str, module_path: Path) -> dict[str, Any]:
        source_files = (
            _python_files(module_path / "src") if (module_path / "src").exists() else []
        )
        test_files = (
            [
                path
                for path in _python_files(module_path / "tests")
                if path.name.startswith("test_") or path.name.endswith("_test.py")
            ]
            if (module_path / "tests").exists()
            else []
        )
        h3_runtime = any(
            "import h3" in path.read_text(encoding="utf-8", errors="ignore")
            or "from h3" in path.read_text(encoding="utf-8", errors="ignore")
            or "h3." in path.read_text(encoding="utf-8", errors="ignore")
            for path in source_files
        )
        pyproject = module_path / "pyproject.toml"
        requirements = list(module_path.glob("requirements*.txt"))
        dependency_text = (
            "\n".join(
                [pyproject.read_text(encoding="utf-8", errors="ignore")]
                if pyproject.exists()
                else []
            )
            + "\n"
            + "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in requirements
            )
        )
        return {
            "module": module_name,
            "path": str(module_path.relative_to(self.project_root)),
            "source_files": len(source_files),
            "test_files": len(test_files),
            "readme_files": len(list(module_path.rglob("README.md"))),
            "agents_files": len(list(module_path.rglob("AGENTS.md"))),
            "skill_files": len(list(module_path.rglob("SKILL.md"))),
            "has_pyproject": pyproject.is_file(),
            "has_requirements": bool(requirements),
            "h3_runtime": h3_runtime,
            "h3_dependency_contract": "h3>=4.5.0,<5" in dependency_text,
            "signpost_gaps": _signpost_gaps(module_path),
        }

    def generate_report(self) -> dict[str, Any]:
        details = {
            name: self.assess_module(name, path) for name, path in self.modules.items()
        }
        h3_modules = [name for name, detail in details.items() if detail["h3_runtime"]]
        missing_h3_contract = [
            name for name in h3_modules if not details[name]["h3_dependency_contract"]
        ]
        return {
            "schema_version": 2,
            "assessment_date": datetime.now(UTC).isoformat(),
            "repository": str(self.project_root),
            "source_of_truth": [
                "filesystem discovery",
                "pyproject.toml and requirements*.txt",
                "GEO-INFER-TEST validators",
            ],
            "statistics": {
                "modules": len(details),
                "source_files": sum(item["source_files"] for item in details.values()),
                "test_files": sum(item["test_files"] for item in details.values()),
                "readme_files": sum(item["readme_files"] for item in details.values()),
                "agents_files": sum(item["agents_files"] for item in details.values()),
                "skill_files": sum(item["skill_files"] for item in details.values()),
                "h3_runtime_modules": len(h3_modules),
            },
            "open_findings": {
                "modules_missing_h3_dependency_contract": missing_h3_contract,
                "modules_with_signpost_gaps": {
                    name: detail["signpost_gaps"]
                    for name, detail in details.items()
                    if detail["signpost_gaps"]
                },
            },
            "module_details": details,
        }

    @staticmethod
    def markdown(report: dict[str, Any]) -> str:
        stats = report["statistics"]
        findings = report["open_findings"]
        lines = [
            "# GEO-INFER current repository assessment",
            "",
            f"Assessment date: `{report['assessment_date']}`",
            "",
            "This receipt is discovery-based. It does not infer implementation status from names, old headings, or fixed module lists.",
            "",
            "## Inventory",
            "",
            f"- Modules: {stats['modules']}",
            f"- Source files: {stats['source_files']}",
            f"- Test files: {stats['test_files']}",
            f"- README files: {stats['readme_files']}",
            f"- AGENTS files: {stats['agents_files']}",
            f"- SKILL files: {stats['skill_files']}",
            f"- Modules with H3 runtime imports: {stats['h3_runtime_modules']}",
            "",
            "## Open findings",
            "",
            f"- H3 dependency-contract gaps: {findings['modules_missing_h3_dependency_contract'] or 'none'}",
            f"- Signpost gaps: {findings['modules_with_signpost_gaps'] or 'none'}",
            "",
            "## Module facts",
            "",
            "| Module | Source | Tests | README | AGENTS | H3 contract |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, detail in report["module_details"].items():
            h3_contract = "yes" if detail["h3_dependency_contract"] else "n/a"
            if detail["h3_runtime"] and not detail["h3_dependency_contract"]:
                h3_contract = "MISSING"
            lines.append(
                f"| {name} | {detail['source_files']} | {detail['test_files']} | "
                f"{detail['readme_files']} | {detail['agents_files']} | {h3_contract} |"
            )
        return "\n".join(lines) + "\n"

    def save_report(
        self, report: dict[str, Any], output: Path | None = None
    ) -> tuple[Path, Path]:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = output or (
            self.project_root
            / "GEO-INFER-INTRA"
            / "assessment_results"
            / "current"
            / f"assessment_{stamp}"
        )
        if output.suffix:
            output = output.with_suffix("")
        output.parent.mkdir(parents=True, exist_ok=True)
        json_path = output.with_suffix(".json")
        markdown_path = output.with_suffix(".md")
        if json_path.exists() or markdown_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing receipt: {output}")
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(self.markdown(report), encoding="utf-8")
        return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Receipt basename; existing JSON/Markdown receipts are never overwritten.",
    )
    args = parser.parse_args()
    assessor = RepositoryAssessment(PROJECT_ROOT)
    report = assessor.generate_report()
    json_path, markdown_path = assessor.save_report(
        report,
        (PROJECT_ROOT / args.output)
        if args.output and not args.output.is_absolute()
        else args.output,
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
