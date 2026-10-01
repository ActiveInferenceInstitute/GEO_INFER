#!/usr/bin/env python3
"""GEO-INFER-GIT module orchestrator.

Runs one documented end-to-end GIT operation: build a synthetic repository
tree in a temporary directory (no cloning, no network) and run the real
``RepositoryAnalyzer`` over it — code-quality metrics, dependency parsing
with known-vulnerability matching, geospatial-content detection, secret and
insecure-pattern scanning, documentation quality, and the weighted overall
score with recommendations.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _build_synthetic_repo(root: Path) -> None:
    """Create a small synthetic repository for in-memory analysis."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(
        "# synthetic-survey-tools\n\n"
        "Synthetic geospatial survey utilities used by the GIT orchestrator "
        "demo. Coordinates are expressed in EPSG:4326.\n",
        encoding="utf-8",
    )
    (root / "docs" / "design.md").write_text(
        "# Design notes\n\nPipeline reads sensor CSVs and stores site polygons.\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text(
        "requests==2.19.0\nnumpy>=1.24\nh3>=4.0\n",
        encoding="utf-8",
    )
    (root / "src" / "survey.py").write_text(
        '"""Survey site utilities (synthetic)."""\n\n'
        "import h3\n\n\n"
        "def cell_for_site(lat: float, lon: float, resolution: int = 8) -> str:\n"
        '    """Return the H3 cell covering a survey site."""\n'
        "    return h3.latlng_to_cell(lat, lon, resolution)\n",
        encoding="utf-8",
    )
    (root / "src" / "legacy_loader.py").write_text(
        "import os\n\n\n"
        "def run_legacy(expression: str) -> None:\n"
        "    # Legacy shortcut, kept deliberately unsafe for the demo scan.\n"
        "    os.system(expression)\n\n\n"
        "def load_admin_password() -> str:\n"
        '    password = "synthetic-demo-credential"\n'
        "    return password\n",
        encoding="utf-8",
    )


def _operation() -> dict[str, Any]:
    from geo_infer_git.core.repo_analyzer import RepositoryAnalyzer

    tmp_root = Path(tempfile.mkdtemp(prefix="geo_infer_git_synthetic_"))
    try:
        _build_synthetic_repo(tmp_root)
        analysis = RepositoryAnalyzer(tmp_root).analyze_repository()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    code = analysis.code_quality
    security = analysis.security_analysis

    return {
        "operation": "synthetic_repo_analysis",
        "code_total_lines": code.total_lines,
        "code_lines": code.code_lines,
        "comment_lines": code.comment_lines,
        "blank_lines": code.blank_lines,
        "cyclomatic_complexity": round(code.cyclomatic_complexity, 3),
        "documentation_coverage": round(code.documentation_coverage, 2),
        "dependencies_found": [
            {"name": dep.name, "version": dep.version} for dep in analysis.dependencies
        ],
        "dependency_vulnerability_count": sum(
            len(dep.vulnerabilities) for dep in analysis.dependencies
        ),
        "has_geospatial_content": analysis.geospatial_content.has_geospatial_data,
        "security_score": round(security.security_score, 1),
        "secrets_detected_count": len(security.secrets_detected),
        "insecure_patterns_detected": security.insecure_patterns,
        "documentation_quality": round(analysis.documentation_quality, 1),
        "overall_score": round(analysis.overall_score, 2),
        "detected_geospatial_formats": analysis.geospatial_content.geospatial_file_formats,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("GIT", _operation))
