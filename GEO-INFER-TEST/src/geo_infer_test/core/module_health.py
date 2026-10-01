"""
GEO-INFER-TEST Module Health Checker.

Inspects each GEO-INFER module for importability, documentation
presence, test coverage, and dependency readiness.
"""

import importlib
import logging
import platform
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HealthMetrics:
    """Health metrics for a single module."""

    module_name: str
    importable: bool = False
    has_readme: bool = False
    has_agents_md: bool = False
    has_tests: bool = False
    test_count: int = 0
    has_pyproject: bool = False
    dependency_status: str = "unknown"  # ok | missing | unknown
    overall_status: str = "unknown"  # healthy | degraded | unhealthy | unknown
    details: Dict[str, Any] = field(default_factory=dict)


class ModuleHealthChecker:
    """
    Checks module importability, test presence, and documentation presence
    across GEO-INFER submodules.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_path = base_path or Path.cwd()
        self.logger = logger or logging.getLogger(__name__)

    def check_module(self, module_name: str) -> HealthMetrics:
        """Run all health checks for a single module."""
        metrics = HealthMetrics(module_name=module_name)

        module_dir = self.base_path / f"GEO-INFER-{module_name}"

        # 1. Directory existence
        if not module_dir.is_dir():
            metrics.overall_status = "unhealthy"
            metrics.details["error"] = f"Directory {module_dir.name} not found"
            return metrics

        # 2. Importability
        pkg_name = f"geo_infer_{module_name.lower()}"
        try:
            importlib.import_module(pkg_name)
            metrics.importable = True
        except Exception as exc:
            metrics.importable = False
            metrics.details["import_error"] = str(exc)

        # 3. Documentation
        metrics.has_readme = (module_dir / "README.md").is_file()
        metrics.has_agents_md = (module_dir / "AGENTS.md").is_file()

        # 4. Tests
        test_dir = module_dir / "tests"
        if test_dir.is_dir():
            test_files = {
                *test_dir.rglob("test_*.py"),
                *test_dir.rglob("*_test.py"),
            }
            metrics.has_tests = len(test_files) > 0
            metrics.test_count = len(test_files)
        else:
            metrics.has_tests = False

        # 5. pyproject.toml (the only packaging source of truth)
        metrics.has_pyproject = (module_dir / "pyproject.toml").is_file()

        # 6. Dependency check
        dep_checker = DependencyChecker(base_path=self.base_path, logger=self.logger)
        dep_result = dep_checker.check_module_dependencies(module_name)
        metrics.dependency_status = dep_result.get("status", "unknown")

        # 7. Overall status
        metrics.overall_status = self._assess_status(metrics)
        return metrics

    def check_all_modules(self, modules: List[str]) -> Dict[str, HealthMetrics]:
        """Check health across a list of module names."""
        results: Dict[str, HealthMetrics] = {}
        for module_name in modules:
            results[module_name] = self.check_module(module_name)
            self.logger.info(
                "Module %s: %s", module_name, results[module_name].overall_status
            )
        return results

    @staticmethod
    def _assess_status(metrics: HealthMetrics) -> str:
        if not metrics.importable:
            return "unhealthy"
        score = sum(
            [
                metrics.has_readme,
                metrics.has_tests,
                metrics.has_pyproject,
                metrics.dependency_status == "ok",
            ]
        )
        if score >= 3:
            return "healthy"
        if score >= 2:
            return "degraded"
        return "unhealthy"


class SystemValidator:
    """
    Validates system-level requirements: Python version, platform,
    disk space, and key global packages.
    """

    MIN_PYTHON = (3, 11)
    REQUIRED_PACKAGES = ["pytest", "numpy", "pandas"]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate(self) -> Dict[str, Any]:
        """Return a comprehensive system-level validation report."""
        py_ok = sys.version_info[:2] >= self.MIN_PYTHON

        missing_packages: List[str] = []
        for pkg in self.REQUIRED_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing_packages.append(pkg)

        total, used, free = shutil.disk_usage(Path.cwd())

        report: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "python_meets_minimum": py_ok,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "missing_packages": missing_packages,
            "disk_free_gb": round(free / (1024**3), 2),
            "system_ok": py_ok and len(missing_packages) == 0,
        }
        self.logger.info(
            "System validation: %s", "OK" if report["system_ok"] else "ISSUES"
        )
        return report


# GS19-87: distribution names whose importable module root differs from the
# normalized (lowercase, ``-`` -> ``_``) distribution name. Probing the raw
# normalized name via importlib produced false "missing" verdicts for every
# module declaring these dependencies (e.g. "pyyaml" is imported as "yaml").
# Keys are lowercase distribution names in their post-normalization
# (dash -> underscore) form; values are importable module roots. The table
# covers the distributions the fleet actually declares.
DISTRIBUTION_IMPORT_NAMES = {
    "bayeux_ml": "bayeux",
    "gitpython": "git",
    "inferactively_pymdp": "pymdp",
    "mkdocs_material": "material",
    "pillow": "PIL",
    "psycopg2_binary": "psycopg2",
    "python_multipart": "multipart",
    "pyjwt": "jwt",
    "pyyaml": "yaml",
    "scikit_learn": "sklearn",
    "z3_solver": "z3",
}


class DependencyChecker:
    """
    Checks whether the dependencies listed in a module's ``pyproject.toml``
    are installed in the current environment.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_path = base_path or Path.cwd()
        self.logger = logger or logging.getLogger(__name__)

    def check_module_dependencies(self, module_name: str) -> Dict[str, Any]:
        """Check installed status of a module's declared dependencies."""
        module_dir = self.base_path / f"GEO-INFER-{module_name}"
        pyproject = module_dir / "pyproject.toml"

        if not pyproject.is_file():
            return {"status": "unknown", "reason": "no pyproject.toml"}

        try:
            deps = self._extract_dependencies(pyproject)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            # A corrupted or unreadable pyproject.toml is unknown, never a
            # success-shaped empty dependency list.
            return {
                "status": "unknown",
                "reason": f"pyproject.toml parse failed: {exc}",
            }
        missing: List[str] = []
        installed: List[str] = []

        for dep in deps:
            pkg = self._normalize_dep_name(dep)
            try:
                importlib.import_module(pkg)
                installed.append(dep)
            except ImportError:
                missing.append(dep)

        status = "ok" if not missing else "missing"
        return {
            "status": status,
            "total": len(deps),
            "installed": installed,
            "missing": missing,
        }

    @staticmethod
    def _extract_dependencies(pyproject_path: Path) -> List[str]:
        """Robust extraction of dependencies using tomllib.

        Raises ``OSError`` when the file cannot be read and
        ``tomllib.TOMLDecodeError`` when it is malformed; callers must
        surface that as an unknown dependency status instead of a
        success-shaped empty result.
        """
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        raw_deps = []
        # Standard PEP 621
        if "project" in data and "dependencies" in data["project"]:
            raw_deps.extend(data["project"]["dependencies"])
        # Standard (optional deps)
        if "project" in data and "optional-dependencies" in data["project"]:
            for group in data["project"]["optional-dependencies"].values():
                raw_deps.extend(group)
        # Poetry
        elif (
            "tool" in data
            and "poetry" in data["tool"]
            and "dependencies" in data["tool"]["poetry"]
        ):
            raw_deps.extend(data["tool"]["poetry"]["dependencies"].keys())
        # Top-level (non-standard but supported by prior implementation)
        elif "dependencies" in data:
            raw_deps.extend(data["dependencies"])

        cleaned = []
        for dep in raw_deps:
            if not isinstance(dep, str):
                continue
            # Handle PEP 508 markers (after ;) and extras (``dep[extra]``);
            # GS19-87: probing "coverage[toml]" as a module always reported
            # the dependency missing.
            dep = dep.split(";")[0].split("[")[0]
            # Handle version specifiers
            for sep in (">=", "<=", "==", "~=", "!=", ">", "<"):
                dep = dep.split(sep)[0]
            dep = dep.strip()
            if dep:
                cleaned.append(dep)
        return cleaned

    @staticmethod
    def _normalize_dep_name(dep: str) -> str:
        """Convert a PyPI package name to an importable module name."""
        normalized = dep.lower().replace("-", "_").replace(" ", "_")
        # GS19-87: some distributions ship a differently-named import root
        # ("pyyaml" is imported as "yaml"); consult the fleet's mapping.
        return DISTRIBUTION_IMPORT_NAMES.get(normalized, normalized)
