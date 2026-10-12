"""
Test discovery module for GEO-INFER-TEST.

This module provides intelligent test discovery capabilities across all
GEO-INFER modules, supporting various test types and patterns.
"""

from typing import Any
from pathlib import Path
import ast
import re
import logging

from ..execution import (
    discover_geo_infer_modules,
    discover_workspace_test_targets,
    Module,
    category_test_paths,
)

logger = logging.getLogger(__name__)


# Canonical list of all GEO-INFER ecosystem modules
ALL_MODULES = [module.name for module in discover_geo_infer_modules()]


class TestDiscoverer:
    """
    Intelligent test discovery system for the GEO-INFER ecosystem.

    Discovers and catalogs all available tests across modules, supporting
    multiple test types and frameworks.
    """

    SUPPORTED_TEST_TYPES = [
        "unit",
        "integration",
        "system",
        "performance",
        "load",
        "stress",
        "manuscript",
        "manuscript-render",
    ]
    TEST_FILE_PATTERNS = [r"test_.*\.py$", r".*_test\.py$", r"test.*\.py$"]

    def __init__(self, base_path: Path | None = None):
        """Initialize the test discoverer."""
        self.base_path = base_path or Path.cwd()
        self.discovered_tests: dict[str, dict[str, list[str]]] = {}
        self.test_metadata: dict[str, dict] = {}

    def discover_all_tests(self, modules: list[str]) -> dict[str, dict[str, list[str]]]:
        """
        Discover all tests for the specified modules.

        Args:
            modules: List of module names to discover tests for

        Returns:
            Dictionary mapping module names to test types to test files
        """
        discovered = {}

        for module in modules:
            if module == "ALL":
                # Discover all available modules
                available_modules = self._find_all_modules()
                for mod in available_modules:
                    module_tests = self._discover_module_tests(mod)
                    if module_tests:
                        discovered[mod] = module_tests
            else:
                module_tests = self._discover_module_tests(module)
                if module_tests:
                    discovered[module] = module_tests

        self.discovered_tests = discovered
        return discovered

    def _find_all_modules(self) -> list[str]:
        """Find all available GEO-INFER modules."""
        return [
            module.name for module in discover_workspace_test_targets(self.base_path)
        ]

    def _discover_module_tests(self, module: str) -> dict[str, list[str]]:
        """Discover tests for a specific module."""
        module_tests: dict[str, list[str]] = {}
        module_path = (
            self.base_path
            if module == "ROOT"
            else self.base_path / f"GEO-INFER-{module}"
        )

        if not module_path.exists():
            return module_tests

        tests_path = module_path / "tests"
        if not tests_path.exists():
            return module_tests

        # Use the same owning-module and registered-root boundary as the CLI.
        descriptor = Module(module, module_path, tests_path, True)
        for test_type in self.SUPPORTED_TEST_TYPES:
            files = category_test_paths(descriptor, test_type)
            base = tests_path / test_type
            # Direct tests retain the historical "general" reporting bucket;
            # execution treats that bucket as unit, without double-counting.
            files = [path for path in files if path.parent != tests_path]
            if files:
                module_tests[test_type] = [
                    str(path.relative_to(base))
                    if path.is_relative_to(base)
                    else str(path.relative_to(module_path))
                    for path in files
                ]

        # Also check for tests directly in the tests directory.
        # GS19-86: this pass is deliberately NON-recursive. The typed buckets
        # above already own every nested file; a recursive scan here
        # re-listed unit/integration/etc. files under "general" and
        # double-counted them in get_test_statistics.
        root_test_files = (
            self._find_test_files(tests_path, recursive=False)
            if module != "ROOT"
            else []
        )
        if root_test_files:
            module_tests["general"] = root_test_files

        return module_tests

    def _find_test_files(self, directory: Path, recursive: bool = True) -> list[str]:
        """Find test files in a directory, optionally recursive."""
        test_files: list[str] = []

        if not directory.exists():
            return test_files

        candidates = directory.rglob("*.py") if recursive else directory.iterdir()
        for file_path in candidates:
            if not file_path.is_file():
                continue
            if self._is_test_file(file_path):
                # Store relative path from the tests directory
                relative_path = file_path.relative_to(directory)
                test_files.append(str(relative_path))

        return sorted(test_files)

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file based on naming patterns."""
        filename = file_path.name

        # Check against common test file patterns
        for pattern in self.TEST_FILE_PATTERNS:
            if re.match(pattern, filename):
                return True

        return False

    def analyze_test_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a test file to extract metadata."""
        funcs_out: list[dict[str, Any]] = []
        classes_out: list[dict[str, Any]] = []
        imports_out: list[str] = []
        metadata: dict[str, Any] = {
            "functions": funcs_out,
            "classes": classes_out,
            "imports": imports_out,
            "dependencies": [],
            "docstring": None,
            "framework": "unknown",
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content)

            # Extract module docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                metadata["docstring"] = tree.body[0].value.value

            # Analyze the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        funcs_out.append(
                            {
                                "name": node.name,
                                "line": node.lineno,
                                "docstring": ast.get_docstring(node),
                            }
                        )

                elif isinstance(node, ast.ClassDef):
                    if "test" in node.name.lower():
                        classes_out.append(
                            {
                                "name": node.name,
                                "line": node.lineno,
                                "docstring": ast.get_docstring(node),
                            }
                        )

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_out.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_out.append(node.module)

            # Detect testing framework
            metadata["framework"] = self._detect_framework(imports_out)

        except Exception as exc:
            # A parse failure means this test file is NOT being discovered;
            # surface it instead of silently dropping the file from runs.
            logger.warning("Could not parse test file %s: %s", file_path, exc)

        return metadata

    def _detect_framework(self, imports: list[str]) -> str:
        """Detect the testing framework being used."""
        frameworks = {
            "pytest": ["pytest"],
            "unittest": ["unittest"],
            "nose": ["nose", "nose2"],
            "hypothesis": ["hypothesis"],
            "locust": ["locust"],
            "selenium": ["selenium"],
        }

        for framework, patterns in frameworks.items():
            for pattern in patterns:
                if any(pattern in imp for imp in imports):
                    return framework

        return "unknown"

    def get_test_statistics(self) -> dict[str, Any]:
        """Get statistics about discovered tests."""
        tests_by_type: dict[str, int] = {}
        tests_by_module: dict[str, int] = {}
        total_files = 0
        stats: dict[str, Any] = {
            "total_modules": len(self.discovered_tests),
            "total_test_files": total_files,
            "tests_by_type": tests_by_type,
            "tests_by_module": tests_by_module,
        }

        for module, test_types in self.discovered_tests.items():
            module_total = sum(len(files) for files in test_types.values())
            tests_by_module[module] = module_total
            total_files += module_total

            for test_type, files in test_types.items():
                if test_type not in tests_by_type:
                    tests_by_type[test_type] = 0
                tests_by_type[test_type] += len(files)

        return stats

    def find_cross_module_tests(self) -> list[tuple[str, str, str]]:
        """Find tests that appear to test cross-module functionality."""
        cross_module_tests = []

        for module, test_types in self.discovered_tests.items():
            for test_type, files in test_types.items():
                for file_path in files:
                    # Check if the test file mentions other modules. Files
                    # discovered directly under tests/ are stored under the
                    # synthetic key "general"; reconstruct their real location
                    # (tests/<file>) instead of the non-existent tests/general/.
                    tests_dir = self.base_path / f"GEO-INFER-{module}" / "tests"
                    base_dir = (
                        tests_dir if test_type == "general" else tests_dir / test_type
                    )
                    full_path = base_dir / file_path

                    if full_path.exists():
                        metadata = self.analyze_test_file(full_path)

                        # Look for imports of other GEO-INFER modules
                        for import_name in metadata["imports"]:
                            if "geo_infer_" in import_name:
                                imported_module = import_name.replace(
                                    "geo_infer_", ""
                                ).upper()
                                if imported_module != module:
                                    cross_module_tests.append(
                                        (module, imported_module, file_path)
                                    )

        return cross_module_tests

    def validate_test_structure(self) -> dict[str, list[str]]:
        """Validate the structure of discovered tests."""
        missing_dirs: list[str] = []
        empty_dirs: list[str] = []
        malformed_files: list[str] = []
        missing_inits: list[str] = []
        issues: dict[str, list[str]] = {
            "missing_test_dirs": missing_dirs,
            "empty_test_dirs": empty_dirs,
            "malformed_test_files": malformed_files,
            "missing_init_files": missing_inits,
        }

        for module in self._find_all_modules():
            module_path = self.base_path / f"GEO-INFER-{module}"
            tests_path = module_path / "tests"

            if not tests_path.exists():
                missing_dirs.append(module)
                continue

            # Check for empty test directories
            test_files = list(tests_path.rglob("test_*.py"))
            if not test_files:
                empty_dirs.append(module)

            # Check for __init__.py files in test directories
            for test_type in self.SUPPORTED_TEST_TYPES:
                test_type_path = tests_path / test_type
                if test_type_path.exists() and test_type_path.is_dir():
                    init_file = test_type_path / "__init__.py"
                    if not init_file.exists():
                        missing_inits.append(f"{module}/{test_type}")

        return issues
