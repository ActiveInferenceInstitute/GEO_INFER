"""
GEO-INFER-TEST Test Orchestrator.

Coordinates test execution across modules, resolves inter-module
dependencies, and manages named test suites.
"""

import logging
from copy import deepcopy

from ..execution import (
    discover_geo_infer_modules,
    module_by_name,
    run_module_category_tests,
)
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable


@dataclass
class TestSuiteDefinition:
    """Definition of a named test suite."""

    name: str
    description: str
    test_patterns: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    timeout_seconds: int = 300


class TestSuiteManager:
    """
    Manages named test suites: register, list, retrieve, and combine.
    """

    # Built-in suite definitions
    _BUILTIN_SUITES = {
        "unit": TestSuiteDefinition(
            name="unit",
            description="Unit tests for individual components",
            test_patterns=["tests/unit/test_*.py"],
            markers=["unit"],
        ),
        "integration": TestSuiteDefinition(
            name="integration",
            description="Cross-module integration tests",
            test_patterns=["tests/integration/test_*.py"],
            markers=["integration"],
        ),
        "performance": TestSuiteDefinition(
            name="performance",
            description="Performance and benchmark tests",
            test_patterns=["tests/performance/test_*.py"],
            markers=["benchmark", "performance"],
            timeout_seconds=600,
        ),
        "slow": TestSuiteDefinition(
            name="slow", description="Slow unit complement", markers=["slow"]
        ),
        "system": TestSuiteDefinition(
            name="system", description="System integration tests", markers=["system"]
        ),
    }

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._suites: dict[str, TestSuiteDefinition] = deepcopy(self._BUILTIN_SUITES)

    def register_suite(self, suite: TestSuiteDefinition) -> None:
        """Register or overwrite a named test suite."""
        self._suites[suite.name] = suite
        self.logger.info("Registered suite: %s", suite.name)

    def get_suite(self, name: str) -> TestSuiteDefinition | None:
        return self._suites.get(name)

    def list_suites(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "patterns": s.test_patterns,
                "modules": s.modules,
                "markers": s.markers,
                "timeout_seconds": s.timeout_seconds,
            }
            for s in self._suites.values()
        ]

    def combine_suites(self, names: list[str]) -> TestSuiteDefinition:
        """Create a combined suite from multiple named suites."""
        patterns: list[str] = []
        modules_set: set[str] = set()
        markers: list[str] = []
        max_timeout = 300

        for name in names:
            suite = self._suites.get(name)
            if suite is None:
                raise ValueError(f"Unknown suite: {name}")
            patterns.extend(suite.test_patterns)
            modules_set.update(suite.modules)
            markers.extend(suite.markers)
            max_timeout = max(max_timeout, suite.timeout_seconds)

        return TestSuiteDefinition(
            name="combined_" + "_".join(names),
            description=f"Combined suite from: {', '.join(names)}",
            test_patterns=list(dict.fromkeys(patterns)),  # deduplicate, keep order
            modules=sorted(modules_set),
            markers=list(dict.fromkeys(markers)),
            timeout_seconds=max_timeout,
        )


class TestOrchestrator:
    """
    Coordinates test execution order based on module dependencies,
    priority levels, and suite definitions.
    """

    # Default dependency graph (module → list of modules it depends on)
    DEFAULT_DEPENDENCIES: dict[str, list[str]] = {
        "TEST": [],
        "LOG": [],
        "MATH": [],
        "DATA": ["LOG"],
        "SPACE": ["MATH", "DATA"],
        "TIME": ["MATH", "DATA"],
        "BAYES": ["MATH"],
        "AI": ["MATH", "DATA", "BAYES"],
        "ACT": ["BAYES", "MATH"],
        "AGENT": ["ACT", "AI"],
        "ANT": ["AGENT"],
        "SIM": ["SPACE", "TIME", "AI"],
        "API": ["DATA", "SEC"],
        "APP": ["API"],
        "SEC": ["LOG"],
        "OPS": ["LOG", "API"],
        "IOT": ["SPACE", "TIME", "DATA"],
        "HEALTH": ["SPACE", "TIME", "AI"],
        "AG": ["SPACE", "TIME", "AI"],
        "ECON": ["MATH", "DATA"],
        "RISK": ["BAYES", "SPACE"],
        "BIO": ["SPACE", "DATA", "AI"],
    }

    def __init__(
        self,
        suite_manager: TestSuiteManager | None = None,
        dependencies: dict[str, list[str]] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.suite_manager = suite_manager or TestSuiteManager()
        self.dependencies = dependencies or dict(self.DEFAULT_DEPENDENCIES)
        self.logger = logger or logging.getLogger(__name__)
        self._execution_log: list[dict[str, Any]] = []

    def resolve_execution_order(self, modules: list[str]) -> list[str]:
        """
        Topological sort of *modules* respecting their dependency graph.
        Modules not in the graph are appended at the end.
        """
        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []
        modules_set = set(modules)

        def _visit(m: str) -> None:
            if m in visited:
                return
            if m in visiting:
                raise ValueError(f"Cyclic test dependencies at {m}")
            visiting.add(m)
            for dep in self.dependencies.get(m, []):
                if dep in modules_set:
                    _visit(dep)
            visiting.remove(m)
            visited.add(m)
            order.append(m)

        for m in modules:
            _visit(m)

        return order

    def plan_execution(
        self,
        suite_name: str = "unit",
        modules: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build an execution plan: ordered modules, suite config, and
        estimated duration.
        """
        suite = self.suite_manager.get_suite(suite_name)
        if suite is None:
            raise ValueError(f"Suite '{suite_name}' not found")

        target_modules = (
            modules
            or suite.modules
            or [module.name for module in discover_geo_infer_modules()]
        )
        ordered = self.resolve_execution_order(target_modules)

        plan: dict[str, Any] = {
            "suite": suite.name,
            "description": suite.description,
            "execution_order": ordered,
            "total_modules": len(ordered),
            "timeout_seconds": suite.timeout_seconds,
            "test_patterns": suite.test_patterns,
            "markers": suite.markers,
        }
        self.logger.info(
            "Execution plan for '%s': %d modules", suite_name, len(ordered)
        )
        return plan

    def execute_plan(
        self,
        plan: dict[str, Any],
        runner_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a plan by calling *runner_fn(module)* for each module
        in order. Uses canonical category execution when no runner is supplied.
        """
        results: dict[str, dict[str, Any]] = {}
        start = time.monotonic()
        if not plan["execution_order"]:
            raise ValueError("Cannot execute an empty test plan")

        if runner_fn is None and plan["suite"] not in {
            "unit",
            "slow",
            "integration",
            "performance",
            "system",
        }:
            raise ValueError("Custom suites require an explicit runner_fn")

        for module in plan["execution_order"]:
            mod_start = time.monotonic()
            try:
                if runner_fn:
                    mod_result = runner_fn(module)
                else:
                    report = run_module_category_tests(
                        plan["suite"],
                        plan["timeout_seconds"],
                        modules=[module_by_name(module)],
                    )
                    mod_result = {
                        "status": "passed" if report.success else "failed",
                        "receipts": [result.receipt for result in report.results],
                    }
                if mod_result.get("status") not in {"passed", "failed", "error"}:
                    raise ValueError(
                        f"Runner returned an unsupported outcome: {mod_result.get('status')}"
                    )
                mod_result["duration"] = time.monotonic() - mod_start
                results[module] = mod_result
                self.logger.info(
                    "Module %s: %s (%.2fs)",
                    module,
                    mod_result.get("status", "unknown"),
                    mod_result["duration"],
                )
            except Exception as exc:
                results[module] = {
                    "status": "error",
                    "error": str(exc),
                    "duration": time.monotonic() - mod_start,
                }
                self.logger.error("Module %s failed: %s", module, exc)

        total_duration = time.monotonic() - start
        execution_report = {
            "suite": plan["suite"],
            "success": bool(results)
            and all(result["status"] == "passed" for result in results.values()),
            "module_results": results,
            "total_modules": len(results),
            "total_duration": total_duration,
            "passed": sum(1 for r in results.values() if r.get("status") == "passed"),
            "failed": sum(
                1 for r in results.values() if r.get("status") in ("failed", "error")
            ),
        }
        self._execution_log.append(execution_report)
        return execution_report

    def get_execution_history(self) -> list[dict[str, Any]]:
        return list(self._execution_log)
