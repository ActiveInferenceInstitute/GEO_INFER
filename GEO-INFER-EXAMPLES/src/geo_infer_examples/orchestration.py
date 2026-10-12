"""Shared engine for the GEO-INFER module orchestrator examples.

Each ``examples/module_orchestrators/<MODULE>/scripts/run_orchestrator.py``
defines one documented end-to-end operation on synthetic data using that
module's primary public API and hands it to :func:`run_module_orchestrator`.
The engine executes the operation, prints structured JSON results on stdout,
and maps failures to deterministic exit codes so shell pipelines and CI can
rely on them:

- ``0``: the operation completed and returned a result dict.
- ``2``: the module's dependencies are missing (``ImportError``); the output
  names the exact ``geo-infer-<module>`` extra to install, verified against
  that module's ``pyproject.toml`` (graceful degradation, not a no-op).
- ``1``: the operation itself failed; the output carries the traceback.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from typing import Any

#: Exit code used when a module's dependencies are not installed.
EXIT_DEPENDENCY_MISSING = 2

#: Exit code used when the orchestrated operation raised.
EXIT_OPERATION_FAILED = 1

#: Verified per-module optional-dependency group used in the
#: ``requires`` hint when a module's imports fail. Values were checked
#: against each ``GEO-INFER-<MODULE>/pyproject.toml``
#: ``[project.optional-dependencies]`` table; modules without any
#: optional group map to ``None`` (plain package install).
MODULE_EXTRAS: dict[str, str | None] = {
    "ACT": "dev",
    "AG": "dev",
    "AGENT": "dev",
    "AI": "dev",
    "ANT": "dev",
    "API": "dev",
    "APP": "dev",
    "ART": "dev",
    "BAYES": "dev",
    "BIO": "dev",
    "CIV": "dev",
    "CLIMATE": "dev",
    "COG": "dev",
    "COMMS": "dev",
    "DATA": "dev",
    "ECON": "dev",
    "EDU": "dev",
    "EMERGENCY": "dev",
    "ENERGY": "dev",
    "FOREST": "test",
    "GIT": "dev",
    "HEALTH": "dev",
    "INSURANCE": "dev",
    "INTRA": "dev",
    "IOT": "dev",
    "LOG": "dev",
    "MATH": "dev",
    "NORMS": "dev",
    "OPS": "dev",
    "ORG": "dev",
    "PEP": "dev",
    "PLACE": "dev",
    "REQ": "dev",
    "RISK": "dev",
    "SEC": "dev",
    "SIM": "dev",
    "SPACE": "dev",
    "SPM": "dev",
    "TEST": "dev",
    "TIME": "dev",
    "TRANSPORT": "dev",
    "MARINE": None,
    "WATER": None,
}


def run_module_orchestrator(
    module_name: str,
    operation: Callable[[], dict[str, Any]],
    requires_extra: str | None = None,
) -> int:
    """Execute one module operation and report it as structured JSON.

    Args:
        module_name: GEO-INFER module short name (e.g. ``"SPACE"``).
        operation: Zero-argument callable running the module's real
            end-to-end operation on synthetic data. Must return a
            JSON-serializable dict of structured results.
        requires_extra: Override for the ``geo-infer-<module>`` optional-
            dependencies group named when dependencies are missing. When
            ``None`` (default), the verified group from :data:`MODULE_EXTRAS`
            is used; modules with no optional groups install the plain
            package (no extra).

    Returns:
        The process exit code: ``0`` on success, ``2`` on missing
        dependencies, ``1`` when the operation raised any other error.
    """
    started = time.perf_counter()
    try:
        result = operation()
    except ImportError as exc:
        payload = {
            "module": module_name,
            "status": "missing-dependency",
            "requires": (
                f"geo-infer-{module_name.lower()}[{extra}]"
                if (extra := requires_extra or MODULE_EXTRAS.get(module_name))
                else f"geo-infer-{module_name.lower()}"
            ),
            "error": str(exc),
        }
        print(json.dumps(payload, indent=2))
        return EXIT_DEPENDENCY_MISSING
    except Exception as exc:
        payload = {
            "module": module_name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OPERATION_FAILED

    if not isinstance(result, dict):
        print(
            json.dumps(
                {
                    "module": module_name,
                    "status": "error",
                    "error": (
                        "orchestrator operation must return a dict, got "
                        f"{type(result).__name__}"
                    ),
                },
                indent=2,
            )
        )
        return EXIT_OPERATION_FAILED

    payload = {
        "module": module_name,
        "status": "ok",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "result": result,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0
