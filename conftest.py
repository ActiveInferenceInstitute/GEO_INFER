"""Strict, shared pytest policy for the GEO-INFER ecosystem."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

pytest_plugins = ["geo_infer_test.testing"]


# Cache and storage layers sign serialized payloads with a key resolved from
# ``GEO_INFER_SERIALIZATION_KEY`` or, failing that, a per-installation key file
# under the developer's config directory. Pinning a deterministic test key
# keeps the suite hermetic: no test creates, reads, or rotates the real key.
SERIALIZATION_KEY_ENV = "GEO_INFER_SERIALIZATION_KEY"
SERIALIZATION_TEST_KEY = "geo-infer-test-suite-serialization-key"


PRIMARY_MARKERS = ("unit", "integration", "system", "performance")
DOMAIN_MARKERS = (
    "slow",
    "core",
    "geospatial",
    "api",
    "reporting",
    "fast",
    "model",
    "reproducibility",
    "artifact",
    "spatial",
)


def pytest_configure(config: pytest.Config) -> None:
    """Register ecosystem-wide markers before collection."""
    # Module-local pyproject files are the nearest pytest configuration for
    # most packages, so the root ``addopts`` cannot be relied on here.
    # Installing the filter in the shared plugin makes the zero-warning policy
    # effective for every module invocation.
    warnings.simplefilter("error")
    config.addinivalue_line("markers", "module: auto-applied module marker")
    for marker in (*PRIMARY_MARKERS, *DOMAIN_MARKERS):
        config.addinivalue_line("markers", f"{marker}: GEO-INFER test taxonomy marker")


@pytest.fixture(autouse=True, scope="session")
def pin_serialization_signing_key() -> None:
    """Pin the payload-signing key so tests never touch the real key file."""
    previous = os.environ.get(SERIALIZATION_KEY_ENV)
    os.environ[SERIALIZATION_KEY_ENV] = SERIALIZATION_TEST_KEY
    yield
    if previous is None:
        os.environ.pop(SERIALIZATION_KEY_ENV, None)
    else:
        os.environ[SERIALIZATION_KEY_ENV] = previous


def _close_imported_matplotlib_figures() -> None:
    """Close test-created figures without importing the optional backend."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(autouse=True)
def isolate_matplotlib_figures():
    """Prevent one test's figures from leaking into later strict-warning tests."""
    yield
    _close_imported_matplotlib_figures()


def _primary_marker_for_path(path: Path) -> str:
    """Infer the required primary marker from the canonical test directory."""
    parts = set(path.parts)
    for marker in PRIMARY_MARKERS:
        if marker in parts:
            return marker
    return "unit"


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply one primary marker to every test and forbid skip controls at runtime."""
    del session, config
    for item in items:
        marker = _primary_marker_for_path(Path(str(item.fspath)))
        # Directory taxonomy is canonical. Remove contradictory explicit or
        # inherited primary markers before adding the one effective marker.
        node = item
        while node is not None:
            node.own_markers = [
                mark for mark in node.own_markers if mark.name not in PRIMARY_MARKERS
            ]
            node = node.parent
        item.add_marker(getattr(pytest.mark, marker))
        item.add_marker(pytest.mark.module)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    """Convert any runtime skip into a failure so skipped tests cannot pass."""
    outcome = yield
    report = outcome.get_result()
    if report.outcome == "skipped":
        report.outcome = "failed"
        report.longrepr = f"Skipped tests are forbidden by the GEO-INFER test contract: {report.longrepr}"


def pytest_terminal_summary(terminalreporter: object) -> None:
    """Fail the session if pytest recorded a skipped or xfailed report."""
    stats = getattr(terminalreporter, "stats", {})
    forbidden = [
        *stats.get("skipped", []),
        *stats.get("xfailed", []),
        *stats.get("xpassed", []),
    ]
    if forbidden:
        raise pytest.UsageError(
            "Skipped/xfail test reports are forbidden by the GEO-INFER test contract: "
            + ", ".join(getattr(report, "nodeid", "unknown") for report in forbidden)
        )
