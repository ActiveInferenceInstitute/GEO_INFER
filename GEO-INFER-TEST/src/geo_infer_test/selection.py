"""Record full collection inventories, including xdist workers before filtering."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_selection: dict = {}


def pytest_configure(config: pytest.Config) -> None:
    global _selection
    _selection = {
        "collected": [],
        "deselected": [],
        "executed": [],
        "workers": {},
        "errors": [],
    }
    config._geo_selection = _selection


def pytest_itemcollected(item: pytest.Item) -> None:
    item.config._geo_selection["collected"].append(item.nodeid)


def pytest_deselected(items: list[pytest.Item]) -> None:
    for item in items:
        item.config._geo_selection["deselected"].append(item.nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "setup" and report.nodeid not in _selection["executed"]:
        _selection["executed"].append(report.nodeid)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: object, error: object) -> None:
    """Receive each worker's pre-filter collection rather than xdist's filtered IDs."""
    worker = getattr(node, "workeroutput", {}).get("geo_infer_selection")
    worker_id = getattr(getattr(node, "gateway", None), "id", "unknown")
    if error or worker is None:
        _selection["errors"].append(
            f"worker {worker_id} failed or omitted its inventory: {error}"
        )
        return
    _selection["workers"][worker_id] = worker
    for key in ("collected", "deselected"):
        _selection[key] = sorted(set(_selection[key]) | set(worker[key]))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    selection = session.config._geo_selection
    selection["selected"] = [item.nodeid for item in session.items]
    selection["exitstatus"] = int(exitstatus)
    if hasattr(session.config, "workerinput"):
        session.config.workeroutput["geo_infer_selection"] = selection
        return
    workers = list(selection["workers"].values())
    if getattr(session.config.option, "numprocesses", 0):
        if not workers:
            selection["errors"].append("xdist supplied no complete worker inventories")
        else:
            reference = workers[0]
            for key in ("collected", "deselected", "selected"):
                if any(set(worker[key]) != set(reference[key]) for worker in workers):
                    selection["errors"].append(
                        f"xdist worker {key} inventories disagree"
                    )
            selection["selected"] = sorted(
                {node for worker in workers for node in worker["selected"]}
            )
    selection["unaccounted"] = sorted(
        set(selection["collected"])
        - set(selection["selected"])
        - set(selection["deselected"])
    )
    target = os.environ.get("GEO_INFER_TEST_SELECTION")
    if target:
        Path(target).write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
