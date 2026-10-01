"""Integration coverage: module orchestrator demo scripts actually execute.

The 43 ``examples/module_orchestrators/<MODULE>/scripts/run_orchestrator.py``
demo scripts were previously never executed by any test or gate. This test
runs a bounded, deterministic subset (fast orchestrators with synthetic-data
operations) end-to-end via subprocess and asserts the engine's success
contract: exit code 0 and JSON output with ``status == "ok"`` on stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_ORCHESTRATORS_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "module_orchestrators"
)

_TIMEOUT_SECONDS = 180

#: Fast, deterministic orchestrators (synthetic data, no network, no heavy
#: Monte Carlo / optimization loops). Each runs in a few seconds.
SUBSET = (
    "SPACE",
    "TIME",
    "LOG",
    "TEST",
    "CIV",
    "DATA",
    "OPS",
    "MATH",
    # Timed individually before admission (all < 3 s, synthetic data,
    # no network): ECON 2.6 s, EDU 0.1 s, EMERGENCY 0.3 s, ENERGY 1.4 s.
    "ECON",
    "EDU",
    "EMERGENCY",
    "ENERGY",
    "REQ",
)


@pytest.mark.parametrize("module", SUBSET)
def test_orchestrator_script_exits_zero_with_ok_status(module: str) -> None:
    """A fast orchestrator script runs end-to-end and reports JSON ok."""
    script = _ORCHESTRATORS_ROOT / module / "scripts" / "run_orchestrator.py"
    assert script.is_file(), f"missing orchestrator script: {script}"

    proc = subprocess.run(
        [sys.executable, str(script)],
        timeout=_TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{module} orchestrator exited {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )

    payload = json.loads(proc.stdout)
    assert payload["module"] == module
    assert payload["status"] == "ok", f"{module} reported: {payload.get('status')}"
    assert isinstance(payload.get("result"), dict)
    assert payload["duration_seconds"] >= 0.0
