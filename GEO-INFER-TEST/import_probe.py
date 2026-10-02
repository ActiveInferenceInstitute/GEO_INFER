"""Bound import processes and require evidence that their final checks ran."""

from __future__ import annotations

import json
import math
from pathlib import Path
import secrets
import importlib.util
import sys
import subprocess


# The build-smoke/import-portability lane intentionally has no GEO runtime.
# Load the owning stdlib-only helper without importing the package __init__.
_process_spec = importlib.util.spec_from_file_location(
    "geo_infer_test_process_standalone",
    Path(__file__).parent / "src" / "geo_infer_test" / "process.py",
)
assert _process_spec is not None and _process_spec.loader is not None
_process = importlib.util.module_from_spec(_process_spec)
sys.modules[_process_spec.name] = _process
_process_spec.loader.exec_module(_process)
run_process = _process.run_process


def run_import_probe(
    command: list[str],
    *,
    package: str,
    timeout: float,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Append a receipt token, run the probe, and reject incomplete zero exits.

    Probe code must print a JSON object containing ``probe_token`` (its final
    argument), ``package`` and ``status: ok`` only after completing all checks.
    Timeout errors retain captured bytes, matching subprocess.run diagnostics.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Import timeout must be finite and positive")
    token = secrets.token_hex(24)
    arguments = [*command, token]
    result = run_process(arguments, timeout=timeout, cwd=cwd, env=env)
    stdout = result.stdout
    result.check_returncode()
    receipts = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get("probe_token") == token:
            receipts.append(value)
    if (
        len(receipts) != 1
        or receipts[0].get("package") != package
        or receipts[0].get("status") != "ok"
    ):
        raise ValueError(f"Import probe for {package} omitted its completion receipt")
    return result
