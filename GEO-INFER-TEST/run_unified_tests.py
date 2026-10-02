#!/usr/bin/env python3
"""Run the owning geo_infer_test execution engine.

Options: --module, --category, --h3-migration, --fail-fast, --show-failures,
--list-modules, --timeout, --results-dir. Run --help for the current CLI contract.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Support direct invocation from a source checkout without imposing a sync.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.execution import main

if __name__ == "__main__":
    raise SystemExit(main())
