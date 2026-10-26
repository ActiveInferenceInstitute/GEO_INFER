#!/usr/bin/env python3
"""Audit every resolved universal lock dependency; no target-package installs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.dependency_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
