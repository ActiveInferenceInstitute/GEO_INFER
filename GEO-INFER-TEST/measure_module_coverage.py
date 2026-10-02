#!/usr/bin/env python3
"""Measure canonical unit/integration coverage; --modules and --json select output."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.coverage import main

if __name__ == "__main__":
    raise SystemExit(main())
