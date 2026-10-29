#!/usr/bin/env python3
"""Launch a workload with the explicit Linux native I/O lookup profile."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.native_io_environment import main

if __name__ == "__main__":
    raise SystemExit(main())
