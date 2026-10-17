#!/usr/bin/env python3
"""Profile a real Python backend through the owning TEST profiling API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.profiling import main

if __name__ == "__main__":
    raise SystemExit(main())
