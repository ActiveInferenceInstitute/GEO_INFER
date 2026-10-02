#!/usr/bin/env python3
"""Record a validator command through GEO-INFER-TEST's shared attempt engine."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geo_infer_test.execution import record_validation_main

if __name__ == "__main__":
    raise SystemExit(record_validation_main())
