#!/usr/bin/env python3
"""Retain bounded manuscript JSON custody evidence before CI artifact upload."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from geo_infer_test.research_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
