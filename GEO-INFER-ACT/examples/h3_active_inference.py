#!/usr/bin/env python3
"""Thin wrapper for the H3 Active Inference scenario."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.h3 import (
    generate_realistic_environmental_observations,
    run_h3_active_inference,
    setup_san_francisco_boundary,
)
from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint

__all__ = [
    "generate_realistic_environmental_observations",
    "run_h3_active_inference",
    "setup_san_francisco_boundary",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned H3 scenario."""
    parser: argparse.ArgumentParser = build_parser(prog="h3_active_inference.py")
    return run_scenario_entrypoint("h3", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
