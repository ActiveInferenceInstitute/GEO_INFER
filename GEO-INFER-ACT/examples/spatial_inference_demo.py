#!/usr/bin/env python3
"""Thin wrapper for the spatial Active Inference scenario."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned spatial scenario."""
    parser: argparse.ArgumentParser = build_parser(prog="spatial_inference_demo.py")
    return run_scenario_entrypoint("spatial", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
