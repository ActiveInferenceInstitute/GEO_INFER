#!/usr/bin/env python3
"""Thin wrapper for the urban-planning Active Inference scenario."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned urban-planning scenario."""
    parser: argparse.ArgumentParser = build_parser(prog="urban_planning.py")
    return run_scenario_entrypoint("urban_planning", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
