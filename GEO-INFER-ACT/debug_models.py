#!/usr/bin/env python3
"""Thin wrapper for the ACT debug scenario."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned debug scenario."""
    parser: argparse.ArgumentParser = build_parser(prog="debug_models.py")
    return run_scenario_entrypoint("debug", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
