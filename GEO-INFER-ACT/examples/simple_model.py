#!/usr/bin/env python3
"""Thin wrapper for the simple Active Inference scenario."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned simple scenario."""
    parser: argparse.ArgumentParser = build_parser(prog="simple_model.py")
    return run_scenario_entrypoint("simple", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
