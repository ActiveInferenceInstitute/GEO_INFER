#!/usr/bin/env python3
"""Thin wrapper for the complete ACT examples suite."""

from __future__ import annotations

import argparse

from geo_infer_act.runners.cli import build_parser
from geo_infer_act.runners.wrapper import run_scenario_entrypoint


def main(argv: list[str] | None = None) -> int:
    """Run all package-owned ACT scenarios."""
    parser: argparse.ArgumentParser = build_parser(prog="run_all_examples.py")
    return run_scenario_entrypoint("all", argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
