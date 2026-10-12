"""Shared helper for thin scenario entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Iterable

from geo_infer_act.runners.cli import build_parser, config_from_args
from geo_infer_act.runners.scenarios import run_all_scenarios, run_scenario


def run_scenario_entrypoint(
    default_scenario: str,
    argv: Iterable[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Run a package-owned scenario implementation from an entrypoint.

    Args:
        default_scenario: Scenario used when ``--scenario`` is not passed.
        argv: CLI arguments; defaults to ``sys.argv[1:]``.
        parser: Parser from :func:`build_parser`; its ``prog`` is recorded as
            the manifest command. A default parser is built when omitted.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if parser is None:
        parser = build_parser(
            default_all=default_scenario == "all", prog="geo-infer-act-run"
        )
    parser.set_defaults(scenario=default_scenario)
    args = parser.parse_args(args_list)
    config = config_from_args(args)
    if "--scenario" not in args_list:
        config.scenario = default_scenario
    command = [parser.prog, *args_list]

    manifest_path: Path
    if config.scenario == "all":
        all_result = run_all_scenarios(
            output_dir=config.output_dir,
            scenarios=["simple", "spatial"] if args.quick else None,
            seed=config.seed,
            timesteps=config.timesteps,
            deterministic=config.deterministic,
            visualizations=config.visualizations,
            command=command,
        )
        manifest_path = all_result.manifest_path
    else:
        scen_result = run_scenario(config, command=command)
        manifest_path = scen_result.manifest_path

    payload = {"manifest": str(manifest_path)}
    print(json.dumps(payload) if args.json else f"Manifest: {manifest_path}")
    return 0
