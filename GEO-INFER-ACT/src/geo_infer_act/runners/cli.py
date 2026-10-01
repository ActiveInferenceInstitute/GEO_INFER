"""Command-line entrypoints for Active Inference scenario runners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Iterable

from geo_infer_act.runners.contracts import SCENARIO_NAMES, RunConfig
from geo_infer_act.runners.scenarios import (
    load_run_config,
    run_all_scenarios,
    run_scenario,
)


def build_parser(default_all: bool = False) -> argparse.ArgumentParser:
    """Build the shared ACT runner CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run GEO-INFER-ACT Active Inference scenarios."
    )
    parser.add_argument(
        "--scenario",
        default="all" if default_all else None,
        help=f"Scenario to run. Choices: all, {', '.join(SCENARIO_NAMES)}",
    )
    parser.add_argument("--config", type=Path, help="Versioned YAML run config.")
    parser.add_argument("--output-dir", type=Path, help="Output directory.")
    parser.add_argument("--seed", type=int, help="Deterministic random seed.")
    parser.add_argument("--timesteps", type=int, help="Number of inference steps.")
    parser.add_argument(
        "--h3-resolution",
        type=int,
        help="H3 resolution for H3 scenarios.",
    )
    parser.add_argument(
        "--h3-ring-size",
        type=int,
        help="H3 grid disk radius for H3 scenarios.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use seeded stochastic policy selection instead of deterministic mode.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable visualization artifacts.",
    )
    parser.add_argument(
        "--research-profile",
        action="store_true",
        help="Use the deterministic non-degenerate H3 research profile.",
    )
    parser.add_argument(
        "--nested-h3",
        action="store_true",
        help="Enable nested H3 hierarchy outputs for H3/spatial scenarios.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the run manifest path as JSON.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a compact simple+spatial suite when scenario is all.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Create a ``RunConfig`` from parsed CLI flags."""
    overrides = {
        "scenario": args.scenario,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "h3_resolution": args.h3_resolution,
        "h3_ring_size": args.h3_ring_size,
        "deterministic": False if args.stochastic else None,
        "visualizations": False if args.no_visualizations else None,
    }
    config = load_run_config(args.config, overrides=overrides)
    if args.research_profile:
        config.parameters["research_profile"] = True
    if args.nested_h3:
        config.parameters["nested_h3"] = True
    return config


def main(argv: Iterable[str] | None = None) -> int:
    """Run one scenario or a full scenario suite."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    command = ["geo-infer-act-run", *(list(argv) if argv is not None else sys.argv[1:])]
    if config.scenario == "all":
        suite = run_all_scenarios(
            output_dir=config.output_dir,
            scenarios=["simple", "spatial"] if args.quick else None,
            seed=config.seed,
            timesteps=config.timesteps,
            deterministic=config.deterministic,
            visualizations=config.visualizations,
            command=command,
        )
        payload = {"manifest": str(suite.manifest_path)}
        message = f"Suite manifest: {suite.manifest_path}"
    else:
        result = run_scenario(config, command=command)
        payload = {"manifest": str(result.manifest_path)}
        message = f"Run manifest: {result.manifest_path}"

    print(json.dumps(payload) if args.json else message)
    return 0


def run_all_main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the full examples suite."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if "--scenario" not in args_list:
        args_list = ["--scenario", "all", *args_list]
    return main(args_list)


if __name__ == "__main__":
    raise SystemExit(main())
