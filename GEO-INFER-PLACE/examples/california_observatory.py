"""Create-only entry point for the California statewide monitoring example."""

from __future__ import annotations

import argparse
from pathlib import Path

from geo_infer_place.locations.california.acquisition import (
    acquire_sources,
    check_sources,
)
from geo_infer_place.locations.california.contracts import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--acquire", type=Path)
    parser.add_argument("--check-sources", type=Path)
    parser.add_argument("--check-experiment", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--write-models", type=Path)
    parser.add_argument("--landscape", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.check_experiment is not None:
        if any(
            value is not None
            for name, value in vars(args).items()
            if name != "check_experiment"
        ):
            parser.error("--check-experiment excludes all other inputs")
        from geo_infer_place.locations.california.monitoring import check_experiment

        print(check_experiment(args.check_experiment).model_dump_json())
        return 0
    if args.check_sources is not None:
        if any(
            value is not None
            for name, value in vars(args).items()
            if name != "check_sources"
        ):
            parser.error("--check-sources excludes all other inputs")
        print(check_sources(args.check_sources).model_dump_json())
        return 0
    operations = tuple(
        name
        for name in ("acquire", "prepare", "write_models", "run", "report")
        if getattr(args, name) is not None
    )
    if len(operations) != 1:
        parser.error("choose exactly one create-only operation")
    operation = operations[0]
    required = {
        "acquire": {"design", "acquire"},
        "prepare": {"design", "sources", "prepare"},
        "write_models": {"design", "write_models"},
        "run": {"landscape", "models", "run"},
        "report": {"landscape", "experiment", "report"},
    }[operation]
    supplied = {name for name, value in vars(args).items() if value is not None}
    if supplied != required:
        parser.error(
            f"{operation} requires exactly: {', '.join('--' + field.replace('_', '-') for field in sorted(required))}"
        )
    match operation:
        case "acquire":
            print(acquire_sources(load_config(args.design), args.acquire))
        case "prepare":
            from geo_infer_place.locations.california.geometry import prepare_landscape

            print(
                prepare_landscape(load_config(args.design), args.sources, args.prepare)
            )
        case "write_models":
            from geo_infer_place.locations.california.models import write_model_sources

            print(write_model_sources(load_config(args.design), args.write_models))
        case "run":
            from geo_infer_place.locations.california.monitoring import run_experiment

            print(run_experiment(args.landscape, args.models, args.run))
        case "report":
            from geo_infer_place.locations.california.report import build_report

            print(build_report(args.landscape, args.experiment, args.report))
        case _:
            parser.error("unsupported California operation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
