# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: use the GEO-INFER workspace, not an isolated script environment:
# uv run python GEO-INFER-ACT/examples/cascadia/cascadia_study.py --help
"""Replay a bounded Cascadia seed grid under three explicit model assumptions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import platform

from study_contract import (
    Condition,
    RunMeasures,
    SeedOutcome,
    StudyDesign,
    condition_parameters,
    load_study_design,
    summarize,
)


@dataclass(frozen=True, slots=True)
class StudyPaths:
    config: Path
    output: Path
    gnn_repo: Path
    gnn_python: Path


def run_study(paths: StudyPaths, design: StudyDesign) -> Path:
    """Retain inputs and each run before qualifying; never overwrite an edition."""
    import h3
    import numpy as np

    import cascadia_homing as homing
    from study_figure import save_study_figure
    from study_custody import freeze_sources, source_problems
    from geo_infer_act.core.gnn_contract import GNNArtifact
    from geo_infer_act.models.river_homing import compile_river_homing
    from geo_infer_act.runners.closed_loop import run_closed_loop, trace_diagnostics

    config = homing.load_config(paths.config)
    reference, _, provenance = homing.build_model(config)
    output = paths.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    custody = freeze_sources(root, output)
    (output / "reference_config.yaml").write_bytes(paths.config.read_bytes())
    homing.write_json(output / "study_design.json", design.model_dump(mode="json"))
    outcomes = []
    for condition in design.conditions:
        folder = output / condition.value
        folder.mkdir()
        model = compile_river_homing(
            reference.network, condition_parameters(reference.parameters, condition)
        )
        sources = homing.render_sources(model, config, provenance)
        for key in ("source", "state_ids", "domain"):
            (folder / homing.FIXTURE_NAMES[key]).write_text(
                sources[key], encoding="utf-8"
            )
        artifact_path = folder / homing.FIXTURE_NAMES["artifact"]
        export = homing.export_with_gnn(
            folder / homing.FIXTURE_NAMES["source"],
            folder / homing.FIXTURE_NAMES["state_ids"],
            artifact_path,
            gnn_repo=paths.gnn_repo.resolve(),
            gnn_python=paths.gnn_python.absolute(),
            step_seconds=model.parameters.step_seconds,
        )
        artifact = GNNArtifact.load(artifact_path)
        if homing.fixture_problems(model, sources, folder):
            raise RuntimeError("Study GNN export differs from the compiled exact model")
        export["source_sha256"] = homing.sha256_file(
            folder / homing.FIXTURE_NAMES["source"]
        )
        export["artifact_canonical_sha256"] = artifact.digest
        homing.write_json(folder / homing.FIXTURE_NAMES["export"], export)
        for seed in design.seeds:
            run_directory = folder / f"seed_{seed}"
            result = run_closed_loop(
                artifact,
                initial_state=model.network.index()[
                    model.landmarks[config["simulation"]["initial_state"]]
                ],
                steps=design.steps,
                process_seed=seed,
                agent_seed=design.agent_seed,
                start=datetime.fromisoformat(config["simulation"]["start"]),
            )
            diagnostics = trace_diagnostics(artifact, result)
            summary = homing.analyze(model, result, diagnostics)
            homing.write_json(run_directory / "closed_loop.json", result.to_dict())
            homing.write_json(run_directory / "diagnostics.json", diagnostics)
            homing.write_json(run_directory / "summary.json", summary)
            measures = RunMeasures.model_validate(summary)
            problems = measures.qualification_problems()
            outcome = SeedOutcome(
                condition=condition,
                process_seed=seed,
                measures=measures,
                run_directory=str(run_directory.relative_to(output)),
                qualified=not problems,
                problems=problems,
            )
            homing.write_json(
                run_directory / "outcome.json", outcome.model_dump(mode="json")
            )
            outcomes.append(outcome)
            print(
                f"{condition.value} seed={seed} home={measures.reached_home_step} qualified={outcome.qualified}",
                flush=True,
            )
    records = tuple(outcomes)
    summaries = summarize(records)
    homing.write_json(
        output / "outcomes.json", [item.model_dump(mode="json") for item in records]
    )
    homing.write_json(
        output / "summary.json", [item.model_dump(mode="json") for item in summaries]
    )
    save_study_figure(output / "seed_outcomes.png", records)
    save_study_figure(output / "seed_outcomes.svg", records)
    files = {
        str(path.relative_to(output)): homing.sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    complete = len(records) == len(design.seeds) * len(design.conditions)
    drift = source_problems(root, custody)
    qualified = complete and all(item.qualified for item in records) and not drift
    homing.write_json(
        output / "study_manifest.json",
        {
            "schema_version": "geo-infer-act/cascadia-seed-study/1",
            "complete": complete,
            "qualified": qualified,
            "native_claim_ready": False,
            "source_custody": custody.model_dump(mode="json"),
            "source_drift": drift,
            "design": design.model_dump(mode="json"),
            "config_sha256": config["_sha256"],
            "network_sha256": reference.network.digest,
            "flowlines": provenance,
            "versions": {
                "python": platform.python_version(),
                "inferactively-pymdp": version("inferactively-pymdp"),
                "jax": version("jax"),
                "jaxlib": version("jaxlib"),
                "h3": h3.__version__,
                "numpy": np.__version__,
                "pydantic": version("pydantic"),
            },
            "files": files,
        },
    )
    if not qualified:
        raise RuntimeError(
            "Study numerical qualification failed; retained outcomes list the failures"
        )
    return output / "study_manifest.json"


def parse_study_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[StudyPaths, StudyDesign]:
    """Parse the seed grid before any outputs are created."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("cascadia_homing.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gnn-repo", type=Path, required=True)
    parser.add_argument("--gnn-python", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        help="Portable study-design JSON; excludes study overrides",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", choices=list(Condition))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--agent-seed", type=int)
    args = parser.parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in ("seeds", "conditions", "steps", "agent_seed")
        if getattr(args, name) is not None
    }
    if args.design is not None and overrides:
        parser.error(
            "--design cannot be combined with --seeds, --conditions, --steps or --agent-seed"
        )
    try:
        if args.design is not None:
            design = load_study_design(args.design)
        else:
            if "seeds" in overrides:
                overrides["seeds"] = tuple(args.seeds)
            if "conditions" in overrides:
                overrides["conditions"] = tuple(
                    Condition(value) for value in args.conditions
                )
            design = StudyDesign(**overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    paths = StudyPaths(args.config, args.output, args.gnn_repo, args.gnn_python)
    return paths, design


def main(argv: Sequence[str] | None = None) -> int:
    paths, design = parse_study_arguments(argv)
    print(run_study(paths, design))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
