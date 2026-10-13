# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run a named GNN agent/process pair through real pymdp closed-loop inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.runners.model_pair import (
    PairDesign,
    check_pair_edition,
    create_pair_edition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path)
    parser.add_argument("--process", type=Path)
    parser.add_argument("--initial-state", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--agent-seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--replay", action="store_true")
    arguments = parser.parse_args()
    if arguments.check is not None:
        if any(
            value is not None
            for value in (
                arguments.agent,
                arguments.process,
                arguments.output,
                arguments.initial_state,
                arguments.steps,
                arguments.seeds,
                arguments.agent_seed,
            )
        ):
            parser.error("--check excludes model and output arguments")
        print(
            check_pair_edition(
                arguments.check, replay=arguments.replay
            ).model_dump_json()
        )
        return
    if arguments.replay or any(
        value is None
        for value in (
            arguments.agent,
            arguments.process,
            arguments.output,
            arguments.initial_state,
        )
    ):
        parser.error(
            "execution requires --agent, --process, --initial-state and --output"
        )
    design = PairDesign(
        initial_state=arguments.initial_state,
        steps=48 if arguments.steps is None else arguments.steps,
        seeds=(7,) if arguments.seeds is None else tuple(arguments.seeds),
        agent_seed=0 if arguments.agent_seed is None else arguments.agent_seed,
    )
    print(
        create_pair_edition(
            GNNArtifact.load(arguments.agent),
            GNNArtifact.load(arguments.process),
            design,
            arguments.output,
        )
    )


if __name__ == "__main__":
    main()
