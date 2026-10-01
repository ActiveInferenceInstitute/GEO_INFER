#!/usr/bin/env python3
"""GEO-INFER-ANT module orchestrator.

Runs one documented end-to-end complex-systems operation on synthetic data:
solve a small synthetic traveling-salesman problem with the module's
``AntColonyOptimization`` (pheromone-mediated swarm optimization) and
cross-check the colony's best path against the brute-force optimum. All
work goes through the real ``geo_infer_ant`` public API.
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _path_length(path: list[int], distances: np.ndarray) -> float:
    """Total open-path length of ``path`` under ``distances``.

    Matches the module's fitness semantics: ``_evaluate_solution`` sums
    consecutive edges only (the path is not closed).
    """
    return float(sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1)))


def _operation() -> dict[str, Any]:
    from geo_infer_ant import AntColonyOptimization

    rng = np.random.default_rng(7)

    # Synthetic foraging landscape: 9 sites on a 100x100 unit plot.
    n_nodes = 9
    coordinates = rng.uniform(0.0, 100.0, (n_nodes, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))

    aco = AntColonyOptimization(
        number_of_ants=12,
        max_iterations=25,
        variant="AS",
        random_seed=42,
    )
    aco.initialize_problem(nodes=list(range(n_nodes)), distance_matrix=distances)
    result = aco.solve()

    # Brute-force optimum over all open Hamiltonian paths, matching the
    # module's open-path fitness semantics: 9! = 362,880 permutations.
    optimum = min(
        _path_length(list(order), distances) for order in permutations(range(n_nodes))
    )

    convergence = result.convergence_history
    return {
        "operation": "ant_colony_tsp_solve",
        "n_nodes": n_nodes,
        "n_ants": 12,
        "max_iterations": 25,
        "best_solution": [int(node) for node in result.best_solution],
        "best_path_length": round(float(result.best_fitness), 4),
        "brute_force_open_path_optimum": round(float(optimum), 4),
        "optimality_gap_percent": round(
            100.0 * (float(result.best_fitness) - optimum) / optimum, 3
        ),
        "iterations_completed": result.iterations_completed,
        "convergence_achieved": result.convergence_achieved,
        "initial_best_fitness": round(float(convergence[0]), 4)
        if convergence
        else None,
        "final_best_fitness": round(float(convergence[-1]), 4) if convergence else None,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ANT", _operation))
