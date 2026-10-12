#!/usr/bin/env python3
"""GEO-INFER-SIM module orchestrator.

Runs one documented end-to-end SIM operation on synthetic data: seed a
12x12 cellular-automata grid with a glider pattern and run 30 deterministic
Game-of-Life steps through the module's ``CellularAutomata`` paradigm,
tracking population and active-cell dynamics across the run. All work goes
through the real ``geo_infer_sim`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_sim import CellularAutomata

    grid_shape = (12, 12)
    initial = np.zeros(grid_shape, dtype=int)
    # Seed a glider in the upper-left quadrant (period-4 spaceship).
    glider = [(1, 2), (2, 0), (2, 1), (2, 2), (3, 1)]
    for row, col in glider:
        initial[row, col] = 1

    ca = CellularAutomata(
        grid_shape=grid_shape,
        initial_states=initial,
        num_states=2,
        random_seed=7,
    )

    n_steps = 30
    populations: list[int] = []
    for _ in range(n_steps):
        ca.step()  # documented default rule: Conway's Game of Life
        populations.append(int(np.sum(ca.grid)))

    final_state = ca.get_state()
    return {
        "operation": "game_of_life_cellular_automata_run",
        "grid_shape": list(grid_shape),
        "initial_population": len(glider),
        "n_steps": n_steps,
        "population_first_step": populations[0],
        "population_final_step": populations[-1],
        "population_max": max(populations),
        "population_min": min(populations),
        "final_state_counts": final_state["state_counts"],
        "final_time": float(final_state["time"]),
        "history_snapshots": len(ca.history),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("SIM", _operation))
