#!/usr/bin/env python3
"""GEO-INFER-MATH module orchestrator.

Runs one documented end-to-end MATH operation on synthetic data: compute
Moran's I and Geary's C spatial autocorrelation statistics for a synthetic
10x10 lattice field (a smooth north-south gradient plus seeded noise) using
a rook-adjacency weights matrix, with a spatially random reference field
for contrast. All work goes through the real ``geo_infer_math`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_math import GearysC, MoranI

    n_side = 10
    n_cells = n_side * n_side
    rng = np.random.default_rng(42)

    # Synthetic lattice field: smooth north-south gradient plus seeded noise.
    rows = np.repeat(np.arange(n_side), n_side).astype(float)
    clustered_field = 5.0 + 0.4 * rows + rng.normal(0.0, 0.5, n_cells)
    random_field = rng.normal(0.0, 1.0, n_cells)

    # Rook-adjacency (4-neighbour) binary weights on the lattice.
    weights = np.zeros((n_cells, n_cells), dtype=float)
    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_side and 0 <= cc < n_side:
                    weights[i, rr * n_side + cc] = 1.0

    moran_clustered = MoranI(weights_matrix=weights).compute(clustered_field)
    geary_clustered = GearysC(
        weights_matrix=weights, rng=42, n_permutations=200
    ).compute(clustered_field)
    moran_random = MoranI(weights_matrix=weights).compute(random_field)

    return {
        "operation": "morans_i_and_gearys_c_on_lattice",
        "lattice_shape": [n_side, n_side],
        "weights_neighbors_total": float(np.sum(weights)),
        "clustered_field": {
            "mean": float(np.mean(clustered_field)),
            "std": float(np.std(clustered_field)),
            "morans_i": {
                "I": float(moran_clustered["I"]),
                "expected_I": float(moran_clustered["expected_I"]),
                "z_score": float(moran_clustered["z_score"]),
                "p_value": float(moran_clustered["p_value"]),
            },
            "gearys_c": {
                "C": float(geary_clustered["C"]),
                "expected_C": float(geary_clustered["expected_C"]),
                "z_score": float(geary_clustered["z_score"]),
                "p_value": float(geary_clustered["p_value"]),
            },
        },
        "random_reference_field": {
            "morans_i": float(moran_random["I"]),
            "z_score": float(moran_random["z_score"]),
            "p_value": float(moran_random["p_value"]),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("MATH", _operation))
