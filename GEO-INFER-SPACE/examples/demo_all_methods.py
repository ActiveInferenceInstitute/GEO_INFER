#!/usr/bin/env python3
"""Run a small native H3/UTC composition example with explicit invariants.

The historical filename is retained for command compatibility. This example
checks the operations below; it is not an all-method or optional-backend audit.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta


def run_demo(*, resolution: int = 8, max_cells: int = 100) -> dict:
    """Check real H3 topology, extensive aggregation and ordered UTC alignment."""
    import numpy as np
    import pandas as pd

    from geo_infer_space import H3StateSpace, align_h3_observations
    from geo_infer_space.backends.h3.core import H3Cell, H3Grid
    from geo_infer_space.core.spatial_methods import SpatialMethods

    cell = H3Cell.from_coordinates(37.7749, -122.4194, resolution)
    if resolution == 15:
        raise ValueError("The aggregation example requires resolution below fifteen")
    children = cell.children(max_cells=max_cells)
    grid = H3Grid(children, name="owned_demo")
    methods = SpatialMethods()
    disaggregated = methods.disaggregate_to_cells(
        [cell.index], [42.0], resolution + 1, max_cells=max_cells
    )["disaggregated"]
    aggregated = methods.aggregate_to_region(
        list(disaggregated), list(disaggregated.values()), resolution, "sum"
    )["aggregated"][cell.index]["value"]
    if not math.isclose(aggregated, 42.0, rel_tol=1e-12):
        raise AssertionError("Extensive aggregation did not conserve the total")

    # Deliberately preserve a nonlexical state order and a missing observation.
    domain = H3StateSpace(
        list(reversed([child.index for child in children])), max_cells=max_cells
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    axis = [start, start + timedelta(hours=1)]
    source = pd.DataFrame(
        {
            "cell": [domain.cells[0], domain.cells[-1]],
            "timestamp": ["2023-12-31T16:00:00-08:00", axis[1]],
            "value": [0.0, 4.0],
        }
    )
    aligned = align_h3_observations(source, state_space=domain, timestamps=axis)
    if tuple(aligned.data.columns) != domain.cells:
        raise AssertionError("Alignment changed state order")
    if aligned.data.iloc[0, 0] != 0 or not math.isnan(aligned.data.iloc[0, -1]):
        raise AssertionError("Observed zero and missing observations were confused")
    stay, diffuse = domain.transitions()
    if not np.allclose(np.asarray(diffuse.sum(axis=0)), 1):
        raise AssertionError("Transition probabilities did not conserve mass")
    return {
        "backend": "native_h3_cpu",
        "checks": [
            "real_children",
            "extensive_total",
            "utc_alignment",
            "state_order",
            "missing_and_zero",
            "stochastic_transitions",
        ],
        "resolution": resolution,
        "child_count": len(grid),
        "total": aggregated,
        "state_order": list(domain.cells),
        "timestamps": [instant.isoformat() for instant in aligned.timestamps],
        "observed_count": int(aligned.data.notna().sum().sum()),
        "missing_count": int(aligned.data.isna().sum().sum()),
        "transition_shapes": [list(stay.shape), list(diffuse.shape)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=100)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_demo(resolution=args.resolution, max_cells=args.max_cells), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
