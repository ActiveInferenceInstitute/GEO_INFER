#!/usr/bin/env python3
"""GEO-INFER-SPACE module orchestrator.

Runs one documented end-to-end SPACE operation on synthetic data: build an
H3 grid over a synthetic region at resolution 7, compact the covering set,
verify neighborhood topology, and measure grid distances between anchor
cells. All work goes through the real ``geo_infer_space`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_space import SpatialIndexingInterface, polygon_to_cells

    indexer = SpatialIndexingInterface()

    # Synthetic region: a hexagon-ish bounding region around a fictional
    # study site (Willamette Valley synthetic plot).
    region = {
        "type": "Polygon",
        "coordinates": [
            [
                [-123.1, 44.9],
                [-122.9, 44.9],
                [-122.8, 45.1],
                [-123.0, 45.2],
                [-123.2, 45.1],
                [-123.1, 44.9],
            ]
        ],
    }

    cells = polygon_to_cells(region, resolution=7)
    if not cells:
        raise RuntimeError("polygon_to_cells returned no cells for the region")

    compacted = indexer.compact_cells(cells)

    anchor = cells[0]
    neighbors = indexer.get_cell_neighbors(anchor, k=1)
    anchor_latlng = indexer.cell_to_latlng(anchor)
    distances = {
        other: indexer.get_cell_distance(anchor, other) for other in cells[1:6]
    }

    return {
        "operation": "h3_grid_build_and_topology_check",
        "resolution": 7,
        "region_cells": len(cells),
        "compacted_cells": len(compacted),
        "anchor_cell": anchor,
        "anchor_center": [anchor_latlng[0], anchor_latlng[1]],
        "anchor_neighbor_count": len(neighbors),
        "distances_from_anchor": distances,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("SPACE", _operation))
