#!/usr/bin/env python3
"""GEO-INFER-COG module orchestrator.

Runs one documented end-to-end cognitive-modeling operation on synthetic
data: feed a synthetic map (points and polygons with visual properties)
through the ``SpatialPerceptionModel`` perceptual pipeline — element
extraction, visual saliency, Gestalt grouping, and attention allocation —
and report the saliency and attention results. All work goes through the
real ``geo_infer_cog`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_cog import SpatialPerceptionModel

    # Synthetic cognitive-map observations: landmarks and regions a wayfinder
    # would perceive on a fictional downtown map.
    spatial_data: dict[str, Any] = {
        "geometries": [
            {
                "type": "Point",
                "coordinates": [-123.10, 44.95],
                "properties": {"color": "red", "label": "clock_tower"},
            },
            {
                "type": "Point",
                "coordinates": [-123.08, 44.96],
                "properties": {"color": "blue", "label": "bus_stop"},
            },
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-123.12, 44.94],
                        [-123.09, 44.94],
                        [-123.09, 44.97],
                        [-123.12, 44.97],
                        [-123.12, 44.94],
                    ]
                ],
                "properties": {"color": "green", "label": "city_park"},
            },
            {
                "type": "Point",
                "coordinates": [-123.11, 44.945],
                "properties": {"color": "orange", "label": "transit_hub"},
            },
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-123.07, 44.93],
                        [-123.05, 44.93],
                        [-123.05, 44.96],
                        [-123.07, 44.96],
                        [-123.07, 44.93],
                    ]
                ],
                "properties": {"color": "yellow", "label": "market_district"},
            },
        ]
    }
    context = {"task_type": "wayfinding", "environment": "urban"}

    perception = SpatialPerceptionModel(
        framework="bayesian_attention", resolution="adaptive"
    )
    result = perception.process_spatial_input(spatial_data, context=context)

    saliency_by_id = {
        element["element_id"]: round(float(element["visual_saliency"]), 4)
        for element in result["spatial_elements"]
    }
    attention = {
        key: round(float(weight), 4)
        for key, weight in result["attention_weights"].items()
    }
    top_element = max(attention, key=lambda key: attention[key])

    return {
        "operation": "spatial_perception_attention_allocation",
        "framework": "bayesian_attention",
        "elements_processed": len(result["spatial_elements"]),
        "grouped_elements": len(result["grouped_elements"]),
        "visual_saliency_by_element": saliency_by_id,
        "attention_weights": attention,
        "top_attended_element": top_element,
        "n_insights": len(result["perceptual_insights"]),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("COG", _operation))
