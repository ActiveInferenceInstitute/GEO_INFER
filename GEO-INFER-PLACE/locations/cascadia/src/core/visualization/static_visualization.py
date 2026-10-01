"""
Static visualization for Cascadia H3 data.
Provides simple, lightweight plots without heavy dependencies.
"""

import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def create_static_plots(backend, output_dir: Path) -> dict[str, str]:
    """
    Create simple static plots for Cascadia data.

    Args:
        backend: Unified backend with processed data
        output_dir: Output directory for plots

    Returns:
        Dictionary with paths to generated plots
    """
    logger.info("Creating static visualization plots...")

    results = {}

    try:
        # Get data from backend
        unified_data = backend.unified_data
        redevelopment_scores = backend.calculate_agricultural_redevelopment_potential()

        # Create summary statistics
        summary_stats = create_summary_statistics(unified_data, redevelopment_scores)

        # Save summary statistics
        stats_path = output_dir / "cascadia_summary_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        results["summary_statistics"] = str(stats_path)

        # Create data export for external tools
        export_data = create_data_export(unified_data, redevelopment_scores)

        # Save as JSON
        data_path = output_dir / "cascadia_visualization_data.json"
        with open(data_path, "w") as f:
            json.dump(export_data, f, indent=2)
        results["visualization_data"] = str(data_path)

        # Create CSV export
        csv_path = output_dir / "cascadia_visualization_data.csv"
        df = pd.DataFrame(export_data["points"])
        df.to_csv(csv_path, index=False)
        results["visualization_csv"] = str(csv_path)

        logger.info("Static visualizations created successfully")

    except Exception as e:
        logger.error(f"Failed to create static visualizations: {e}")
        results["error"] = str(e)

    return results


def create_summary_statistics(unified_data: dict, redevelopment_scores: dict) -> dict:
    """
    Create summary statistics for visualization.

    Args:
        unified_data: H3 unified data
        redevelopment_scores: Redevelopment scores

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Creating summary statistics...")

    # Import H3 utilities
    from h3 import cell_to_latlng

    stats = {
        "total_hexagons": len(unified_data),
        "modules": {},
        "redevelopment_scores": {
            "min": float("inf"),
            "max": float("-inf"),
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        },
        "geographic_bounds": {
            "min_lat": float("inf"),
            "max_lat": float("-inf"),
            "min_lng": float("inf"),
            "max_lng": float("-inf"),
        },
    }

    # Calculate statistics
    redevelopment_values = []
    coordinates = []

    for h3_id, hex_data in unified_data.items():
        try:
            # Get coordinates
            lat, lng = cell_to_latlng(h3_id)
            coordinates.append((lat, lng))

            # Update geographic bounds
            stats["geographic_bounds"]["min_lat"] = min(stats["geographic_bounds"]["min_lat"], lat)
            stats["geographic_bounds"]["max_lat"] = max(stats["geographic_bounds"]["max_lat"], lat)
            stats["geographic_bounds"]["min_lng"] = min(stats["geographic_bounds"]["min_lng"], lng)
            stats["geographic_bounds"]["max_lng"] = max(stats["geographic_bounds"]["max_lng"], lng)

            # Get redevelopment score
            score_data = redevelopment_scores.get(h3_id, {})
            composite_score = (
                score_data.get("composite_score", 0.0) if isinstance(score_data, dict) else 0.0
            )
            redevelopment_values.append(composite_score)

            # Calculate module statistics
            for module_name in ["zoning", "current_use", "ownership", "improvements"]:
                if module_name not in stats["modules"]:
                    stats["modules"][module_name] = {
                        "total_cells": 0,
                        "cells_with_data": 0,
                        "min_score": float("inf"),
                        "max_score": float("-inf"),
                        "mean_score": 0.0,
                    }

                module_data = hex_data.get(module_name, {})
                if isinstance(module_data, dict):
                    stats["modules"][module_name]["total_cells"] += 1
                    if module_data:  # treat presence as coverage; score may be zero legitimately
                        score = float(module_data.get("score", 0.0))
                        stats["modules"][module_name]["cells_with_data"] += 1
                        stats["modules"][module_name]["min_score"] = min(
                            stats["modules"][module_name]["min_score"], score
                        )
                        stats["modules"][module_name]["max_score"] = max(
                            stats["modules"][module_name]["max_score"], score
                        )

        except Exception as e:
            logger.warning(f"Could not process H3 cell {h3_id}: {e}")
            continue

    # Calculate redevelopment score statistics
    if redevelopment_values:
        stats["redevelopment_scores"]["min"] = min(redevelopment_values)
        stats["redevelopment_scores"]["max"] = max(redevelopment_values)
        stats["redevelopment_scores"]["mean"] = sum(redevelopment_values) / len(
            redevelopment_values
        )
        stats["redevelopment_scores"]["median"] = sorted(redevelopment_values)[
            len(redevelopment_values) // 2
        ]
        stats["redevelopment_scores"]["std"] = (
            sum((x - stats["redevelopment_scores"]["mean"]) ** 2 for x in redevelopment_values)
            / len(redevelopment_values)
        ) ** 0.5

    # Calculate module statistics
    for module_stats in stats["modules"].values():
        if module_stats["cells_with_data"] > 0:
            module_stats["coverage_percentage"] = (
                module_stats["cells_with_data"] / module_stats["total_cells"]
            ) * 100
        else:
            module_stats["coverage_percentage"] = 0.0

    logger.info(f"Summary statistics created for {len(unified_data)} hexagons")
    return stats


def create_data_export(unified_data: dict, redevelopment_scores: dict) -> dict:
    """
    Create data export for external visualization tools.

    Args:
        unified_data: H3 unified data
        redevelopment_scores: Redevelopment scores

    Returns:
        Dictionary with exported data
    """
    logger.info("Creating data export for external visualization...")

    # Import H3 utilities
    from h3 import cell_to_latlng

    export_data = {
        "metadata": {
            "total_points": len(unified_data),
            "description": "Cascadia Agricultural Analysis Data",
            "coordinate_system": "WGS84",
            "data_format": "H3 Hexagons",
        },
        "points": [],
    }

    # Sample data for performance (every 5th cell)
    count = 0
    for h3_id, hex_data in unified_data.items():
        if count % 5 == 0:  # Sample every 5th cell
            try:
                # Get coordinates
                lat, lng = cell_to_latlng(h3_id)

                # Get redevelopment score
                score_data = redevelopment_scores.get(h3_id, {})
                composite_score = (
                    score_data.get("composite_score", 0.0) if isinstance(score_data, dict) else 0.0
                )

                # Extract module scores
                module_scores = {}
                for module_name in ["zoning", "current_use", "ownership", "improvements"]:
                    module_data = hex_data.get(module_name, {})
                    if isinstance(module_data, dict):
                        module_scores[module_name] = module_data.get("score", 0.0)
                    else:
                        module_scores[module_name] = 0.0

                # Create data point
                point = {
                    "h3_id": h3_id,
                    "latitude": lat,
                    "longitude": lng,
                    "redevelopment_score": composite_score,
                    "module_scores": module_scores,
                    "total_coverage": sum(module_scores.values()),
                }
                export_data["points"].append(point)

            except Exception as e:
                logger.warning(f"Could not process H3 cell {h3_id}: {e}")
                continue

        count += 1

    logger.info(f"Data export created with {len(export_data['points'])} sampled points")
    return export_data
