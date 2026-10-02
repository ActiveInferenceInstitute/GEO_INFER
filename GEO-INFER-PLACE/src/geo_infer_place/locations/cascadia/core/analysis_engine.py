#!/usr/bin/env python3
"""
Analysis Engine for Cascadia Agricultural Analysis

This module handles all analysis operations including spatial analysis,
data processing, and result generation.
"""

import logging
from typing import Any

# Import the necessary components
from geo_infer_space.core.spatial_processor import SpatialProcessor

SPACE_CORE_AVAILABLE = True


def perform_enhanced_spatial_analysis(
    backend, spatial_processor: SpatialProcessor
) -> dict[str, Any]:
    """Perform enhanced spatial analysis using SPACE capabilities"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Performing enhanced spatial analysis with SPACE integration...")

    try:
        analysis_results = {
            "spatial_correlations": {},
            "hotspot_analysis": {},
            "buffer_analysis": {},
            "proximity_analysis": {},
            "multi_overlay_analysis": {},
        }

        # Skip spatial correlations - they are handled by the unified backend
        logger.info(
            "📊 Spatial correlations are handled by the unified backend - skipping redundant calculation"
        )

        # Perform hotspot analysis
        try:
            redevelopment_scores = (
                backend.calculate_agricultural_redevelopment_potential()
            )
            if redevelopment_scores:
                # Identify hotspots (areas with high redevelopment potential)
                high_potential = {}
                for h3, score_data in redevelopment_scores.items():
                    if isinstance(score_data, dict):
                        composite_score = score_data.get("composite_score", 0)
                        if composite_score > 0.7:
                            high_potential[h3] = composite_score

                analysis_results["hotspot_analysis"] = {
                    "high_potential_count": len(high_potential),
                    "high_potential_hexagons": list(high_potential.keys()),
                    "hotspot_density": len(high_potential) / len(redevelopment_scores)
                    if redevelopment_scores
                    else 0,
                }
                logger.info(
                    f"🔥 Identified {len(high_potential)} high-potential hotspots"
                )
        except Exception as e:
            logger.warning(f"⚠️ Hotspot analysis failed: {e}")

        # Perform buffer and proximity analysis using H3 cell area lookup
        try:
            # H3 resolution to approximate cell area in km² (from official H3 docs)
            h3_area_km2 = {
                0: 4250546.85,
                1: 607220.98,
                2: 86745.85,
                3: 12392.26,
                4: 1770.32,
                5: 252.90,
                6: 36.13,
                7: 5.16,
                8: 0.7373,
                9: 0.1053,
                10: 0.01505,
                11: 0.002149,
                12: 0.000307,
                13: 0.0000439,
                14: 0.00000627,
                15: 0.000000895,
            }
            resolution = getattr(
                backend, "h3_resolution", getattr(backend, "resolution", 8)
            )
            cell_area = h3_area_km2.get(resolution, 0.7373)
            num_hexagons = len(backend.target_hexagons)

            analysis_results["buffer_analysis"] = {
                "h3_resolution": resolution,
                "cell_area_km2": cell_area,
                "total_hexagons": num_hexagons,
                "total_coverage_km2": round(num_hexagons * cell_area, 2),
            }

            analysis_results["proximity_analysis"] = {
                "method": "h3_grid_distance",
                "total_hexagons_analyzed": num_hexagons,
                "h3_resolution": resolution,
            }
        except Exception as e:
            logger.warning(f"Buffer/proximity analysis failed: {e}")

        logger.info("✅ Enhanced spatial analysis completed")
        return analysis_results

    except Exception as e:
        logger.error(f"❌ Enhanced spatial analysis failed: {e}")
        return {}


def run_comprehensive_analysis(backend, modules: dict, args) -> tuple[dict, dict]:
    """
    Run comprehensive analysis with real data tracking and enhanced reporting.

    Args:
        backend: Unified backend with processed data
        modules: Dictionary of initialized modules
        args: Command line arguments

    Returns:
        Tuple of (redevelopment_scores, summary)
    """
    logger = logging.getLogger(__name__)
    logger.info("📊 Starting comprehensive analysis with real data tracking...")

    # Pre-analysis data check
    logger.info("🔍 Pre-analysis data check for all modules...")
    for module_name, module in modules.items():
        try:
            data_files = list(module.data_dir.glob("*.geojson"))
            logger.info(f"  📁 {module_name}: Found {len(data_files)} data files")
            for file in data_files:
                logger.info(f"    📄 {file.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ {module_name}: Could not check data files: {e}")

    # Run backend analysis
    try:
        # First run the comprehensive analysis to populate unified data
        logger.info(
            "🔧 Running comprehensive backend analysis to populate unified data..."
        )
        backend.run_comprehensive_analysis()

        # Now calculate redevelopment scores
        redevelopment_scores = backend.calculate_agricultural_redevelopment_potential()
        summary = backend.get_comprehensive_summary()

        # Post-analysis data acquisition summary
        logger.info(" Post-analysis data acquisition summary:")
        data_acquisition_summary = {}

        for module_name, module in modules.items():
            try:
                # Convert generator to list to get proper length
                data_files = list(module.data_dir.glob("*.geojson"))
                processed_count = len(data_files)
                logger.info(
                    f"  ✅ {module_name}: {processed_count} processed data files"
                )
                data_acquisition_summary[module_name] = processed_count
            except Exception as e:
                logger.error(f"  ❌ {module_name}: Error in post-analysis check: {e}")
                data_acquisition_summary[module_name] = 0

        logger.info("✅ Comprehensive analysis completed successfully")
        return redevelopment_scores, summary

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback

        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
