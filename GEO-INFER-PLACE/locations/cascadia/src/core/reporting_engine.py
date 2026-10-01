#!/usr/bin/env python3
"""
Reporting Engine for Cascadia Agricultural Analysis

This module handles all reporting operations including spatial analysis reports,
dashboard generation, and comprehensive analysis reports.
"""

import logging
import json
from typing import Any
from pathlib import Path
from datetime import datetime
import numpy as np

# Import the necessary components
try:
    from geo_infer_space.core.spatial_processor import SpatialProcessor
    from geo_infer_space.core.visualization_receipt import write_visualization_receipt

    SPACE_CORE_AVAILABLE = True
except ImportError:
    SPACE_CORE_AVAILABLE = False

    class SpatialProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def calculate_spatial_correlation(self, scores1, scores2):
            try:
                common_hexagons = set(scores1.keys()) & set(scores2.keys())
                if len(common_hexagons) < 2:
                    return 0.0
                values1 = [scores1[h] for h in common_hexagons]
                values2 = [scores2[h] for h in common_hexagons]
                correlation = np.corrcoef(values1, values2)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            except Exception:
                return 0.0

    def write_visualization_receipt(*args, **kwargs):  # type: ignore[no-redef]
        """Fallback no-op when SPACE core is unavailable."""
        return None


def generate_spatial_analysis_report(backend, output_dir: Path) -> str:
    """Generate comprehensive spatial analysis report using SPACE capabilities"""
    logger = logging.getLogger(__name__)
    logger.info("Generating spatial analysis report with SPACE integration...")

    try:
        # Perform spatial analysis
        spatial_analysis = {
            "h3_coverage": {
                "total_hexagons": len(backend.target_hexagons),
                "resolution": backend.resolution,
                "coverage_area_km2": len(backend.target_hexagons)
                * 0.46,  # Approximate area per H3 cell at res 8
            },
            "module_spatial_distribution": {},
            "spatial_correlations": {},
            "hotspot_analysis": {},
        }

        # Analyze spatial distribution of each module
        for module_name in backend.modules.keys():
            module_hexagons = [
                h
                for h in backend.target_hexagons
                if backend.unified_data.get(h, {}).get(module_name)
            ]
            spatial_analysis["module_spatial_distribution"][module_name] = {
                "hexagon_count": len(module_hexagons),
                "coverage_percentage": len(module_hexagons) / len(backend.target_hexagons) * 100,
            }

        # Generate report
        report_path = (
            output_dir / f"cascadia_spatial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(report_path, "w") as f:
            f.write("# Cascadia Spatial Analysis Report\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## H3 Coverage Analysis\n\n")
            f.write(
                f"- **Total Hexagons:** {spatial_analysis['h3_coverage']['total_hexagons']:,}\n"
            )
            f.write(f"- **H3 Resolution:** {spatial_analysis['h3_coverage']['resolution']}\n")
            f.write(
                f"- **Coverage Area:** {spatial_analysis['h3_coverage']['coverage_area_km2']:.1f} km²\n\n"
            )

            f.write("## Module Spatial Distribution\n\n")
            f.write("| Module | Hexagons | Coverage (%) |\n")
            f.write("|--------|----------|--------------|\n")
            for module, data in spatial_analysis["module_spatial_distribution"].items():
                f.write(
                    f"| {module} | {data['hexagon_count']:,} | {data['coverage_percentage']:.1f} |\n"
                )

            f.write("\n## Spatial Analysis Summary\n\n")
            f.write(
                "This report was generated using GEO-INFER-SPACE spatial processing capabilities.\n"
            )
            f.write(
                "The analysis leverages H3 hexagonal spatial indexing for consistent geospatial operations.\n"
            )

        logger.info(f"✅ Spatial analysis report generated: {report_path}")

        # Write a deterministic visualization receipt alongside the report so the
        # markdown artifact is auditable (input hash, H3 version, accessibility).
        try:
            write_visualization_receipt(
                artifact_path=report_path,
                input_payload=spatial_analysis,
                schema_version="geo-infer-cascadia-spatial-report/v1",
                title_marker="Cascadia Spatial Analysis Report",
            )
        except Exception as receipt_err:
            logger.warning(f"Could not write spatial report receipt: {receipt_err}")

        return str(report_path)

    except Exception as e:
        logger.error(f"❌ Failed to generate spatial analysis report: {e}")
        return ""


def generate_enhanced_dashboard(backend, output_dir: Path, visualization_engine) -> str:
    """Generate enhanced interactive dashboard using SPACE visualization engine"""
    logger = logging.getLogger(__name__)
    logger.info("🎨 Generating enhanced interactive dashboard with SPACE visualization...")

    try:
        # Prepare analysis results for dashboard
        analysis_results = {
            "domain_results": {
                "agricultural_analysis": {
                    "zoning_data": backend.unified_data,
                    "redevelopment_scores": backend.calculate_agricultural_redevelopment_potential(),
                    "module_coverage": {
                        name: len(backend.modules[name].target_hexagons)
                        for name in backend.modules.keys()
                    },
                }
            },
            "integrated_results": {
                "h3_hexagons": backend.target_hexagons,
                "spatial_analysis": backend.spatial_analysis_results,
                "hotspot_analysis": backend.hotspot_analysis,
            },
        }

        # Generate comprehensive dashboard
        dashboard_path = visualization_engine.create_comprehensive_dashboard(analysis_results)

        logger.info(f"✅ Enhanced dashboard generated: {dashboard_path}")
        return dashboard_path

    except Exception as e:
        logger.error(f"❌ Failed to generate enhanced dashboard: {e}")
        return ""


def generate_analysis_report(summary: dict[str, Any], output_path: Path) -> None:
    """
    Generate a comprehensive analysis report in Markdown format with SPACE integration.

    Args:
        summary: Comprehensive summary data from the backend.
        output_path: Path to save the Markdown file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating enhanced analysis report to {output_path}...")

    bioregion = summary.get("bioregion", "Unknown Bioregion")
    bioregion_desc = {
        "Cascadia": "encompassing northern California and all of Oregon.",
        "Columbia": "encompassing the Columbia River Basin region.",
    }.get(bioregion, f"in the {bioregion} bioregion.")

    def fmt(value):
        return f"{value:,}" if isinstance(value, (int, float)) else str(value)

    rp = summary.get("redevelopment_potential", {})

    with open(output_path, "w") as f:
        f.write(f"# {bioregion} Agricultural Land Analysis Report\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        f.write("## 1. Executive Summary\n")
        f.write(
            f"This report presents a comprehensive agricultural land analysis for the **{bioregion}** bioregion, {bioregion_desc}\n"
        )
        f.write(
            f"The analysis utilized H3 spatial indexing at resolution **{summary.get('h3_resolution', 'Unknown')}** to integrate and assess data from **{len(summary.get('modules_analyzed', []))} specialized modules**. The key output is a redevelopment potential score for each hexagonal area, identifying promising locations for agricultural transition.\n\n"
        )

        f.write("## 2. Analysis Overview\n")
        f.write(f"- **Total Hexagons Analyzed:** {fmt(summary.get('total_hexagons', 0))}\n")
        f.write(f"- **H3 Resolution:** {summary.get('h3_resolution', 'Unknown')}\n")
        f.write(f"- **Modules Executed:** `{', '.join(summary.get('modules_analyzed', []))}`\n\n")

        f.write("## 3. Redevelopment Potential Insights\n")
        f.write(f"- **Mean Redevelopment Score:** {rp.get('mean_score', 0):.3f}\n")
        f.write(f"- **Median Redevelopment Score:** {rp.get('median_score', 0):.3f}\n")
        f.write(
            f"- **High Potential Areas (>0.75):** {fmt(rp.get('high_potential_hexagons', 0))} hexagons\n"
        )
        f.write(
            f"- **Low Potential Areas (<0.25):** {fmt(rp.get('low_potential_hexagons', 0))} hexagons\n\n"
        )

        f.write("## 4. Module Coverage\n")
        f.write(
            "This section details the data coverage for each analysis module across the target hexagons.\n\n"
        )
        f.write("| Module                    | Processed Hexagons | Coverage (%) |\n")
        f.write("|---------------------------|--------------------|--------------|\n")

        module_summaries = summary.get("module_summaries", {})

        for module_name in summary.get("modules_analyzed", []):
            module_data = module_summaries.get(module_name, {})
            processed_count = module_data.get("processed_hexagons", 0)
            coverage_pct = module_data.get("coverage", 0.0)
            f.write(
                f"| {module_name.title():<25} | {processed_count:>16} | {coverage_pct:>11.2f} |\n"
            )

        f.write("\n## 5. Technical Framework & Methodology\n")
        f.write(
            "The analysis is built on a **Unified H3 Backend**, which standardizes diverse geospatial datasets into a common hexagonal grid. This enables:\n\n"
        )
        f.write("- **Cross-border Analysis**: Seamless integration of California and Oregon data\n")
        f.write(
            "- **Multi-source Integration**: Harmonization of zoning, water rights, ownership, and infrastructure data\n"
        )
        f.write(
            "- **Spatial Consistency**: Uniform resolution and coordinate system across all analyses\n"
        )
        f.write("- **Scalable Processing**: Efficient handling of large geospatial datasets\n")
        f.write(
            "- **SPACE Integration**: Advanced spatial analysis using GEO-INFER-SPACE capabilities\n\n"
        )

        f.write("## 6. Data Sources & Quality\n")
        f.write("The analysis integrates data from multiple authoritative sources:\n\n")
        f.write("- **Zoning Data**: FMMP (California), ORMAP (Oregon)\n")
        f.write("- **Water Rights**: eWRIMS (California), Oregon WRD\n")
        f.write("- **Current Use**: NASS CDL, Land IQ\n")
        f.write("- **Infrastructure**: Building footprints, power transmission lines\n")
        f.write("- **Ownership**: County parcel records, USDA ERS\n\n")

        f.write("## 7. Redevelopment Scoring Methodology\n")
        f.write("The redevelopment potential score combines multiple factors:\n\n")
        f.write("- **Zoning Compatibility** (25%): Agricultural zoning classifications\n")
        f.write("- **Water Availability** (20%): Surface and groundwater access\n")
        f.write("- **Infrastructure** (15%): Power, roads, and improvements\n")
        f.write("- **Ownership Patterns** (15%): Parcel size and ownership concentration\n")
        f.write("- **Current Use** (15%): Existing agricultural activities\n")
        f.write("- **Financial Factors** (10%): Mortgage debt and economic indicators\n\n")

        f.write("## 8. SPACE Integration Features\n")
        f.write("This analysis leverages advanced GEO-INFER-SPACE capabilities:\n\n")
        f.write("- **H3 Spatial Indexing**: Efficient hexagonal grid processing\n")
        f.write(
            "- **OSC Integration**: OS-Climate tool integration for standardized geospatial operations\n"
        )
        f.write(
            "- **Spatial Analysis**: Correlation analysis, hotspot detection, and proximity analysis\n"
        )
        f.write("- **Enhanced Visualization**: Interactive dashboards with multi-layer overlays\n")
        f.write("- **Real-time Data Integration**: Dynamic data loading and processing\n\n")

        f.write("## 9. Limitations & Considerations\n")
        f.write(
            "- **Data Availability**: Some modules may have limited data coverage in certain areas\n"
        )
        f.write(
            "- **Temporal Aspects**: Data represents a snapshot in time; conditions may change\n"
        )
        f.write("- **Resolution Trade-offs**: H3 resolution 8 provides ~0.46 km² hexagons\n")
        f.write("- **Cross-border Harmonization**: Different data standards between states\n\n")

        f.write("## 10. Next Steps & Recommendations\n")
        f.write("Based on the analysis results, recommended next steps include:\n\n")
        f.write(
            "1. **Field Validation**: Ground-truth high-potential areas identified by the analysis\n"
        )
        f.write(
            "2. **Stakeholder Engagement**: Consult with local agricultural communities and landowners\n"
        )
        f.write(
            "3. **Policy Development**: Develop targeted policies for agricultural redevelopment\n"
        )
        f.write(
            "4. **Infrastructure Planning**: Coordinate with utility and transportation agencies\n"
        )
        f.write(
            "5. **Water Rights Assessment**: Detailed analysis of water availability and rights\n\n"
        )

        f.write("---\n")
        f.write(
            "*This report was generated using the GEO-INFER framework with enhanced SPACE integration for advanced geospatial analysis.*\n"
        )

    logger.info(f"✅ Analysis report generated: {output_path}")

    # Write a deterministic visualization receipt alongside the report so the
    # markdown artifact is auditable (input hash, H3 version, accessibility).
    try:
        write_visualization_receipt(
            artifact_path=output_path,
            input_payload=summary,
            schema_version="geo-infer-cascadia-analysis-report/v1",
            title_marker=f"{bioregion} Agricultural Land Analysis Report",
        )
    except Exception as receipt_err:
        logger.warning(f"Could not write analysis report receipt: {receipt_err}")


def export_data_provenance(provenance: dict[str, Any], output_dir: Path) -> Path:
    """Write a machine-readable data provenance manifest.

    The manifest records, per module, whether empirical data was used,
    source URLs when available, input feature counts, output hexagon counts,
    cache file locations, and fusion cache usage.

    Args:
        provenance: Provenance dictionary assembled by the main runner
        output_dir: Directory to save the manifest

    Returns:
        Path to the saved JSON manifest
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        output_dir / f"cascadia_data_provenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    try:
        with open(manifest_path, "w") as f:
            json.dump(provenance, f, indent=2)
        logger.info(f"✅ Data provenance manifest written: {manifest_path}")
    except Exception as e:
        logger.error(f"❌ Failed to write data provenance manifest: {e}")
    return manifest_path
