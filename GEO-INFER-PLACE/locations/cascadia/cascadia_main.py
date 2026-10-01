#!/usr/bin/env python3
"""
Cascadia Agricultural Analysis Framework

Thin orchestration of the Cascadia pipeline across real-data acquisition,
H3 processing, fusion, scoring, and visualization. Designed for:
- Reproducible, cache-first runs
- Config/CLI driven toggles for modules and layers
- Parallel module processing (I/O-heavy acquisition + native geospatial ops)
- Optional profiling to guide further optimization

References for pipeline optimization patterns:
- Vectorization, generators, and multiprocessing guidance [1][2][3]

[1] https://python.plainenglish.io/optimizing-data-processing-in-python-best-practices-for-data-scientists-0fd6bac3fa5a
[2] https://medium.com/@huzaifazahoor654/how-to-optimize-python-code-for-faster-data-processing-1b7eeea0f379
[3] https://medium.com/@yaswanth.thod/stop-writing-slow-python-14-mind-blowing-speed-hacks-you-need-right-now-aa495862ec00
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from geo_infer_space.utils.h3_utils import cell_to_latlng, cell_to_latlng_boundary

# The script directory (on sys.path when run as a script, and on the pytest
# pythonpath) makes the local ``src`` package importable.
from src.core.setup_manager import setup_logging, load_analysis_config
from src.core.data_processor import create_shared_backend, export_results
from src.core.reporting_engine import generate_analysis_report, export_data_provenance

# Import enhanced modules for H3 geospatial data fusion
from src.core.enhanced_data_manager import create_enhanced_data_manager
from src.core.enhanced_h3_fusion import create_enhanced_h3_fusion
from src.core.enhanced_logging import EnhancedLoggingConfig
from src.core.real_data_acquisition import create_real_data_acquisition
from src.core.visualization.comprehensive_visualization import (
    create_comprehensive_visualization_engine,
)


def parse_counties(counties_str: str) -> dict:
    """Parse counties string into a mapping of state -> list[county].

    Args:
        counties_str: Comma-separated string like "CA:Del Norte,OR:Josephine" or "all".

    Returns:
        Dict mapping state abbreviations to list of county names.
    """
    counties_dict = {}
    if counties_str and counties_str != "all":
        for county_pair in counties_str.split(","):
            if ":" in county_pair:
                state, county = county_pair.strip().split(":", 1)
                if state not in counties_dict:
                    counties_dict[state] = []
                counties_dict[state].append(county)
            else:
                # Default to CA if no state specified
                if "CA" not in counties_dict:
                    counties_dict["CA"] = []
                counties_dict["CA"].append(county_pair.strip())
    else:
        counties_dict = {"CA": ["all"], "OR": ["all"]}
    return counties_dict


def initialize_analysis(args):
    """Initialize logging, config, backend, data manager, fusion, acquisition, and viz.

    Returns:
        (backend, modules, data_manager, h3_fusion, real_data_acquisition, viz_engine)
    """
    logger = logging.getLogger(__name__)

    # Set up enhanced logging
    log_file = Path(args.output_dir) / "cascadia_analysis.log"
    EnhancedLoggingConfig.setup_logging(
        log_level="INFO" if not args.debug else "DEBUG",
        log_file=log_file,
        console_output=True,
        include_timestamps=True,
        include_module_names=True,
    )

    logger.info("🚀 Initializing Cascadia Agricultural Analysis Framework")
    logger.info(f"Enhanced logging initialized: {log_file}")

    # Load configuration
    config = load_analysis_config()

    # Validate configuration
    logger.info("🔍 Validating configuration...")
    from src.core.enhanced_config import create_enhanced_config_manager

    config_manager = create_enhanced_config_manager()
    validation_result = config_manager.validate_configuration()

    if not validation_result["valid"]:
        logger.error("❌ Configuration validation failed:")
        for error in validation_result["errors"]:
            logger.error(f"   • {error}")
        logger.info("💡 Recommendations:")
        for recommendation in validation_result["recommendations"]:
            logger.info(f"   • {recommendation}")
        sys.exit(1)

    if validation_result["warnings"]:
        logger.warning("⚠️ Configuration warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"   • {warning}")

    # Bioregion mode: override resolution and include ecology module
    if getattr(args, "bioregion", False):
        if args.h3_resolution == 8:  # only override default, not explicit user choice
            args.h3_resolution = 7
        if args.counties == "CA:Del Norte":  # only override default
            args.counties = "all"

    # Parse counties and modules with validation
    counties_dict = parse_counties(args.counties)
    # Resolve active modules from CLI or config. '--modules all' or missing -> config list
    cfg_active = (config.get("analysis_settings") or {}).get("active_modules") or []
    if (not args.modules) or (args.modules.strip().lower() == "all"):
        active_modules = list(cfg_active)
    else:
        active_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    # Bioregion mode always includes ecology
    if getattr(args, "bioregion", False) and "ecology" not in active_modules:
        active_modules.append("ecology")

    # Validate parsed modules against available modules
    valid_modules = [
        "zoning",
        "current_use",
        "ownership",
        "improvements",
        "water_rights",
        "ground_water",
        "surface_water",
        "power_source",
        "mortgage_debt",
        "ecology",
    ]
    invalid_modules = [m for m in active_modules if m not in valid_modules]
    if invalid_modules:
        logger.error(f"❌ Invalid modules specified: {invalid_modules}")
        logger.info(f"💡 Available modules: {valid_modules}")
        sys.exit(1)

    logger.info(f"Target counties: {counties_dict}")
    logger.info(f"Active modules: {active_modules}")
    logger.info(
        f"📊 Configuration validation: {'✅ PASSED' if validation_result['valid'] else '❌ FAILED'}"
    )

    # Create shared backend
    shared_backend = create_shared_backend(args.h3_resolution, counties_dict, Path(args.output_dir))

    # Initialize enhanced data manager
    data_manager = create_enhanced_data_manager(
        base_data_dir=Path(args.output_dir) / "data", h3_resolution=args.h3_resolution
    )

    # Initialize enhanced H3 fusion engine with cache directory inside output
    fusion_cache_dir = Path(args.output_dir) / "data" / "cache" / "fusion"
    # Choose fusion mode from config/env/arg (geom_intersect | key_join)
    fusion_mode = getattr(args, "fusion_mode", None) or os.environ.get(
        "CASCADIA_FUSION_MODE", "geom_intersect"
    )
    h3_fusion = create_enhanced_h3_fusion(
        h3_resolution=args.h3_resolution,
        enable_spatial_analysis=args.spatial_analysis,
        cache_dir=fusion_cache_dir,
        fusion_mode=fusion_mode,
    )

    # Initialize real data acquisition system
    real_data_acquisition = create_real_data_acquisition(
        output_dir=Path(args.output_dir) / "real_data"
    )

    # Initialize comprehensive visualization engine
    viz_engine = create_comprehensive_visualization_engine(
        output_dir=Path(args.output_dir) / "visualizations"
    )

    # Validate H3 operations
    h3_validation = h3_fusion.validate_h3_operations()
    logger.info(f"H3 validation result: {h3_validation}")

    # Initialize modules with enhanced data management
    modules = initialize_modules_with_enhanced_data_management(
        active_modules, shared_backend, data_manager, h3_fusion
    )

    if not modules:
        logger.error("❌ No modules could be initialized. Exiting.")
        sys.exit(1)

    return (
        shared_backend,
        modules,
        data_manager,
        h3_fusion,
        real_data_acquisition,
        viz_engine,
    )


def initialize_modules_with_enhanced_data_management(
    active_modules, shared_backend, data_manager, h3_fusion
):
    """Initialize modules with enhanced data management and H3 fusion"""
    logger = logging.getLogger(__name__)
    modules = {}

    # Import all the specialized modules from the 'cascadia' location
    try:
        from src.data_modules.zoning.geo_infer_zoning import GeoInferZoning

        ZONING_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Zoning module not available: {e}")
        ZONING_AVAILABLE = False
        GeoInferZoning = None

    try:
        from src.data_modules.current_use.geo_infer_current_use import (
            GeoInferCurrentUse,
        )

        CURRENT_USE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Current use module not available: {e}")
        CURRENT_USE_AVAILABLE = False
        GeoInferCurrentUse = None

    try:
        from src.data_modules.ownership.geo_infer_ownership import GeoInferOwnership

        OWNERSHIP_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Ownership module not available: {e}")
        OWNERSHIP_AVAILABLE = False
        GeoInferOwnership = None

    try:
        from src.data_modules.improvements.geo_infer_improvements import (
            GeoInferImprovements,
        )

        IMPROVEMENTS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Improvements module not available: {e}")
        IMPROVEMENTS_AVAILABLE = False
        GeoInferImprovements = None

    # Additional modules
    try:
        from src.data_modules.water_rights.geo_infer_water_rights import (
            GeoInferWaterRights,
        )

        WATER_RIGHTS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Water rights module not available: {e}")
        WATER_RIGHTS_AVAILABLE = False
        GeoInferWaterRights = None

    try:
        from src.data_modules.ground_water.geo_infer_ground_water import (
            GeoInferGroundWater,
        )

        GROUND_WATER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Ground water module not available: {e}")
        GROUND_WATER_AVAILABLE = False
        GeoInferGroundWater = None

    try:
        from geo_infer_place.hydrography import (
            GeoInferSurfaceWater,
        )

        SURFACE_WATER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Surface water module not available: {e}")
        SURFACE_WATER_AVAILABLE = False
        GeoInferSurfaceWater = None

    try:
        from src.data_modules.power_source.geo_infer_power_source import (
            GeoInferPowerSource,
        )

        POWER_SOURCE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Power source module not available: {e}")
        POWER_SOURCE_AVAILABLE = False
        GeoInferPowerSource = None

    try:
        from src.data_modules.mortgage_debt.geo_infer_mortgage_debt import (
            GeoInferMortgageDebt,
        )

        MORTGAGE_DEBT_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Mortgage debt module not available: {e}")
        MORTGAGE_DEBT_AVAILABLE = False
        GeoInferMortgageDebt = None

    try:
        from src.data_modules.ecology.geo_infer_ecology import GeoInferEcology

        ECOLOGY_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Ecology module not available: {e}")
        ECOLOGY_AVAILABLE = False
        GeoInferEcology = None

    # Initialize available modules using the shared backend with enhanced data management
    if "zoning" in active_modules and ZONING_AVAILABLE:
        try:
            modules["zoning"] = GeoInferZoning(shared_backend)
            # Integrate enhanced data management
            modules["zoning"].data_manager = data_manager
            modules["zoning"].h3_fusion = h3_fusion
            logger.info("✅ Zoning module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize zoning module: {e}")

    if "current_use" in active_modules and CURRENT_USE_AVAILABLE:
        try:
            modules["current_use"] = GeoInferCurrentUse(shared_backend)
            # Integrate enhanced data management
            modules["current_use"].data_manager = data_manager
            modules["current_use"].h3_fusion = h3_fusion
            logger.info("✅ Current use module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize current use module: {e}")

    if "ownership" in active_modules and OWNERSHIP_AVAILABLE:
        try:
            modules["ownership"] = GeoInferOwnership(shared_backend)
            # Integrate enhanced data management
            modules["ownership"].data_manager = data_manager
            modules["ownership"].h3_fusion = h3_fusion
            logger.info("✅ Ownership module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ownership module: {e}")

    if "improvements" in active_modules and IMPROVEMENTS_AVAILABLE:
        try:
            modules["improvements"] = GeoInferImprovements(shared_backend)
            # Integrate enhanced data management
            modules["improvements"].data_manager = data_manager
            modules["improvements"].h3_fusion = h3_fusion
            logger.info("✅ Improvements module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize improvements module: {e}")

    if "water_rights" in active_modules and WATER_RIGHTS_AVAILABLE:
        try:
            modules["water_rights"] = GeoInferWaterRights(shared_backend)
            modules["water_rights"].data_manager = data_manager
            modules["water_rights"].h3_fusion = h3_fusion
            logger.info("✅ Water rights module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize water rights module: {e}")

    if "ground_water" in active_modules and GROUND_WATER_AVAILABLE:
        try:
            modules["ground_water"] = GeoInferGroundWater(shared_backend)
            modules["ground_water"].data_manager = data_manager
            modules["ground_water"].h3_fusion = h3_fusion
            logger.info("✅ Ground water module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ground water module: {e}")

    if "surface_water" in active_modules and SURFACE_WATER_AVAILABLE:
        try:
            modules["surface_water"] = GeoInferSurfaceWater(shared_backend)
            modules["surface_water"].data_manager = data_manager
            modules["surface_water"].h3_fusion = h3_fusion
            logger.info("✅ Surface water module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize surface water module: {e}")

    if "power_source" in active_modules and POWER_SOURCE_AVAILABLE:
        try:
            modules["power_source"] = GeoInferPowerSource(shared_backend)
            modules["power_source"].data_manager = data_manager
            modules["power_source"].h3_fusion = h3_fusion
            logger.info("✅ Power source module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize power source module: {e}")

    if "mortgage_debt" in active_modules and MORTGAGE_DEBT_AVAILABLE:
        try:
            modules["mortgage_debt"] = GeoInferMortgageDebt(shared_backend)
            modules["mortgage_debt"].data_manager = data_manager
            modules["mortgage_debt"].h3_fusion = h3_fusion
            logger.info("✅ Mortgage debt module initialized with enhanced data management")
        except Exception as e:
            logger.error(f"❌ Failed to initialize mortgage debt module: {e}")

    if "ecology" in active_modules and ECOLOGY_AVAILABLE and GeoInferEcology:
        try:
            modules["ecology"] = GeoInferEcology()
            logger.info("✅ Ecology module initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ecology module: {e}")

    if not modules:
        logger.error("❌ No modules could be initialized.")
        return {}

    # Update the shared backend with initialized modules
    shared_backend.modules = modules
    logger.info(f"✅ Updated shared backend with {len(modules)} active modules")

    return modules


def generate_reports(summary, output_dir, spatial_analysis=False):
    """Generate analysis reports"""
    logger = logging.getLogger(__name__)

    # Generate Markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"cascadia_analysis_report_{timestamp}.md"
    generate_analysis_report(summary, report_path)
    logger.info(f"📋 Analysis report: {report_path}")


"""
Cascadian Agricultural Land Analysis Framework - Enhanced Main Integration Script

This script orchestrates the complete agricultural land analysis across the
Cascadian bioregion (Northern California + Oregon) using the unified H3-indexed
backend and all 8 specialized data modules, with maximum integration of
GEO-INFER-SPACE methods and capabilities.

Enhanced Features:
- Full SPACE module integration (H3 utilities, OSC tools, visualization engines)
- Advanced multi-layer visualization with California demo patterns
- Comprehensive data processing with SPACE utilities
- Enhanced reporting and diagnostics
- Real-time data integration capabilities
- Spatial analysis and correlation features
- Advanced dashboard generation
- Enhanced H3 v4 geospatial data fusion
- Reproducible data module structure with intelligent caching

Usage:
    python cascadia_main.py [options]

Options:
    --resolution: H3 resolution level (default: 8)
    --output-dir: Output directory for results (default: ./output)
    --export-format: Export format (geojson, csv, json) (default: geojson)
    --counties: Comma-separated list of counties to analyze (default: all)
    --modules: Comma-separated list of modules to run (default: all)
    --verbose: Enable verbose logging
    --check-deps: Check and install dependencies
    --enhanced-viz: Enable enhanced visualization features
    --real-time: Enable real-time data integration
    --spatial-analysis: Enable advanced spatial analysis features
    --generate-dashboard: Generate interactive dashboard
    --force-refresh: Force refresh of cached data
    --validate-h3: Validate H3 operations and API usage
"""


# Import from the new core location


# Import all the specialized modules from the 'cascadia' location
try:
    from src.data_modules.zoning.geo_infer_zoning import GeoInferZoning

    ZONING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Zoning module not available: {e}")
    ZONING_AVAILABLE = False
    GeoInferZoning = None

try:
    from src.data_modules.current_use.geo_infer_current_use import GeoInferCurrentUse

    CURRENT_USE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Current use module not available: {e}")
    CURRENT_USE_AVAILABLE = False
    GeoInferCurrentUse = None

try:
    from src.data_modules.ownership.geo_infer_ownership import GeoInferOwnership

    OWNERSHIP_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Ownership module not available: {e}")
    OWNERSHIP_AVAILABLE = False
    GeoInferOwnership = None

try:
    from src.data_modules.improvements.geo_infer_improvements import (
        GeoInferImprovements,
    )

    IMPROVEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Improvements module not available: {e}")
    IMPROVEMENTS_AVAILABLE = False
    GeoInferImprovements = None


# Remove the large method definitions since they're now in utils


def parse_arguments():
    """Parse command line arguments for Cascadia analysis"""
    parser = argparse.ArgumentParser(
        description="Cascadia Agricultural Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python3 cascadia_main.py

  # Analysis with custom H3 resolution and specific counties
  python3 cascadia_main.py --h3-resolution 8 --counties "CA:Del Norte,CA:Humboldt"

  # Analysis with spatial analysis and lightweight visualization
  python3 cascadia_main.py --spatial-analysis --lightweight-viz

  # Analysis with Datashader visualization (recommended for large datasets)
  python3 cascadia_main.py --datashader-viz

  # Analysis with Deepscatter visualization (web-based, lightweight)
  python3 cascadia_main.py --deepscatter-viz

  # Export in different formats
  python3 cascadia_main.py --export-format csv --verbose

  # Force refresh of cached data
  python3 cascadia_main.py --force-refresh --verbose

  # Validate H3 operations
  python3 cascadia_main.py --validate-h3

  # Validate configuration only
  python3 cascadia_main.py --validate-config

  # Run performance benchmarks only
  python3 cascadia_main.py --benchmark
        """,
    )

    # Core analysis parameters
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=8,
        help="H3 resolution for analysis (default: 8)",
    )

    parser.add_argument(
        "--counties",
        type=str,
        default="CA:Del Norte",
        help='Target counties in format "STATE:County,STATE:County" (default: CA:Del Norte)',
    )

    parser.add_argument(
        "--modules",
        type=str,
        default="zoning,current_use,ownership,improvements,water_rights,ground_water,surface_water,power_source,mortgage_debt",
        help="Comma-separated list of modules to run (default: all modules)",
    )

    # Output and export options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )

    parser.add_argument(
        "--export-format",
        type=str,
        choices=["geojson", "csv", "json"],
        default="geojson",
        help="Export format for unified data (default: geojson)",
    )

    # Visualization options
    parser.add_argument(
        "--generate-dashboard",
        action="store_true",
        help="Generate interactive dashboard (may be large and slow)",
    )

    parser.add_argument(
        "--lightweight-viz",
        action="store_true",
        help="Generate lightweight static visualizations (recommended)",
    )

    parser.add_argument(
        "--datashader-viz",
        action="store_true",
        help="Generate Datashader visualizations (best for large datasets)",
    )

    parser.add_argument(
        "--deepscatter-viz",
        action="store_true",
        help="Generate Deepscatter visualizations (web-based, lightweight)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Number of parallel workers for module acquisition/processing (0=auto)",
    )
    parser.add_argument(
        "--profile-run",
        action="store_true",
        help="Profile the run with cProfile and write results under output/profile/",
    )
    parser.add_argument(
        "--skip-deepscatter",
        action="store_true",
        help="Skip Deepscatter even if enabled elsewhere",
    )
    parser.add_argument(
        "--visible-layers",
        type=str,
        default=None,
        help="Comma-separated list of layers (modules) initially visible on interactive map (e.g., zoning,ownership)",
    )
    parser.add_argument(
        "--include-layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to include on interactive map; defaults to all executed modules",
    )

    # Analysis options
    parser.add_argument(
        "--spatial-analysis",
        action="store_true",
        help="Perform enhanced spatial analysis",
    )

    parser.add_argument(
        "--skip-cache", action="store_true", help="Skip cache and regenerate all data"
    )

    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached data")

    parser.add_argument(
        "--validate-h3",
        action="store_true",
        help="Validate H3 operations and API usage",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        choices=["geom_intersect", "key_join"],
        default=None,
        help="Fusion mode: geometric intersection or direct key join",
    )

    # Logging and debugging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed error reporting",
    )

    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit without running analysis",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks and exit without running analysis",
    )

    parser.add_argument(
        "--bioregion",
        action="store_true",
        help="Run in bioregion mode: H3 resolution 7, full Cascadia extent, ecology module enabled",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start HTTP server after pipeline completes and open browser",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8765,
        help="Port for --serve mode (default: 8765)",
    )

    return parser.parse_args()


def main():
    """Main execution function for Cascadia analysis"""
    args = parse_arguments()

    # Handle configuration validation
    if args.validate_config:
        logger = logging.getLogger(__name__)
        logger.info("🔍 Validating configuration only...")

        # Load and validate configuration
        config = load_analysis_config()
        from src.core.enhanced_config import create_enhanced_config_manager

        config_manager = create_enhanced_config_manager()
        validation_result = config_manager.validate_configuration()

        # Print detailed validation results
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)

        print(f"\n📊 Overall Status: {'✅ VALID' if validation_result['valid'] else '❌ INVALID'}")

        if validation_result["errors"]:
            print(f"\n❌ Errors ({len(validation_result['errors'])}):")
            for error in validation_result["errors"]:
                print(f"   • {error}")

        if validation_result["warnings"]:
            print(f"\n⚠️ Warnings ({len(validation_result['warnings'])}):")
            for warning in validation_result["warnings"]:
                print(f"   • {warning}")

        print(f"\n💡 Recommendations ({len(validation_result['recommendations'])}):")
        for recommendation in validation_result["recommendations"]:
            print(f"   • {recommendation}")

        print("\n📋 Validation Details:")
        for key, details in validation_result["validation_details"].items():
            if isinstance(details, dict):
                print(f"   {key}:")
                for sub_key, value in details.items():
                    if isinstance(value, list):
                        print(f"     {sub_key}: {', '.join(value) if value else 'None'}")
                    else:
                        print(f"     {sub_key}: {value}")
            else:
                print(f"   {key}: {details}")

        print("\n" + "=" * 60)
        sys.exit(0 if validation_result["valid"] else 1)

    # Handle performance benchmarking
    if args.benchmark:
        logger = logging.getLogger(__name__)
        logger.info("📊 Running performance benchmarks...")

        # Load configuration
        config = load_analysis_config()
        from src.core.enhanced_config import create_enhanced_config_manager

        config_manager = create_enhanced_config_manager()

        # Get active modules
        cfg_active = (config.get("analysis_settings") or {}).get("active_modules") or []
        if (not args.modules) or (args.modules.strip().lower() == "all"):
            active_modules = list(cfg_active)
        else:
            active_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

        # Create enhanced data manager for benchmarking
        from src.core.enhanced_data_manager import EnhancedDataManager

        benchmark_data_manager = EnhancedDataManager(
            Path(args.output_dir) / "data", args.h3_resolution
        )

        # Run benchmarks for each module
        all_benchmarks = {}
        for module_name in active_modules:
            logger.info(f"Benchmarking {module_name} module...")
            try:
                benchmark_result = benchmark_data_manager.benchmark_performance(module_name)
                all_benchmarks[module_name] = benchmark_result
            except Exception as e:
                logger.error(f"Benchmark failed for {module_name}: {e}")
                all_benchmarks[module_name] = {"error": str(e)}

        # Print comprehensive benchmark report
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        print("\n🖥️ System Information:")
        system_info = all_benchmarks.get(list(all_benchmarks.keys())[0], {}).get("system_info", {})
        for key, value in system_info.items():
            print(f"   {key}: {value}")

        print("\n📊 Module Benchmarks:")
        for module_name, benchmark in all_benchmarks.items():
            if "error" in benchmark:
                print(f"   {module_name}: ❌ ERROR - {benchmark['error']}")
                continue

            print(f"   {module_name}:")
            for benchmark_type, results in benchmark.get("benchmarks", {}).items():
                if "performance_score" in results:
                    score = results["performance_score"]
                    status = (
                        "🟢 Excellent"
                        if score > 0.8
                        else "🟡 Good"
                        if score > 0.5
                        else "🔴 Needs Optimization"
                    )
                    print(f"     {benchmark_type}: {status} (score: {score:.2f})")

                    # Show key metrics
                    if "load_time_seconds" in results:
                        print(f"       Load time: {results['load_time_seconds']:.2f}s")
                    if "memory_increase_mb" in results:
                        print(f"       Memory increase: {results['memory_increase_mb']:.1f} MB")
                    if "file_size_mb" in results:
                        print(f"       File size: {results['file_size_mb']:.1f} MB")

        # Print recommendations
        all_recommendations = []
        for benchmark in all_benchmarks.values():
            if "recommendations" in benchmark:
                all_recommendations.extend(benchmark["recommendations"])

        if all_recommendations:
            print(f"\n💡 Performance Recommendations ({len(all_recommendations)}):")
            for recommendation in all_recommendations:
                print(f"   • {recommendation}")
        else:
            print(
                "\n💡 Performance Recommendations: No specific recommendations - performance appears optimal"
            )

        print("\n" + "=" * 80)
        sys.exit(0)

    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("🌲 Starting Cascadia Agricultural Analysis Framework")
    logger.info("📊 Analysis Parameters:")
    logger.info(f"   H3 Resolution: {args.h3_resolution}")
    logger.info(f"   Target Counties: {args.counties}")
    logger.info(f"   Active Modules: {args.modules}")
    logger.info(f"   Export Format: {args.export_format}")
    logger.info(f"   Output Directory: {args.output_dir}")

    # Visualization options
    viz_options = []
    if args.generate_dashboard:
        viz_options.append("Interactive Dashboard")
    if args.lightweight_viz:
        viz_options.append("Lightweight Static Visualizations")
    if args.datashader_viz:
        viz_options.append("Datashader Visualizations")
    if args.deepscatter_viz:
        viz_options.append("Deepscatter Visualizations")

    if viz_options:
        logger.info(f"   Visualization Options: {', '.join(viz_options)}")
    else:
        logger.info("   Visualization: None (use --lightweight-viz for efficient options)")

    # Enhanced features
    if args.force_refresh:
        logger.info("   Force Refresh: Enabled (will refresh cached data)")
    if args.validate_h3:
        logger.info("   H3 Validation: Enabled")

    try:
        # Initialize analysis with enhanced modules
        backend, modules, data_manager, h3_fusion, real_data_acquisition, viz_engine = (
            initialize_analysis(args)
        )

        # Validate H3 operations if requested
        if args.validate_h3:
            logger.info("🔍 Validating H3 operations...")
            h3_validation = h3_fusion.validate_h3_operations()
            logger.info(f"H3 validation result: {h3_validation}")

            if h3_validation.get("errors"):
                logger.error(f"H3 validation failed: {h3_validation['errors']}")
                return

        # Run comprehensive analysis with enhanced data management
        if args.profile_run:
            # Profile end-to-end analysis
            import cProfile
            import pstats

            prof_dir = Path(args.output_dir) / "profile"
            prof_dir.mkdir(parents=True, exist_ok=True)
            prof_path = (
                prof_dir / f"cascadia_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            )
            profiler = cProfile.Profile()
            profiler.enable()
            redevelopment_scores, summary, module_data = (
                run_comprehensive_analysis_with_enhanced_data(
                    backend,
                    modules,
                    data_manager,
                    h3_fusion,
                    real_data_acquisition,
                    viz_engine,
                    args,
                )
            )
            profiler.disable()
            with open(prof_path, "w") as f:
                ps = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
                ps.print_stats(50)
            logger.info(f"📈 Profile written: {prof_path}")
        else:
            redevelopment_scores, summary, module_data = (
                run_comprehensive_analysis_with_enhanced_data(
                    backend,
                    modules,
                    data_manager,
                    h3_fusion,
                    real_data_acquisition,
                    viz_engine,
                    args,
                )
            )

        # Export results with visualization options
        export_paths = export_results_with_visualizations(
            backend, redevelopment_scores, summary, args
        )

        # Generate reports
        generate_reports(summary, args.output_dir, args.spatial_analysis)

        # Run GEO-INFER integration suite on fused data (before map so results can be passed)
        integration_results: dict = {}
        try:
            from src.core.geo_infer_integrations import (
                build_integration_suite,
                get_availability_report,
            )

            availability = get_availability_report()
            available_modules = [k for k, v in availability.items() if v]
            if available_modules:
                logger.info(f"🔗 GEO-INFER integrations available: {available_modules}")
                suite = build_integration_suite()
                integration_results["data_quality"] = suite["data_quality"].validate_module_outputs(
                    module_data
                )
                summary["geo_infer_integrations"] = integration_results
            else:
                logger.info("ℹ️  GEO-INFER integration modules not installed (optional enrichment)")
        except Exception as e:
            logger.debug(f"GEO-INFER integration suite skipped: {e}")

        # Generate bioregion map if --bioregion flag set
        if getattr(args, "bioregion", False):
            try:
                from geo_infer_place.core.bioregion_visualization import (
                    create_bioregion_map,
                )

                config_dir = Path(__file__).resolve().parent / "config"
                bioregion_map_path = (
                    Path(args.output_dir)
                    / "visualizations"
                    / "interactive"
                    / "cascadia_bioregion_map.html"
                )
                bioregion_map_path.parent.mkdir(parents=True, exist_ok=True)
                create_bioregion_map(
                    config_dir,
                    redevelopment_scores,
                    bioregion_map_path,
                    integration_results=integration_results,
                )
                summary["bioregion_map_path"] = str(bioregion_map_path)
                logger.info(f"🗺️ Bioregion map: {bioregion_map_path}")
            except Exception as e:
                summary["bioregion_map_error"] = str(e)
                logger.warning(f"Bioregion map generation failed: {e}")

        # Print summary
        print_analysis_summary(summary, export_paths, args)

        logger.info("✅ Cascadia analysis completed successfully!")

        # Start server if --serve flag set
        if getattr(args, "serve", False):
            try:
                import subprocess

                port = getattr(args, "server_port", 8765)
                logger.info(f"🌐 Starting Cascadia server at http://localhost:{port}")
                server_script = Path(__file__).resolve().parent / "cascadia_server.py"
                subprocess.run(
                    [
                        sys.executable,
                        str(server_script),
                        "--port",
                        str(port),
                        "--output-dir",
                        str(args.output_dir),
                        "--open-browser",
                    ],
                    check=False,
                )
            except Exception as e:
                logger.error(f"Failed to start server: {e}")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_comprehensive_analysis_with_enhanced_data(
    backend, modules, data_manager, h3_fusion, real_data_acquisition, viz_engine, args
):
    """Run comprehensive analysis with enhanced data management, real acquisition, and H3 fusion.

    Returns ``(redevelopment_scores, summary, module_data)``; ``module_data`` maps
    each module name to its H3 output for downstream validation.

    Implements parallel module processing when --parallelism > 0, using threads for I/O-bound
    acquisition and CPU-backed geospatial ops that release the GIL in vectorized libs.
    """
    logger = logging.getLogger(__name__)

    logger.info(
        "🚀 Starting enhanced comprehensive analysis with real data acquisition and H3 geospatial fusion..."
    )

    # Collect data from all modules with enhanced data management
    module_data = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_single_module(module_name: str, module) -> tuple[str, dict | None]:
        logger.info(f"📊 Processing module: {module_name}")
        try:
            # Ecology module: loads YAML data and overlays H3 grid — no real_data_path pipeline
            if module_name == "ecology":
                raw_summary = module.acquire_raw_data()
                logger.info(f"🌿 Ecology raw data: {raw_summary}")
                # Build minimal lat/lon h3_data for overlay
                hex_with_coords = {}
                for hex_id in list(backend.target_hexagons)[:5000]:
                    try:
                        lat, lon = cell_to_latlng(hex_id)
                        hex_with_coords[hex_id] = {"lat": lat, "lon": lon}
                    except Exception as exc:
                        logger.warning(
                            "Cascadia analysis step failed; continuing without it: %s", exc
                        )
                h3_data = module.run_final_analysis(hex_with_coords)
                return module_name, h3_data

            real_data_path = None
            if module_name == "zoning":
                real_data_path = real_data_acquisition.acquire_zoning_data()
            elif module_name == "current_use":
                real_data_path = real_data_acquisition.acquire_current_use_data()
            elif module_name == "ownership" and hasattr(
                real_data_acquisition, "acquire_ownership_data"
            ):
                real_data_path = real_data_acquisition.acquire_ownership_data()  # type: ignore[attr-defined]
            elif module_name == "improvements" and hasattr(
                real_data_acquisition, "acquire_improvements_data"
            ):
                real_data_path = real_data_acquisition.acquire_improvements_data()  # type: ignore[attr-defined]

            use_real = bool(real_data_path and real_data_path.exists())
            if use_real:
                logger.info(f"✅ Using real data for {module_name}: {real_data_path}")
                data_path = real_data_path
            else:
                logger.info(f"ℹ️  Acquiring data via standard module extraction for {module_name}")
                data_path = data_manager.acquire_data_with_caching(
                    module_name=module_name,
                    data_source_func=module.acquire_raw_data,
                    force_refresh=(args.force_refresh or getattr(args, "skip_cache", False)),
                )

            h3_data = data_manager.process_to_h3_with_caching(
                data_path=data_path,
                module_name=module_name,
                target_hexagons=list(backend.target_hexagons),
            )
            quality_report = data_manager.get_data_quality_report(module_name)
            empirical_sources = [
                k
                for k, v in quality_report.get("data_sources", {}).items()
                if v.get("is_empirical", False)
            ]
            if empirical_sources:
                logger.info(
                    f"📋 {module_name} data quality: {quality_report.get('quality_metrics', {}).get('empirical', {}).get('quality_score', 0):.2f} (empirical: {empirical_sources})"
                )
            else:
                logger.info(
                    f"📋 {module_name} data quality: no empirical source was accepted, recommendations: {quality_report.get('recommendations', [])}"
                )
            return module_name, h3_data
        except Exception as e:
            logger.error(f"❌ Failed to process {module_name}: {e}")
            return module_name, None

    workers = (
        args.parallelism
        if args.parallelism and args.parallelism > 0
        else min(8, max(1, os.cpu_count() or 2))
    )
    # Keep a small thread pool due to external I/O; geospatial ops often release GIL
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_module, name, mod): name
            for name, mod in modules.items()
        }
        for future in as_completed(futures):
            mn, res = future.result()
            if res is not None:
                module_data[mn] = res

        # Perform enhanced H3 geospatial fusion
    logger.info("🔗 Performing enhanced H3 geospatial fusion...")
    # Flatten module_data to hex maps for fusion
    flat_sources = {}
    for _mn, _data in module_data.items():
        if isinstance(_data, dict) and "hexagons" in _data and isinstance(_data["hexagons"], dict):
            flat_sources[_mn] = _data["hexagons"]
        elif isinstance(_data, dict):
            flat_sources[_mn] = _data
        else:
            flat_sources[_mn] = {}

    # Ensure line-based modules have polygonized features upstream via acquire_raw_data

    # Try loading fusion result from cache unless skipping cache
    fused_data = None
    if not getattr(args, "skip_cache", False):
        try:
            fused_data = h3_fusion.load_fusion_cache(flat_sources, list(backend.target_hexagons))
        except Exception as e:
            logger.warning(f"Fusion cache load attempt failed: {e}")

    if not fused_data or getattr(args, "force_refresh", False):
        fused_data = h3_fusion.fuse_geospatial_data(
            data_sources=flat_sources, target_hexagons=list(backend.target_hexagons)
        )
        try:
            # Save report alongside cache for observability
            fusion_report = {
                "modules": list(flat_sources.keys()),
                "target_hex_count": len(backend.target_hexagons),
                "fused_hex_count": len(fused_data),
            }
            h3_fusion.save_fusion_cache(
                flat_sources, list(backend.target_hexagons), fused_data, fusion_report
            )
        except Exception as e:
            logger.warning(f"Fusion cache save failed: {e}")

    # Run final analysis on fused data with caching
    logger.info("🔬 Running final analysis on fused data...")
    redevelopment_scores = {}

    # Compute fusion signature to key score cache
    scores_cache_dir = Path(args.output_dir) / "data" / "cache" / "scores"
    scores_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fusion_cache_key = h3_fusion._compute_fusion_signature(
            flat_sources, list(backend.target_hexagons)
        )  # noqa: SLF001 (intentional use)
    except Exception:
        fusion_cache_key = f"res{args.h3_resolution}_default"
    scores_cache_file = scores_cache_dir / f"scores_{fusion_cache_key}.json"

    # Try load scores from cache
    if (
        scores_cache_file.exists()
        and not getattr(args, "skip_cache", False)
        and not getattr(args, "force_refresh", False)
    ):
        try:
            with open(scores_cache_file) as f:
                payload = json.load(f)
            cached = payload.get("scores", {})
            # Validate count roughly matches fused set
            if isinstance(cached, dict) and len(cached) >= int(0.6 * max(1, len(fused_data))):
                redevelopment_scores = {k: float(v) for k, v in cached.items()}
                logger.info(
                    f"Loaded redevelopment scores from cache: {scores_cache_file} ({len(redevelopment_scores)} hexes)"
                )
            else:
                logger.info("Scores cache present but insufficient coverage; recalculating.")
        except Exception as e:
            logger.warning(f"Failed to load scores cache: {e}; recalculating.")

    # Compute scores if not loaded
    if not redevelopment_scores or getattr(args, "force_refresh", False):
        for hex_id, hex_data in fused_data.items():
            try:
                score = calculate_enhanced_redevelopment_score(hex_data)
                redevelopment_scores[hex_id] = score
            except Exception as e:
                logger.warning(f"Error calculating score for {hex_id}: {e}")
                redevelopment_scores[hex_id] = 0.0
        # Persist scores cache
        try:
            cache_payload = {
                "meta": {
                    "h3_resolution": int(args.h3_resolution),
                    "fused_hex_count": int(len(fused_data)),
                    "generated_at": datetime.now().isoformat(),
                },
                "scores": redevelopment_scores,
            }
            with open(scores_cache_file, "w") as f:
                json.dump(cache_payload, f)
            logger.info(f"Saved redevelopment scores cache: {scores_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save scores cache: {e}")

    # Persist fused data into backend for export/reporting
    try:
        # Flatten fused data into backend.unified_data structure
        # Each fused hex -> dict of module_name -> list[dict]
        backend.unified_data = {}
        for hex_id in backend.target_hexagons:
            backend.unified_data[hex_id] = {
                "hex_id": hex_id,
                "boundary": (
                    cell_to_latlng_boundary(hex_id)
                    if "cell_to_latlng_boundary" in globals()
                    else []
                ),
            }
        for module_name, module_hex_map in flat_sources.items():
            for hex_id, items in module_hex_map.items():
                if hex_id not in backend.unified_data:
                    backend.unified_data[hex_id] = {"hex_id": hex_id}
                # Summarize and score per-module data for this hex
                rep = (
                    items[0]
                    if isinstance(items, list) and items
                    else (items if isinstance(items, dict) else {})
                )
                scored = _summarize_and_score_module(module_name, rep)
                backend.unified_data[hex_id][module_name] = scored
        # Calculate redevelopment scores using backend logic
        _ = backend.calculate_agricultural_redevelopment_potential()
    except Exception as e:
        logger.error(f"Failed to persist fused data into backend: {e}")

    # Generate summary using backend to include module coverage properly
    backend_summary = backend.get_comprehensive_summary()

    summary = {
        "total_hexagons": len(backend.target_hexagons),
        "processed_hexagons": len(fused_data),
        "module_count": len(modules),
        "data_sources": list(module_data.keys()),
        "h3_resolution": args.h3_resolution,
        "spatial_analysis_enabled": args.spatial_analysis,
        "enhanced_data_management": True,
        "h3_fusion_enabled": True,
        "real_data_acquisition": True,
        "interactive_visualization": True,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    # Merge in backend module coverage and redevelopment stats
    try:
        summary["modules_analyzed"] = backend_summary.get("modules_analyzed", [])
        summary["module_summaries"] = backend_summary.get("module_summaries", {})
        summary["redevelopment_potential"] = backend_summary.get("redevelopment_potential", {})
        summary["bioregion"] = backend_summary.get("bioregion", "Cascadia")
    except Exception as exc:
        logger.warning("Cascadia analysis step failed; continuing without it: %s", exc)

    # Create comprehensive visualizations
    if args.generate_dashboard or args.lightweight_viz:
        logger.info("🎨 Creating comprehensive visualizations...")
        try:
            # Create interactive H3 map
            # Resolve layer visibility/include overrides from CLI/env
            visible_layers_env = os.environ.get("CASCADIA_VISIBLE_LAYERS")
            visible_layers: list[str] | None = None
            include_layers: list[str] | None = None
            if args.visible_layers:
                visible_layers = [
                    layer.strip() for layer in args.visible_layers.split(",") if layer.strip()
                ]
            elif visible_layers_env:
                visible_layers = [
                    layer.strip() for layer in visible_layers_env.split(",") if layer.strip()
                ]
            if args.include_layers:
                include_layers = [
                    layer.strip() for layer in args.include_layers.split(",") if layer.strip()
                ]
            # Build per-module status for interactive HTML panel
            module_status: dict[str, Any] = {}
            try:
                for mod_name, _data in module_data.items():
                    ds = data_manager.get_data_structure(mod_name)
                    cache_file = ds.get("h3_cache")
                    hex_count = 0
                    input_features = 0
                    if isinstance(_data, dict):
                        hex_map = _data.get("hexagons", _data)
                        if isinstance(hex_map, dict):
                            hex_count = len(hex_map)
                        input_features = (
                            int(_data.get("input_features", 0)) if "input_features" in _data else 0
                        )
                    module_status[mod_name] = {
                        "h3_cache": str(cache_file),
                        "output_hexagons": hex_count,
                        "input_features": input_features,
                        "empirical_exists": ds["empirical_data"].exists(),
                        "raw_exists": ds["raw_data"].exists(),
                    }
            except Exception as exc:
                logger.warning("Cascadia analysis step failed; continuing without it: %s", exc)

            interactive_map = viz_engine.create_interactive_h3_map(
                h3_data=fused_data,
                data_sources=module_data,
                target_hexagons=list(backend.target_hexagons)[:5000],
                output_filename="cascadia_interactive_map.html",
                initial_visible_layers=visible_layers,
                include_layers=include_layers,
                module_status=module_status,
                redevelopment_scores=redevelopment_scores,
            )
            summary["interactive_map_path"] = str(interactive_map)

            # Create static visualizations
            static_viz = viz_engine.create_static_visualizations(
                h3_data=fused_data,
                data_sources=module_data,
                redevelopment_scores=redevelopment_scores,
            )
            summary["static_visualizations"] = {k: str(v) for k, v in static_viz.items()}

            # Create comprehensive dashboard
            if args.generate_dashboard:
                dashboard_path = viz_engine.create_dashboard(
                    h3_data=fused_data,
                    data_sources=module_data,
                    redevelopment_scores=redevelopment_scores,
                    summary=summary,
                )
                summary["dashboard_path"] = str(dashboard_path)

            # Export visualization data
            export_paths = viz_engine.export_visualization_data(
                h3_data=fused_data,
                data_sources=module_data,
                redevelopment_scores=redevelopment_scores,
            )
            summary["export_paths"] = {k: str(v) for k, v in export_paths.items()}

            logger.info("✅ Comprehensive visualizations created")
            logger.info(f"   Interactive Map: {interactive_map}")
            logger.info(f"   Static Visualizations: {len(static_viz)} files")
            if args.generate_dashboard:
                logger.info(f"   Dashboard: {dashboard_path}")
            logger.info(f"   Export Files: {len(export_paths)} files")

        except Exception as e:
            logger.error(f"❌ Failed to create visualizations: {e}")

    # Persist data provenance manifest
    try:
        provenance = {
            "timestamp": datetime.now().isoformat(),
            "h3_resolution": args.h3_resolution,
            "modules": {},
            "fusion_cache": str(Path(args.output_dir) / "data" / "cache" / "fusion"),
            "scores_cache": str(scores_cache_file),
        }
        for mod_name, _data in module_data.items():
            # detect cache path and counts if available
            ds = data_manager.get_data_structure(mod_name)
            cache_file = ds.get("h3_cache")
            output_hex = 0
            input_features = 0
            if isinstance(_data, dict):
                hex_map = _data.get("hexagons", _data)
                if isinstance(hex_map, dict):
                    output_hex = len(hex_map)
                input_features = (
                    int(_data.get("input_features", 0)) if "input_features" in _data else 0
                )
            provenance["modules"][mod_name] = {
                "h3_cache": str(cache_file),
                "output_hexagons": output_hex,
                "input_features": input_features,
                "empirical_exists": ds["empirical_data"].exists(),
                "raw_exists": ds["raw_data"].exists(),
            }
        export_data_provenance(provenance, Path(args.output_dir))
    except Exception as e:
        logger.warning(f"Failed to write provenance manifest: {e}")

    return redevelopment_scores, summary, module_data


def _summarize_and_score_module(module_name: str, rep: dict) -> dict:
    """Create a compact, scored representation for a module's per-hex data.

    Ensures a numeric 'score' is present so downstream summaries/plots are non-empty.
    """
    try:
        rep = dict(rep) if isinstance(rep, dict) else {}
        name = module_name.lower()
        if name == "zoning":
            zone = str(rep.get("zone_type", rep.get("zone", ""))).lower()
            is_ag = any(k in zone for k in ["ag", "agricultural", "agriculture"])
            allows_redev = not any(k in zone for k in ["conservation", "preserve"])
            score = 0.6 if is_ag else 0.3
            if allows_redev:
                score += 0.2
            return {
                "zone_type": rep.get("zone_type", rep.get("zone", "Unknown")),
                "zone_code": rep.get("zone_code", rep.get("code", "")),
                "is_ag_zone": bool(is_ag),
                "allows_redevelopment": bool(allows_redev),
                "score": float(max(0.0, min(1.0, score))),
            }
        if name == "current_use":
            crop = str(rep.get("crop_type", rep.get("land_use", "")))
            intensity_str = str(
                rep.get("intensity", rep.get("agricultural_intensity", "medium"))
            ).lower()
            intensity = (
                0.7 if intensity_str == "high" else (0.4 if intensity_str == "medium" else 0.2)
            )
            # Easier redevelopment when intensity is lower
            score = 1.0 - intensity
            return {
                "primary_use": rep.get("land_use", crop),
                "crop_type": crop,
                "agricultural_intensity": float(intensity),
                "score": float(max(0.0, min(1.0, score))),
            }
        if name == "ownership":
            owner_type = str(rep.get("owner_type", rep.get("owner_category", ""))).lower()
            # Assume individual < trust < corporate concentration
            concentration = (
                0.3
                if "individual" in owner_type
                else (0.6 if "trust" in owner_type else 0.8 if owner_type else 0.5)
            )
            score = 1.0 - concentration
            return {
                "owner_type": rep.get("owner_type", rep.get("owner_category", "Unknown")),
                "ownership_concentration": float(concentration),
                "score": float(max(0.0, min(1.0, score))),
            }
        if name == "improvements":
            bval = rep.get("building_value", 0) or 0
            try:
                bnorm = float(bval) / 200000.0
            except Exception:
                bnorm = 0.0
            modernization = max(0.0, min(1.0, bnorm))
            # Higher modernization can either hinder or help; here lower modernization => easier redevelopment
            score = 1.0 - modernization
            return {
                "improvement_type": rep.get("improvement_type", "Unknown"),
                "modernization_score": float(modernization),
                "score": float(max(0.0, min(1.0, score))),
            }
        # Default: create value score from any numeric field
        numeric_vals = [v for v in rep.values() if isinstance(v, (int, float))]
        score = float(
            max(
                0.0,
                min(
                    1.0,
                    ((sum(numeric_vals) / len(numeric_vals) / 100.0) if numeric_vals else 0.0),
                ),
            )
        )
        rep["score"] = score
        return rep
    except Exception:
        return {"score": 0.0}


def calculate_enhanced_redevelopment_score(hex_data):
    """Calculate enhanced redevelopment score based on fused data"""
    score = 0.0
    factors = []

    # Analyze zoning data
    if "zoning" in hex_data:
        zoning_data = hex_data["zoning"]
        if isinstance(zoning_data, list) and len(zoning_data) > 0:
            # Extract zoning information
            zoning_score = 0.0
            for zoning_item in zoning_data:
                if isinstance(zoning_item, dict):
                    zone_type = zoning_item.get("zone_type", "").lower()
                    if "agricultural" in zone_type:
                        zoning_score += 0.3
                    elif "residential" in zone_type:
                        zoning_score += 0.1
            factors.append(zoning_score)

    # Analyze current use data
    if "current_use" in hex_data:
        current_use_data = hex_data["current_use"]
        if isinstance(current_use_data, list) and len(current_use_data) > 0:
            # Extract current use information
            use_score = 0.0
            for use_item in current_use_data:
                if isinstance(use_item, dict):
                    crop_type = use_item.get("crop_type", "").lower()
                    if "hay" in crop_type or "alfalfa" in crop_type:
                        use_score += 0.2
                    elif "vegetables" in crop_type:
                        use_score += 0.3
            factors.append(use_score)

    # Analyze ownership data
    if "ownership" in hex_data:
        ownership_data = hex_data["ownership"]
        if isinstance(ownership_data, list) and len(ownership_data) > 0:
            # Extract ownership information
            ownership_score = 0.0
            for ownership_item in ownership_data:
                if isinstance(ownership_item, dict):
                    owner_type = ownership_item.get("owner_type", "").lower()
                    if "individual" in owner_type:
                        ownership_score += 0.2
                    elif "corporate" in owner_type:
                        ownership_score += 0.1
            factors.append(ownership_score)

    # Analyze improvements data
    if "improvements" in hex_data:
        improvements_data = hex_data["improvements"]
        if isinstance(improvements_data, list) and len(improvements_data) > 0:
            # Extract improvements information
            improvements_score = 0.0
            for improvement_item in improvements_data:
                if isinstance(improvement_item, dict):
                    improvement_type = improvement_item.get("improvement_type", "").lower()
                    if "barn" in improvement_type:
                        improvements_score += 0.2
                    elif "house" in improvement_type:
                        improvements_score += 0.1
            factors.append(improvements_score)

    # Calculate final score
    if factors:
        score = sum(factors) / len(factors)

    return min(score, 1.0)  # Normalize to 0-1 range


def export_results_with_visualizations(backend, redevelopment_scores, summary, args):
    """Export results with selected visualization options"""
    logger = logging.getLogger(__name__)

    bioregion_lower = "cascadia"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)

    # Export core data
    export_paths = export_results(
        backend,
        redevelopment_scores,
        summary,
        output_dir,
        timestamp,
        bioregion_lower,
        args.export_format,
    )

    # Generate selected visualizations
    if args.lightweight_viz:
        logger.info("📊 Generating lightweight static visualizations...")
        try:
            from src.core.visualization.static_visualization import create_static_plots

            static_results = create_static_plots(backend, output_dir)
            export_paths.update(static_results)
            logger.info("✅ Lightweight visualizations created")
        except Exception as e:
            logger.error(f"Failed to create lightweight visualizations: {e}")

    if args.datashader_viz:
        logger.info("📊 Generating Datashader visualizations...")
        try:
            from src.core.visualization.datashader_visualization import (
                create_datashader_visualization,
            )

            datashader_results = create_datashader_visualization(backend, output_dir)
            export_paths.update(datashader_results)
            logger.info("✅ Datashader visualizations created")
        except Exception as e:
            logger.error(f"Failed to create Datashader visualizations: {e}")

    if args.deepscatter_viz and not getattr(args, "skip_deepscatter", False):
        logger.info("📊 Generating Deepscatter visualizations...")
        try:
            from src.core.visualization.deepscatter_visualization import (
                create_deepscatter_visualization,
            )

            deepscatter_results = create_deepscatter_visualization(backend, output_dir)
            export_paths.update(deepscatter_results)
            logger.info("✅ Deepscatter visualizations created")
        except Exception as e:
            logger.error(f"Failed to create Deepscatter visualizations: {e}")

    return export_paths


def print_analysis_summary(summary, export_paths, args):
    """Print a comprehensive analysis summary"""
    print("\n" + "=" * 80)
    print("🌲 CASCADIA AGRICULTURAL ANALYSIS SUMMARY")
    print("=" * 80)

    # Analysis statistics
    print("\n📊 Analysis Statistics:")
    print(f"   Total Hexagons: {summary.get('total_hexagons', 0):,}")
    print(f"   Processed Hexagons: {summary.get('processed_hexagons', 0):,}")
    print(f"   Data Sources: {len(summary.get('data_sources', []))}")
    print(f"   H3 Resolution: {summary.get('h3_resolution', 8)}")

    # Enhanced features
    print("\n🔧 Enhanced Features:")
    print(f"   Enhanced Data Management: {summary.get('enhanced_data_management', False)}")
    print(f"   H3 Fusion Enabled: {summary.get('h3_fusion_enabled', False)}")
    print(f"   Spatial Analysis: {summary.get('spatial_analysis_enabled', False)}")

    # Data sources
    print("\n📋 Data Sources:")
    for source in summary.get("data_sources", []):
        print(f"   • {source}")

    # Visualization outputs
    print("\n🎨 Generated Visualizations:")
    viz_files = []
    for key, path in export_paths.items():
        if any(keyword in key.lower() for keyword in ["viz", "dashboard", "html", "csv", "json"]):
            viz_files.append(f"   {key}: {path}")

    if viz_files:
        for file_info in viz_files:
            print(file_info)
    else:
        print("   No visualizations generated (use --lightweight-viz for efficient options)")

    # Recommendations
    print("\n💡 Recommendations:")
    if not any(
        [
            args.lightweight_viz,
            args.datashader_viz,
            args.deepscatter_viz,
            args.generate_dashboard,
        ]
    ):
        print("   • Use --lightweight-viz for efficient static visualizations")
        print("   • Use --datashader-viz for large dataset rendering")
        print("   • Use --deepscatter-viz for web-based interactive plots")

    if args.generate_dashboard:
        print(
            "   • Consider using --lightweight-viz instead of --generate-dashboard for better performance"
        )

    if not args.spatial_analysis:
        print("   • Use --spatial-analysis for enhanced spatial correlation analysis")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
