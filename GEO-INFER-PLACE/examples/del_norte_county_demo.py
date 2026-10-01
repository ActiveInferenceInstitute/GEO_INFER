#!/usr/bin/env python3
"""
Del Norte County Comprehensive Dashboard Demonstration

This script demonstrates the full capabilities of the Del Norte County comprehensive
dashboard including real California data integration, H3 spatial analysis, multi-domain
analysis (forest health, coastal resilience, fire risk), and interactive visualization
generation adapted from the climate integration example.

Usage:
    python del_norte_county_demo.py [--config CONFIG_PATH] [--output OUTPUT_DIR] [--api-keys API_KEYS_FILE]

Example:
    python del_norte_county_demo.py --output ./del_norte_dashboard --api-keys api_keys.json

Required packages:
    - folium
    - h3
    - pandas
    - geopandas
    - numpy
    - requests
"""

import sys
import importlib
import argparse
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import traceback

from geo_infer_place.locations.del_norte_county.comprehensive_dashboard import (
    DelNorteComprehensiveDashboard,
)
from geo_infer_place.locations.del_norte_county.dashboard.core import AdvancedDashboard
from geo_infer_place.utils.data_sources import CaliforniaDataSources
from geo_infer_place.core.api_clients import CaliforniaAPIManager

# Log to the system temp directory — never scatter log files into the cwd
_log_path = Path(tempfile.gettempdir()) / "del_norte_demo.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(_log_path)],
)

logger = logging.getLogger(__name__)
logger.info("Demo log file: %s", _log_path)


def check_dependencies():
    """Verify required dependencies are importable.

    This example never installs packages: mutating the user's environment at
    runtime is the package manager's job. Missing deps are a hard error with
    actionable instructions.

    Returns:
        True when all required packages are importable.
    """
    logger.info("=== Dependency Check ===")

    # Required packages — all declared in pyproject.toml [project.dependencies]
    required_packages = {
        "folium": "folium",
        "h3": "h3",
        "pandas": "pandas",
        "geopandas": "geopandas",
        "numpy": "numpy",
        "requests": "requests",
        "yaml": "pyyaml",
        "plotly": "plotly",
        "matplotlib": "matplotlib",
        "branca": "branca",
        "shapely": "shapely",
    }

    available_packages = []
    missing_packages = []

    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            available_packages.append(module_name)
            logger.info("✓ %s is available", module_name)
        except ImportError:
            missing_packages.append(package_name)
            logger.warning("✗ %s is missing", module_name)

    if missing_packages:
        logger.error(
            "Missing required packages: %s. Install them with:\n"
            "  uv add %s\n"
            "or:\n"
            "  uv pip install %s",
            ", ".join(missing_packages),
            " ".join(missing_packages),
            " ".join(missing_packages),
        )
        return False

    logger.info(
        "All %d required packages available: %s",
        len(available_packages),
        ", ".join(available_packages),
    )
    return True


# Check dependencies first
print("=== Checking Dependencies ===")
if not check_dependencies():
    print("Dependencies check failed. Install the missing packages listed above.")
    sys.exit(1)

# Now import the modules after ensuring dependencies are available
print("=== Importing Core Modules ===")
try:
    import folium

    print(f"✓ folium version: {folium.__version__}")
except ImportError as e:
    print(f"✗ folium import failed: {e}")

try:
    import h3

    print(f"✓ h3 version: {h3.__version__}")
except ImportError as e:
    print(f"✗ h3 import failed: {e}")

try:
    import pandas as pd

    print(f"✓ pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np

    print(f"✓ numpy version: {np.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import requests

    print(f"✓ requests version: {requests.__version__}")
except ImportError as e:
    print(f"✗ requests import failed: {e}")

# Try to import advanced visualization packages
try:
    import plotly

    print(f"✓ plotly version: {plotly.__version__}")
except ImportError as e:
    print(f"○ plotly not available: {e}")

try:
    import matplotlib

    print(f"✓ matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"○ matplotlib not available: {e}")

try:
    import seaborn

    print(f"✓ seaborn version: {seaborn.__version__}")
except ImportError as e:
    print(f"○ seaborn not available: {e}")

print("=== Core Module Import Complete ===")
print()


def load_api_keys(api_keys_file: str) -> dict:
    """Load API keys from JSON file."""
    try:
        with open(api_keys_file) as f:
            api_keys = json.load(f)
        logger.info(f"Loaded API keys from {api_keys_file}")
        return api_keys
    except FileNotFoundError:
        logger.warning(f"API keys file not found: {api_keys_file}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing API keys file: {e}")
        return {}


def demonstrate_data_sources():
    """Demonstrate the California data sources catalog."""
    logger.info("=== Demonstrating California Data Sources Catalog ===")

    data_sources = CaliforniaDataSources()

    # Get summary of available data sources
    summary = data_sources.get_source_summary()
    logger.info(f"Total data sources available: {summary['total_sources']}")
    logger.info(f"Categories: {list(summary['categories'].keys())}")

    # Show fire-related data sources
    fire_sources = data_sources.get_sources_by_category("fire")
    logger.info(f"Fire-related data sources: {len(fire_sources)}")
    for source in fire_sources[:3]:  # Show first 3
        logger.info(f"  - {source.name}: {source.description[:100]}...")

    # Show coastal data sources
    coastal_sources = data_sources.get_sources_by_category("coastal")
    logger.info(f"Coastal data sources: {len(coastal_sources)}")

    # Demonstrate location-specific source discovery
    del_norte_bounds = (
        -124.4,
        41.5,
        -123.5,
        42.0,
    )  # Del Norte County approximate bounds
    location_sources = data_sources.get_sources_for_location(
        location_bounds=del_norte_bounds, location_name="Del Norte County"
    )

    logger.info(
        f"High-priority sources for Del Norte County: {len(location_sources['high_priority'])}"
    )
    logger.info(f"Medium-priority sources: {len(location_sources['medium_priority'])}")

    return data_sources


def demonstrate_api_connections(api_keys: dict):
    """Demonstrate API connections and validation."""
    logger.info("=== Demonstrating API Connections ===")

    api_manager = CaliforniaAPIManager()

    # Test individual API connections
    logger.info("Testing individual API endpoints...")

    # Test CAL FIRE
    try:
        calfire_data = api_manager.calfire.fetch_perimeters(year=2023)
        status = "✓ Connected" if calfire_data else "✗ Failed"
        logger.info(f"CAL FIRE: {status}")
    except Exception as e:
        logger.warning(f"CAL FIRE: ✗ Failed - {e}")

    # Test NOAA
    try:
        noaa_data = api_manager.noaa.fetch_tide_data("9419750", "20240101", "20240102")
        status = "✓ Connected" if noaa_data else "✗ Failed"
        logger.info(f"NOAA: {status}")
    except Exception as e:
        logger.warning(f"NOAA: ✗ Failed - {e}")

    # Test USGS (if available)
    try:
        usgs_data = api_manager.usgs.fetch_water_data("20240101", "20240102")
        status = "✓ Connected" if usgs_data else "✗ Failed"
        logger.info(f"USGS: {status}")
    except Exception as e:
        logger.warning(f"USGS: ✗ Failed - {e}")

    # Test CDEC (if available)
    try:
        cdec_data = api_manager.cdec.fetch_sensor_data(
            "DNP", "1", "20240101", "20240102"
        )
        status = "✓ Connected" if cdec_data else "✗ Failed"
        logger.info(f"CDEC: {status}")
    except Exception as e:
        logger.warning(f"CDEC: ✗ Failed - {e}")

    return api_manager


def run_comprehensive_demo(
    config_path: str = None, output_dir: str = None, api_keys: dict = None
):
    """Run the comprehensive Del Norte County dashboard demonstration."""
    logger.info("=== Starting Del Norte County Comprehensive Dashboard Demo ===")

    try:
        # Initialize dashboard
        logger.info("Initializing comprehensive dashboard...")
        dashboard = DelNorteComprehensiveDashboard(
            config_path=config_path,
            api_keys=api_keys or {},
            h3_resolution=8,
            output_dir=output_dir or "./del_norte_dashboard",
        )

        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration...")
        config = dashboard.load_configuration()
        logger.info(
            f"Configuration loaded for location: {config.get('location_code', 'Unknown')}"
        )

        if dashboard.location_bounds:
            logger.info(f"Location bounds: {dashboard.location_bounds.to_bbox()}")
            center = dashboard.location_bounds.center()
            logger.info(f"Center coordinates: {center[0]:.4f}, {center[1]:.4f}")

        # Step 2: Fetch real data
        logger.info("Step 2: Fetching real data from APIs...")
        real_data = dashboard.fetch_real_data()

        successful_fetches = sum(1 for result in real_data.values() if result.success)
        total_fetches = len(real_data)
        logger.info(
            f"Data fetch results: {successful_fetches}/{total_fetches} successful"
        )

        for data_type, response in real_data.items():
            status = "✓" if response.success else "✗"
            logger.info(f"  {status} {data_type}: {response.error or 'Success'}")

        # Step 3: Run comprehensive analysis
        logger.info("Step 3: Running comprehensive analysis...")
        analysis_results = dashboard.run_comprehensive_analysis()

        logger.info(f"Analysis completed for {len(analysis_results)} domains:")
        for domain in analysis_results:
            logger.info(f"  ✓ {domain}")

        # Step 4: Generate interactive dashboard
        logger.info("Step 4: Generating interactive dashboard...")
        dashboard_path = dashboard.generate_comprehensive_dashboard()
        logger.info(f"Interactive dashboard generated: {dashboard_path}")

        # Step 5: Export analysis results
        logger.info("Step 5: Exporting analysis results...")
        results_path = dashboard.export_analysis_results()
        logger.info(f"Analysis results exported: {results_path}")

        # Step 6: Generate summary report
        logger.info("Step 6: Generating summary report...")
        summary_report = dashboard.generate_summary_report()

        # Save summary report
        summary_path = (
            Path(dashboard.output_dir)
            / f"del_norte_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(summary_path, "w") as f:
            f.write(summary_report)

        logger.info(f"Summary report saved: {summary_path}")

        # Display key results
        logger.info("=== Key Analysis Results ===")
        if "h3_aggregation" in analysis_results:
            h3_data = analysis_results["h3_aggregation"]
            logger.info("H3 spatial analysis:")
            logger.info(f"  - Total H3 cells: {h3_data.get('total_cells', 0)}")
            logger.info(
                f"  - Coverage area: {h3_data.get('coverage_area_km2', 0):.2f} km²"
            )
            logger.info(f"  - Resolution: {h3_data.get('resolution', 'Unknown')}")

        if "integration" in analysis_results:
            logger.info("Cross-domain integration:")
            logger.info(
                f"  - Climate vulnerability index: {dashboard._calculate_climate_vulnerability_index():.2f}"
            )
            logger.info(
                f"  - Integrated risk score: {dashboard._calculate_integrated_risk_score():.2f}"
            )

        # Success summary
        logger.info("=== Demo Completed Successfully ===")
        logger.info(f"Dashboard: {dashboard_path}")
        logger.info(f"Results: {results_path}")
        logger.info(f"Summary: {summary_path}")

        return {
            "success": True,
            "dashboard_path": dashboard_path,
            "results_path": results_path,
            "summary_path": str(summary_path),
            "analysis_results": analysis_results,
        }

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def demonstrate_h3_spatial_analysis():
    """Demonstrate H3 spatial analysis capabilities."""
    logger.info("=== Demonstrating H3 Spatial Analysis ===")

    import h3
    import numpy as np

    # Del Norte County center
    center_lat, center_lon = 41.75, -124.0

    # Generate H3 cells at different resolutions
    resolutions = [6, 7, 8, 9]
    for resolution in resolutions:
        h3_cell = h3.latlng_to_cell(center_lat, center_lon, resolution)
        cell_area = h3.cell_area(h3_cell, unit="km^2")

        # Get neighboring cells
        neighbors = sorted(set(h3.grid_disk(h3_cell, 1)) - {h3_cell})

        logger.info(f"Resolution {resolution}:")
        logger.info(f"  - Cell: {h3_cell}")
        logger.info(f"  - Area: {cell_area:.2f} km²")
        logger.info(f"  - Neighbors: {len(neighbors)}")

    # Generate a small H3 grid for Del Norte County
    logger.info("Generating H3 grid for Del Norte County...")

    # Approximate Del Norte County bounds
    north, south = 42.0, 41.5
    east, west = -123.5, -124.4

    # Generate grid of points
    lat_points = np.linspace(south, north, 10)
    lon_points = np.linspace(west, east, 10)

    h3_cells = set()
    for lat in lat_points:
        for lon in lon_points:
            cell = h3.latlng_to_cell(lat, lon, 8)
            h3_cells.add(cell)

    total_area = sum(h3.cell_area(cell, unit="km^2") for cell in h3_cells)

    logger.info(f"Generated {len(h3_cells)} H3 cells covering {total_area:.2f} km²")

    return h3_cells


def run_advanced_demo(output_dir: str = None, api_keys: dict = None):
    """Run the advanced geospatial intelligence dashboard demo."""
    logger.info("=== Running Advanced Geospatial Intelligence Dashboard Demo ===")

    try:
        # Initialize advanced dashboard
        logger.info(
            "Initializing advanced dashboard with real-time data integration..."
        )
        advanced_dashboard = AdvancedDashboard(
            output_dir=output_dir or "./del_norte_dashboard", api_keys=api_keys or {}
        )

        # Fetch real-time data
        logger.info("Fetching real-time California data...")
        real_time_data = advanced_dashboard.fetch_real_time_data()

        # Log data fetch results
        for data_type, response in real_time_data.items():
            if data_type == "fetch_timestamp":
                continue
            status = "✓" if response.get("success", False) else "✗"
            logger.info(f"  {status} {data_type}: {response.get('error', 'Success')}")

        # Generate comprehensive dashboard
        logger.info("Generating comprehensive intelligence dashboard...")
        dashboard_path = advanced_dashboard.save_dashboard()

        # Generate policy report
        logger.info("Generating policy support report...")
        policy_report = advanced_dashboard.generate_policy_report()

        # Save policy report
        report_path = (
            Path(advanced_dashboard.output_dir)
            / f"policy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(policy_report, f, indent=2, default=str)

        logger.info(f"Policy report saved: {report_path}")

        # Log key insights
        logger.info("=== Advanced Dashboard Key Features ===")
        logger.info("✓ Multi-panel layout with specialized analysis windows")
        logger.info("✓ Real-time California dataset integration")
        logger.info("✓ Interactive layer toggles and controls")
        logger.info("✓ Climate, zoning, and agro-economic analysis")
        logger.info("✓ H3 spatial indexing and forest health analysis")
        logger.info("✓ Policy scenario modeling capabilities")
        logger.info("✓ Advanced visualization and reporting")

        return {
            "success": True,
            "dashboard_path": dashboard_path,
            "policy_report_path": str(report_path),
            "real_time_data": real_time_data,
            "message": "Advanced dashboard generated successfully",
        }

    except Exception as e:
        logger.error(f"Advanced dashboard demo failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Del Norte County Comprehensive Dashboard Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python del_norte_county_demo.py
  python del_norte_county_demo.py --output ./dashboard_output
  python del_norte_county_demo.py --api-keys api_keys.json --output ./results
  python del_norte_county_demo.py --config custom_config.yaml --output ./custom_dashboard

API Keys File Format (JSON):
{
  "noaa": "your_noaa_api_key",
  "calfire": "your_calfire_api_key",
  "usgs": "your_usgs_api_key",
  "cdec": "your_cdec_api_key"
}
        """,
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")

    parser.add_argument(
        "--output",
        type=str,
        default="./del_norte_dashboard",
        help="Output directory for generated files (default: ./del_norte_dashboard)",
    )

    parser.add_argument(
        "--api-keys", type=str, help="Path to JSON file containing API keys"
    )

    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Run component demonstrations only (no full dashboard generation)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== Del Norte County Comprehensive Dashboard Demo ===")
    logger.info(f"Arguments: {vars(args)}")

    # Load API keys if provided
    api_keys = {}
    if args.api_keys:
        api_keys = load_api_keys(args.api_keys)

    # Run component demonstrations
    logger.info("Running component demonstrations...")

    # Demonstrate data sources catalog
    demonstrate_data_sources()

    # Demonstrate API connections
    demonstrate_api_connections(api_keys)

    # Demonstrate H3 spatial analysis
    demonstrate_h3_spatial_analysis()

    if args.demo_only:
        logger.info("Demo-only mode completed.")
        return

    # Generate the two data-backed dashboards.
    logger.info("=== Running Dashboard Generation ===")
    logger.info("Step 1: Generating advanced geospatial intelligence dashboard...")
    advanced_result = run_advanced_demo(output_dir=args.output, api_keys=api_keys)

    logger.info("Step 2: Generating comprehensive dashboard...")
    comprehensive_result = run_comprehensive_demo(
        config_path=args.config, output_dir=args.output, api_keys=api_keys
    )

    # Open dashboards in browser
    import webbrowser

    dashboard_paths = []

    if advanced_result.get("success", False):
        advanced_path = advanced_result["dashboard_path"]
        dashboard_paths.append(("Advanced Intelligence Dashboard", advanced_path))
        logger.info(f"Opening advanced dashboard: {advanced_path}")
        # Open advanced dashboard in a new tab after a small delay
        import threading
        import time

        def open_advanced():
            time.sleep(2)
            webbrowser.open(f"file://{Path(advanced_path).absolute()}")

        threading.Thread(target=open_advanced).start()

    if comprehensive_result and comprehensive_result.get("success", False):
        comp_path = comprehensive_result.get(
            "dashboard_path"
        ) or comprehensive_result.get("map_path")
        if comp_path:
            dashboard_paths.append(("Comprehensive Dashboard", comp_path))
            logger.info(f"Opening comprehensive dashboard: {comp_path}")

            def open_comprehensive():
                time.sleep(4)
                webbrowser.open(f"file://{Path(comp_path).absolute()}")

            threading.Thread(target=open_comprehensive).start()

    # Results summary
    success_count = sum(
        [
            advanced_result.get("success", False),
            comprehensive_result.get("success", False),
        ]
    )

    if success_count > 0:
        logger.info("=== DEMO COMPLETED SUCCESSFULLY ===")
        logger.info(f"Generated {success_count} dashboard(s):")

        if advanced_result.get("success", False):
            logger.info(
                f"  🗺️ Advanced Intelligence Dashboard: {advanced_result['dashboard_path']}"
            )
            if "policy_report_path" in advanced_result:
                logger.info(
                    f"  📄 Policy Report: {advanced_result['policy_report_path']}"
                )

        if comprehensive_result and comprehensive_result.get("success", False):
            if "dashboard_path" in comprehensive_result:
                logger.info(
                    f"  🔬 Comprehensive Dashboard: {comprehensive_result['dashboard_path']}"
                )
            if "results_path" in comprehensive_result:
                logger.info(
                    f"  📊 Analysis Results: {comprehensive_result['results_path']}"
                )
            if "summary_path" in comprehensive_result:
                logger.info(
                    f"  📋 Summary Report: {comprehensive_result['summary_path']}"
                )

        logger.info("")
        logger.info("=== Dashboard Outputs ===")
        if advanced_result.get("success", False):
            logger.info("Advanced Intelligence Dashboard:")
            logger.info("  • Multi-panel layout with specialized analysis")
            logger.info("  • Real-time California data integration")
            logger.info("  • Climate, zoning, and economic analysis")
            logger.info("  • Interactive layer controls and tools")
            logger.info("  • Policy support and scenario modeling")
            logger.info("  • H3 spatial indexing and analytics")

        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Compare the different dashboard approaches")
        logger.info("2. Explore interactive features in each dashboard")
        logger.info("3. Review the policy report for insights")
        logger.info("4. Customize configuration for specific needs")
        logger.info("5. Integrate with additional data sources")

    else:
        logger.error("=== ALL DEMOS FAILED ===")
        logger.error("Check the logs above for detailed error information.")

        if not advanced_result.get("success", False):
            logger.error(
                f"Advanced dashboard error: {advanced_result.get('error', 'Unknown')}"
            )
        if comprehensive_result and not comprehensive_result.get("success", False):
            logger.error(
                f"Comprehensive dashboard error: {comprehensive_result.get('error', 'Unknown')}"
            )

        sys.exit(1)


if __name__ == "__main__":
    main()
