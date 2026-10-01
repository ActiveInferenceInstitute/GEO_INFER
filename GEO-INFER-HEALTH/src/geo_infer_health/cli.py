#!/usr/bin/env python3
"""
Command Line Interface for GEO-INFER-HEALTH module.

Provides command-line tools for running health analysis, data processing,
and API server management.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger

from geo_infer_health.api import router
from geo_infer_health.core import (
    DiseaseHotspotAnalyzer,
    EnvironmentalHealthAnalyzer,
)
from geo_infer_health.models import EnvironmentalData, Location, PopulationData
from geo_infer_health.utils.config import load_config
from geo_infer_health.utils.logging import setup_logging


def setup_cli() -> argparse.ArgumentParser:
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="GEO-INFER-HEALTH: Geospatial Health Analytics Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the API server
  geo-infer-health serve --host 0.0.0.0 --port 8000

  # Run disease hotspot analysis
  geo-infer-health analyze hotspots --input data/disease_reports.geojson

  # Run healthcare accessibility analysis
  geo-infer-health analyze accessibility --facilities hospitals.geojson --population census.geojson

  # Process environmental health data
  geo-infer-health analyze environment --air-quality pm25.tif --population census.geojson
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/health_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run health analysis")
    analyze_subparsers = analyze_parser.add_subparsers(
        dest="analysis_type", help="Analysis type"
    )

    # Disease hotspots
    hotspots_parser = analyze_subparsers.add_parser(
        "hotspots", help="Disease hotspot analysis"
    )
    hotspots_parser.add_argument(
        "--input", required=True, help="Input disease reports file"
    )
    hotspots_parser.add_argument("--population", help="Population data file")
    hotspots_parser.add_argument(
        "--output", default="hotspots.geojson", help="Output file"
    )
    hotspots_parser.add_argument(
        "--threshold", type=int, default=5, help="Case threshold for hotspots"
    )
    hotspots_parser.add_argument(
        "--radius", type=float, default=1.0, help="Analysis radius in km"
    )

    # Healthcare accessibility
    accessibility_parser = analyze_subparsers.add_parser(
        "accessibility", help="Healthcare accessibility analysis"
    )
    accessibility_parser.add_argument(
        "--facilities", required=True, help="Healthcare facilities file"
    )
    accessibility_parser.add_argument(
        "--population", required=True, help="Population data file"
    )
    accessibility_parser.add_argument(
        "--output", default="accessibility.geojson", help="Output file"
    )
    accessibility_parser.add_argument(
        "--method",
        choices=["distance", "gravity", "2sfca"],
        default="distance",
        help="Accessibility method",
    )

    # Environmental health
    environment_parser = analyze_subparsers.add_parser(
        "environment", help="Environmental health analysis"
    )
    environment_parser.add_argument("--air-quality", help="Air quality data file")
    environment_parser.add_argument("--water-quality", help="Water quality data file")
    environment_parser.add_argument(
        "--radius", type=float, default=10.0, help="Exposure search radius in km"
    )
    environment_parser.add_argument(
        "--time-window-days", type=int, default=30, help="Time window in days"
    )
    environment_parser.add_argument(
        "--population", required=True, help="Population data file"
    )
    environment_parser.add_argument(
        "--output", default="env_health.geojson", help="Output file"
    )

    # Batch processing
    batch_parser = subparsers.add_parser(
        "batch", help="Batch processing of multiple files"
    )
    batch_parser.add_argument(
        "--config", required=True, help="Batch processing configuration file"
    )
    batch_parser.add_argument("--output-dir", default="output", help="Output directory")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data files")
    validate_parser.add_argument(
        "--input", required=True, help="Input file to validate"
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = setup_cli()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(level=args.log_level, verbose=args.verbose)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Execute command
        if args.command == "serve":
            run_server(args, config)
        elif args.command == "analyze":
            run_analysis(args, config)
        elif args.command == "batch":
            run_batch_processing(args, config)
        elif args.command == "validate":
            run_validation(args, config)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_server(args: argparse.Namespace, config: Any) -> None:
    """Run the API server."""
    logger.info(f"Starting GEO-INFER-HEALTH API server on {args.host}:{args.port}")

    # Import FastAPI app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="GEO-INFER-HEALTH API",
        description="Spatial Health Analytics and Epidemiological Intelligence",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "GEO-INFER-HEALTH API", "version": "1.0.0"}

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


def run_analysis(args: argparse.Namespace, config: Any) -> None:
    """Run health analysis."""
    logger.info(f"Running {args.analysis_type} analysis")

    if args.analysis_type == "hotspots":
        run_hotspot_analysis(args, config)
    elif args.analysis_type == "accessibility":
        run_accessibility_analysis(args, config)
    elif args.analysis_type == "environment":
        run_environment_analysis(args, config)
    else:
        logger.error(f"Unknown analysis type: {args.analysis_type}")


def run_hotspot_analysis(args: argparse.Namespace, config: Any) -> None:
    """Run disease hotspot analysis."""
    # Load disease reports
    import geopandas as gpd

    reports_gdf = gpd.read_file(args.input)
    logger.info(f"Loaded {len(reports_gdf)} disease reports")

    # Convert to internal format
    from geo_infer_health.models import DiseaseReport, Location, PopulationData

    # Location stores EPSG:4326 coordinates; reproject any other CRS
    # before extracting point coordinates or centroids.
    if reports_gdf.crs is not None and reports_gdf.crs.to_epsg() != 4326:
        reports_gdf = reports_gdf.to_crs("EPSG:4326")

    reports: list[DiseaseReport] = []
    for _, row in reports_gdf.iterrows():
        geometry = row.geometry
        if geometry is None:
            raise ValueError(
                "geo_infer_health.cli.run_hotspot_analysis: disease report "
                f"{len(reports)} has no geometry; a point or areal geometry "
                "is required to derive a Location"
            )
        if geometry.geom_type != "Point":
            # Areal (polygon/multipolygon) inputs are collapsed to their
            # centroid so hotspot analysis still operates on a point.
            geometry = geometry.centroid
        report_date = row.get("report_date")
        if report_date is None:
            raise ValueError(
                "geo_infer_health.cli.run_hotspot_analysis: input is missing "
                "the required 'report_date' column (or a row has no value); "
                "every disease report needs a report date for hotspot "
                "analysis"
            )
        location = Location(latitude=geometry.y, longitude=geometry.x)
        report = DiseaseReport(
            report_id=str(row.get("report_id", f"report_{len(reports)}")),
            disease_code=row.get("disease_code", "UNKNOWN"),
            location=location,
            report_date=report_date,
            case_count=int(row.get("case_count", 1)),
        )
        reports.append(report)

    # Load population data if provided
    population_data: list[PopulationData] | None = None
    if args.population:
        pop_gdf = gpd.read_file(args.population)
        from geo_infer_health.models import PopulationData

        population_data = []
        for _, row in pop_gdf.iterrows():
            pop_data = PopulationData(
                area_id=str(row.get("area_id", f"area_{len(population_data)}")),
                population_count=int(row.get("population", 0)),
            )
            population_data.append(pop_data)
        logger.info(f"Loaded {len(population_data)} population areas")

    # Run analysis
    analyzer = DiseaseHotspotAnalyzer(reports=reports, population_data=population_data)
    hotspots = analyzer.identify_simple_hotspots(
        threshold_case_count=args.threshold, scan_radius_km=args.radius
    )

    # Save results
    import json

    with open(args.output, "w") as f:
        json.dump(hotspots, f, indent=2)

    logger.info(f"Found {len(hotspots)} hotspots, saved to {args.output}")


def run_accessibility_analysis(args: argparse.Namespace, config: Any) -> None:
    """Run healthcare accessibility analysis."""
    # Load facilities and population data
    import geopandas as gpd

    facilities_gdf = gpd.read_file(args.facilities)
    population_gdf = gpd.read_file(args.population)

    logger.info(f"Loaded {len(facilities_gdf)} facilities")
    logger.info(f"Loaded {len(population_gdf)} population areas")

    # Convert to internal format
    from geo_infer_health.models import HealthFacility, PopulationData, Location

    facilities: list[HealthFacility] = []
    for _, row in facilities_gdf.iterrows():
        location = Location(latitude=row.geometry.y, longitude=row.geometry.x)
        facility = HealthFacility(
            facility_id=str(row.get("facility_id", f"facility_{len(facilities)}")),
            name=row.get("name", "Unknown"),
            facility_type=row.get("facility_type", "Unknown"),
            location=location,
            capacity=int(row.get("capacity", 0)) if row.get("capacity") else None,
        )
        facilities.append(facility)

    population_data: list[PopulationData] = []
    for _, row in population_gdf.iterrows():
        pop_data = PopulationData(
            area_id=str(row.get("area_id", f"area_{len(population_data)}")),
            population_count=int(row.get("population", 0)),
        )
        population_data.append(pop_data)

    # Summary output reports basic statistics
    total_facilities = len(facilities)
    total_population = sum(p.population_count for p in population_data)
    ratio = total_facilities / total_population * 1000 if total_population > 0 else 0

    results = {
        "total_facilities": total_facilities,
        "total_population": total_population,
        "facility_ratio_per_1000": ratio,
        "method": args.method,
    }

    # Save results
    import json

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Accessibility analysis completed, saved to {args.output}")


def _load_environmental_readings(
    file_path: str, default_parameter: str, default_unit: str
) -> list[EnvironmentalData]:
    """Load environmental readings from a CSV file.

    Expected columns: ``latitude``, ``longitude``, ``value`` and optionally
    ``parameter_name``, ``unit``, ``timestamp`` (ISO 8601), ``data_id``.
    """
    readings: list[EnvironmentalData] = []
    with open(file_path, newline="", encoding="utf-8") as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            timestamp_raw = row.get("timestamp")
            timestamp = (
                datetime.fromisoformat(str(timestamp_raw))
                if timestamp_raw
                else datetime.now(UTC)
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            readings.append(
                EnvironmentalData(
                    data_id=row.get("data_id", f"{Path(file_path).stem}_{index}"),
                    parameter_name=row.get("parameter_name", default_parameter),
                    value=float(row["value"]),
                    unit=row.get("unit", default_unit),
                    location=Location(
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                    ),
                    timestamp=timestamp,
                )
            )
    return readings


def run_environment_analysis(args: argparse.Namespace, config: Any) -> None:
    """Run environmental health analysis."""
    if not args.air_quality and not args.water_quality:
        raise ValueError(
            "Environmental analysis requires --air-quality and/or --water-quality"
        )
    import geopandas as gpd

    population_gdf = gpd.read_file(args.population)
    logger.info(f"Loaded {len(population_gdf)} population areas")

    # Population area centroids become the exposure target locations.
    target_locations: list[Location] = []
    population_data: list[PopulationData] = []
    for index, (_, row) in enumerate(population_gdf.iterrows()):
        centroid = row.geometry.centroid
        target_locations.append(Location(latitude=centroid.y, longitude=centroid.x))
        population_data.append(
            PopulationData(
                area_id=str(row.get("area_id", f"area_{index}")),
                population_count=int(row.get("population", 0)),
            )
        )

    results: dict[str, Any] = {
        "analysis_type": "environmental_health",
        "population_areas": len(population_data),
        "total_population": sum(p.population_count for p in population_data),
    }

    exposure_inputs = [
        ("air_quality", getattr(args, "air_quality", None), "air_quality", "AQI"),
        (
            "water_quality",
            getattr(args, "water_quality", None),
            "water_quality",
            "index",
        ),
    ]
    radius_km = float(getattr(args, "radius", 10.0))
    time_window_days = int(getattr(args, "time_window_days", 30))

    for key, path, default_parameter, default_unit in exposure_inputs:
        if not path:
            continue
        readings = _load_environmental_readings(path, default_parameter, default_unit)
        results[f"{key}_file"] = path
        results[f"{key}_readings"] = len(readings)
        if readings:
            analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)
            results[f"{key}_average_exposure"] = analyzer.calculate_average_exposure(
                target_locations=target_locations,
                radius_km=radius_km,
                parameter_name=default_parameter,
                time_window_days=time_window_days,
            )
        else:
            results[f"{key}_average_exposure"] = {}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Environmental health analysis completed, saved to {args.output}")


def run_batch_processing(args: argparse.Namespace, config: Any) -> list[dict[str, Any]]:
    """Run batch processing of multiple files."""
    logger.info(f"Running batch processing with config: {args.config}")
    jobs = config.get("jobs", config.get("batch", {}).get("jobs", []))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch configuration must contain a non-empty jobs list")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Batch job {index} must be a mapping")
        job_type = job.get("analysis_type", job.get("type"))
        job_args = argparse.Namespace(**job)
        if job_type == "hotspots":
            job_args.output = str(
                output_dir / job.get("output", f"hotspots_{index}.json")
            )
            run_hotspot_analysis(job_args, config)
        elif job_type == "accessibility":
            job_args.output = str(
                output_dir / job.get("output", f"accessibility_{index}.json")
            )
            run_accessibility_analysis(job_args, config)
        elif job_type == "environment":
            job_args.output = str(
                output_dir / job.get("output", f"environment_{index}.json")
            )
            run_environment_analysis(job_args, config)
        else:
            raise ValueError(
                f"Batch job {index} has unsupported analysis_type: {job_type!r}"
            )
        completed.append(
            {
                "index": index,
                "analysis_type": job_type,
                "output": job_args.output,
                "status": "completed",
            }
        )

    manifest = output_dir / "batch_manifest.json"
    manifest.write_text(json.dumps({"jobs": completed}, indent=2), encoding="utf-8")
    logger.info(f"Completed {len(completed)} batch jobs; manifest saved to {manifest}")
    return completed


def run_validation(args: argparse.Namespace, config: Any) -> None:
    """Validate data files.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file contains no features.
        Exception: If the file cannot be read as geospatial data.
    """
    input_path = Path(args.input)
    logger.info(f"Validating file: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    import geopandas as gpd

    gdf = gpd.read_file(input_path)

    # Basic validation
    if gdf.empty:
        raise ValueError(f"File contains no data: {input_path}")

    if gdf.crs is None:
        logger.warning("File has no coordinate reference system")

    logger.info(f"Validation passed for {input_path}")
    logger.info(f"Features: {len(gdf)}, Columns: {list(gdf.columns)}")


if __name__ == "__main__":
    main()
