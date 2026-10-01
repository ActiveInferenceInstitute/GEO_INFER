#!/usr/bin/env python3
"""
Place Analyzer Module

This module provides comprehensive analysis capabilities for specific places,
integrating spatial processing, data integration, and visualization.
"""

import logging
import json
import pandas as pd
import geopandas as gpd
from typing import Any
from datetime import datetime
from pathlib import Path
from shapely.geometry import shape
from geo_infer_space.core.spatial_processor import SpatialProcessor

logger = logging.getLogger(__name__)


class WorkspacePlaceAnalyzer:
    """
    Comprehensive analyzer for place-based geospatial intelligence.
    """

    def __init__(
        self,
        place_name: str,
        base_dir: Path,
        processor: SpatialProcessor | None = None,
    ) -> None:
        """
        Initialize the WorkspacePlaceAnalyzer.

        Args:
            place_name: Name of the place/region being analyzed
            base_dir: Base directory for data and outputs
            processor: Optional SpatialProcessor instance
        """
        self.place_name = place_name
        self.base_dir = base_dir
        self.processor = processor or SpatialProcessor()
        self.analysis_results: dict[str, Any] = {}
        self.integrated_data: gpd.GeoDataFrame = gpd.GeoDataFrame()

        # Setup directories
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "outputs"
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized WorkspacePlaceAnalyzer for {place_name}")

    def load_place_data(self, data_sources: list[dict[str, str]]) -> None:
        """
        Load and integrate data from multiple sources.

        Args:
            data_sources: List of dictionaries with 'name' and 'path'
        """
        dataframes = []
        target_crs = None

        for source in data_sources:
            try:
                if source["path"].endswith(".geojson"):
                    with open(source["path"]) as f:
                        geojson = json.load(f)
                    features = []
                    for feat in geojson["features"]:
                        geom = shape(feat["geometry"])
                        props = feat["properties"]
                        features.append({"geometry": geom, **props})
                    df = gpd.GeoDataFrame(features, crs="EPSG:4326")
                elif source["path"].endswith(".csv"):
                    df = pd.read_csv(source["path"])
                    # Convert to GeoDataFrame if geometry column exists
                    if "geometry" in df.columns:
                        df = gpd.GeoDataFrame(
                            df, geometry=gpd.GeoSeries.from_wkt(df["geometry"])
                        )
                elif source["path"].endswith(".shp"):
                    df = gpd.read_file(source["path"])
                else:
                    logger.warning(f"Unsupported file type for {source['name']}")
                    continue

                # Set target CRS from first valid source
                if target_crs is None and df.crs is not None:
                    target_crs = df.crs

                if df.crs is None and target_crs is not None and "geometry" in df:
                    df = df.set_crs(target_crs)

                # Harmonize CRS if needed
                if (
                    df.crs is not None
                    and target_crs is not None
                    and df.crs != target_crs
                ):
                    df = df.to_crs(target_crs)
                    logger.info(
                        f"Transformed {source['name']} from {df.crs} to {target_crs}"
                    )

                df["source"] = source["name"]
                dataframes.append(df)
                logger.info(
                    f"Loaded data from {source['name']} with {len(df)} features"
                )
            except Exception as e:
                logger.error(f"Failed to load {source['name']}: {e}")

        if dataframes:
            self.integrated_data = gpd.GeoDataFrame(
                pd.concat(dataframes, ignore_index=True), crs=target_crs
            )
            logger.info(f"Integrated {len(self.integrated_data)} total features")

    def perform_spatial_analysis(self, analysis_types: list[str]) -> None:
        """
        Perform specified spatial analyses on integrated data.

        Args:
            analysis_types: List of analysis types to perform
        """
        if self.integrated_data.empty:
            logger.warning("No integrated data available for analysis")
            return

        for analysis in analysis_types:
            try:
                if analysis == "buffer":
                    self.analysis_results["buffer"] = self.processor.buffer_analysis(
                        self.integrated_data, buffer_distance=1000
                    )
                elif analysis == "proximity":
                    self.analysis_results["proximity"] = (
                        self.processor.proximity_analysis(
                            self.integrated_data, self.integrated_data
                        )
                    )
                elif analysis == "overlay":
                    layers = self.integrated_data["source"].unique()
                    if len(layers) >= 2:
                        self.analysis_results["overlay"] = (
                            self.processor.perform_multi_overlay(
                                {
                                    layer: self.integrated_data[
                                        self.integrated_data["source"] == layer
                                    ]
                                    for layer in layers
                                }
                            )
                        )
                elif analysis == "correlation":
                    self.analysis_results["correlation"] = (
                        self.processor.calculate_spatial_correlation(
                            self.integrated_data
                        )
                    )
                logger.info(f"Completed {analysis} analysis")
            except Exception as e:
                logger.error(f"Failed {analysis} analysis: {e}")

    def generate_report(self, output_format: str = "json") -> str:
        """
        Generate analysis report in specified format.

        Args:
            output_format: 'json' or 'html'

        Returns:
            Path to generated report
        """
        report: dict[str, Any] = {
            "place": self.place_name,
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_features": len(self.integrated_data),
                "sources": (
                    list(self.integrated_data["source"].unique())
                    if not self.integrated_data.empty
                    else []
                ),
            },
            "analysis_results": self.analysis_results,
        }

        output_path = self.output_dir / f"{self.place_name}_report.{output_format}"
        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        elif output_format == "html":
            # Simple HTML report
            html = f"<h1>{self.place_name} Analysis Report</h1>"
            html += f"<p>Generated: {report['timestamp']}</p>"
            html += "<h2>Data Summary</h2>"
            html += f"<p>Total Features: {report['data_summary']['total_features']}</p>"
            html += f"<p>Sources: {', '.join(report['data_summary']['sources'])}</p>"
            html += "<h2>Analysis Results</h2>"
            for key, value in report["analysis_results"].items():
                html += f"<h3>{key.capitalize()}</h3>"
                html += f"<pre>{json.dumps(value, indent=2)}</pre>"
            with open(output_path, "w") as f:
                f.write(html)

        logger.info(f"Generated report at {output_path}")
        return str(output_path)

    def run_full_analysis(
        self, data_sources: list[dict[str, str]], analysis_types: list[str]
    ) -> dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Args:
            data_sources: List of data sources
            analysis_types: List of analyses to perform

        Returns:
            Dictionary with analysis results and report path
        """
        self.load_place_data(data_sources)
        self.perform_spatial_analysis(analysis_types)
        report_path = self.generate_report()

        return {"results": self.analysis_results, "report_path": report_path}
