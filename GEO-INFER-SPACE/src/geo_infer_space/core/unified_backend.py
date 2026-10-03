#!/usr/bin/env python3
"""
Unified H3 Backend for Geospatial Analysis

This module provides a unified interface for integrating multiple data sources
through H3 spatial indexing, enabling general geospatial analysis.
"""

import json
from typing import Any
from datetime import datetime, UTC
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping
import folium
from folium.plugins import HeatMap

import h3

# --- Local Core Imports ---
# Base class for type hinting
from geo_infer_space.core.base_module import BaseAnalysisModule

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and Shapely geometries."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        # Handle Shapely geometry objects
        elif hasattr(obj, "__geo_interface__"):
            return obj.__geo_interface__
        return super().default(obj)


class UnifiedH3Backend:
    """
    Unified H3-indexed backend for comprehensive geospatial analysis.
    """

    def __init__(
        self,
        modules: dict[str, "BaseAnalysisModule"],
        resolution: int = 8,
        target_region: str = "Global",
        target_areas: dict[str, list[str]] | None = None,
        base_data_dir: Path | None = None,
        geojson_path: Path | None = None,
    ) -> None:
        """
        Initialize the unified backend with H3 spatial indexing.

        Args:
            modules: A dictionary of initialized analysis module instances.
            resolution: H3 resolution level (default: 8)
            target_region: Region identifier (default: 'Global')
            target_areas: A dict specifying areas to run, e.g., {'CA': ['all']}.
            base_data_dir: The root directory for data caching.
            geojson_path: Path to the target-areas GeoJSON file. When omitted,
                no area geometries are loaded and ``target_areas`` resolves to
                an empty region.
        """
        self.modules = modules
        self.resolution = resolution
        self.target_region = target_region
        if base_data_dir:
            self.base_data_dir = Path(base_data_dir)
        else:
            self.base_data_dir = Path("./data")
        self.geojson_path: Path | None = (
            Path(geojson_path) if geojson_path is not None else None
        )
        self.unified_data: dict[str, dict] = {}
        self.analysis_scores: dict[str, dict] = {}

        self.target_hexagons_by_area, self.target_hexagons = self._define_target_region(
            target_areas
        )

        logger.info(
            f"UnifiedH3Backend initialized for '{self.target_region}' with {len(self.modules)} active modules at H3 resolution {self.resolution}"
        )
        logger.info(f"Active modules: {list(self.modules.keys())}")
        logger.info(
            f"Defined {len(self.target_hexagons)} total target hexagons across {len(self.target_hexagons_by_area)} areas."
        )

    def _define_target_region(
        self, target_areas: dict[str, list[str]] | None = None
    ) -> tuple[dict[str, list[str]], list[str]]:
        """
        Define the target region based on geometries.

        Args:
            target_areas: Dictionary mapping areas to lists of subareas

        Returns:
            Tuple of (hexagons_by_area, all_hexagons)
        """
        area_geoms = self._get_geometries(target_areas)

        if not area_geoms:
            logger.error(
                "No geometries could be loaded or defined. Cannot define a target region."
            )
            return {}, []

        hexagons_by_area: dict[str, set] = {area: set() for area in area_geoms.keys()}

        for area, geoms in area_geoms.items():
            for geom_name, geom in geoms.items():
                logger.info(f"Generating hexagons for {geom_name}, {area}...")
                try:
                    # Handle Shapely Polygon
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        # Convert to GeoJSON format for H3 v4 API
                        geojson_geom = mapping(geom)

                        # Ensure proper GeoJSON structure for H3 v4
                        if geojson_geom.get("type") == "Polygon" and geojson_geom.get(
                            "coordinates"
                        ):
                            # Ensure coordinates are properly nested for H3 v4
                            if not isinstance(
                                geojson_geom["coordinates"][0][0], (list, tuple)
                            ):
                                geojson_geom["coordinates"] = [
                                    geojson_geom["coordinates"]
                                ]

                        # Use H3 v4 API
                        hexagons_in_area = h3.geo_to_cells(
                            geojson_geom, self.resolution
                        )
                        hexagons_by_area[area].update(hexagons_in_area)
                        logger.info(
                            f"Generated {len(hexagons_in_area)} hexagons for {geom_name}, {area}"
                        )

                    # Handle GeoJSON dict
                    elif isinstance(geom, dict) and geom.get("type") == "Polygon":
                        # Ensure coordinates are properly nested for H3 v4
                        if not isinstance(geom["coordinates"][0][0], (list, tuple)):
                            geom["coordinates"] = [geom["coordinates"]]

                        # Use H3 v4 API
                        hexagons_in_area = h3.geo_to_cells(geom, self.resolution)
                        hexagons_by_area[area].update(hexagons_in_area)
                        logger.info(
                            f"Generated {len(hexagons_in_area)} hexagons for {geom_name}, {area}"
                        )

                    # Handle GeoJSON Feature
                    elif isinstance(geom, dict) and geom.get("type") == "Feature":
                        geometry = geom.get("geometry", {})
                        if geometry and geometry.get("type") in (
                            "Polygon",
                            "MultiPolygon",
                        ):
                            # Ensure coordinates are properly nested for H3 v4
                            if geometry.get("type") == "Polygon" and geometry.get(
                                "coordinates"
                            ):
                                if not isinstance(
                                    geometry["coordinates"][0][0], (list, tuple)
                                ):
                                    geometry["coordinates"] = [geometry["coordinates"]]

                            # Use H3 v4 API
                            hexagons_in_area = h3.geo_to_cells(
                                geometry, self.resolution
                            )
                            hexagons_by_area[area].update(hexagons_in_area)
                            logger.info(
                                f"Generated {len(hexagons_in_area)} hexagons for {geom_name}, {area}"
                            )
                        else:
                            logger.warning(
                                f"Invalid or missing geometry in Feature for {geom_name}, {area}"
                            )

                    else:
                        logger.warning(
                            f"Skipping invalid geometry for {geom_name}, {area}: {type(geom)}"
                        )

                except Exception as e:
                    logger.error(f"H3 geo_to_cells failed for {geom_name}, {area}: {e}")
                    logger.debug(f"Geometry that failed: {geom}")

                    # Try fallback method with polygon_to_cells
                    try:
                        from geo_infer_space.utils.h3_utils import polygon_to_cells

                        if isinstance(geom, (Polygon, MultiPolygon)):
                            geojson_geom = mapping(geom)
                            hexagons_in_area = polygon_to_cells(
                                geojson_geom, self.resolution
                            )
                        elif isinstance(geom, dict):
                            hexagons_in_area = polygon_to_cells(geom, self.resolution)
                        else:
                            raise ValueError(f"Unsupported geometry type: {type(geom)}")

                        hexagons_by_area[area].update(hexagons_in_area)
                        logger.info(
                            f"Generated {len(hexagons_in_area)} hexagons using fallback for {geom_name}, {area}"
                        )
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback also failed for {geom_name}, {area}: {fallback_error}"
                        )

        final_hex_by_area = {
            k: sorted(list(v)) for k, v in hexagons_by_area.items() if v
        }
        final_all_hexagons = sorted(
            list(set.union(*[set(v) for v in hexagons_by_area.values()]))
        )

        if not final_all_hexagons:
            logger.error(
                f"Failed to generate any H3 hexagons for region '{self.target_region}' with filters {target_areas}"
            )
            return {}, []

        return final_hex_by_area, final_all_hexagons

    def _get_geometries(
        self, target_areas: dict[str, list[str]] | None
    ) -> dict[str, dict[str, Any]]:
        """
        Loads geometries for the specified areas from a GeoJSON file.
        """
        if not target_areas:
            return {}

        geojson_path = self.geojson_path
        if geojson_path is None:
            logger.warning(
                "No geojson_path configured; cannot resolve target areas %s",
                sorted(target_areas),
            )
            return {}
        if not geojson_path.exists():
            logger.error(f"GeoJSON file not found: {geojson_path}")
            return {}

        try:
            gdf = gpd.read_file(geojson_path)
            output_geoms = {}
            for area, subareas in target_areas.items():
                area_gdf = gdf[gdf["area"] == area]
                if area_gdf.empty:
                    continue
                geom_dict = {}
                for subarea in subareas:
                    if subarea == "all":
                        geom = area_gdf.geometry.union_all()
                        geom_dict["all"] = geom
                    else:
                        sub_gdf = area_gdf[area_gdf["subarea"] == subarea]
                        if not sub_gdf.empty:
                            geom_dict[subarea] = sub_gdf.geometry.union_all()
                if geom_dict:
                    output_geoms[area] = geom_dict
            return output_geoms
        except Exception as e:
            logger.error(f"Failed to load geometries: {e}")
            return {}

    def run_comprehensive_analysis(self) -> None:
        """
        Execute the full analysis pipeline for all active modules.
        """
        logger.info("Starting comprehensive analysis...")
        module_results = {}

        for name, module in self.modules.items():
            logger.info(f"--- Processing Module: {name.upper()} ---")
            try:
                result = module.run_analysis()
                module_results[name] = result
                logger.info(f"Successfully processed module: {name.upper()}")
            except Exception as e:
                logger.error(
                    f"Failed to process module {name.upper()}: {e}", exc_info=True
                )
                module_results[name] = {}

        self._aggregate_module_results(module_results)
        self.calculate_analysis_scores()
        logger.info(
            "Comprehensive analysis complete. All module data has been aggregated."
        )

    def _aggregate_module_results(self, results: dict[str, dict]) -> None:
        """Combine all module results into a unified H3-indexed dataset."""
        logger.info("Aggregating results from all modules...")

        for hexagon in self.target_hexagons:
            hex_data: dict[str, Any] = {"hex_id": hexagon}

            # Add geometry and metadata
            try:
                # Use the correct h3-py v4.x API
                lat, lng = h3.cell_to_latlng(hexagon)
                hex_data["centroid"] = [lat, lng]
                hex_data["boundary"] = h3.cell_to_boundary(hexagon)
            except Exception as e:
                logger.warning(f"Could not process geometry for {hexagon}: {e}")
                hex_data["centroid"] = None
                hex_data["boundary"] = None

            # Add module data
            for module_name, module_data in results.items():
                hex_data[module_name] = module_data.get(hexagon, {})

            self.unified_data[hexagon] = hex_data

        logger.info(
            f"Aggregated data for {len(self.target_hexagons)} hexagons from {len(results)} modules."
        )

    def calculate_analysis_scores(self) -> dict[str, dict]:
        """
        Calculate analysis scores based on the unified dataset.

        Returns:
            Dictionary of scores for each hexagon.
        """
        logger.info("Calculating analysis scores...")
        if not self.unified_data:
            logger.warning("Unified data is not available. Cannot calculate scores.")
            return {}

        for h3_index, hex_data in self.unified_data.items():
            scores: dict[str, Any] = {}
            module_scores = []
            for module_data in hex_data.values():
                if isinstance(module_data, dict) and "score" in module_data:
                    module_scores.append(module_data["score"])
            if module_scores:
                composite_score = np.mean(module_scores)
            else:
                composite_score = 0.0  # type: ignore[assignment]

            self.analysis_scores[h3_index] = {
                "composite_score": composite_score,
                "factors": scores,
            }

        logger.info(f"Calculated scores for {len(self.analysis_scores)} hexagons.")
        return self.analysis_scores

    def get_comprehensive_summary(self) -> dict[str, Any]:
        """
        Generate a comprehensive summary of the analysis results.

        Returns:
            Dictionary containing the analysis summary.
        """
        if not self.unified_data:
            return {"error": "Analysis has not been run."}

        scores = [s["composite_score"] for s in self.analysis_scores.values()]

        summary: dict[str, Any] = {
            "target_region": self.target_region,
            "h3_resolution": self.resolution,
            "total_hexagons": len(self.target_hexagons),
            "modules_analyzed": list(self.modules.keys()),
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "analysis_potential": {
                "mean_score": round(np.mean(scores), 3) if scores else 0,
                "median_score": round(np.median(scores), 3) if scores else 0,
                "std_dev": round(np.std(scores), 3) if scores else 0,
                "high_potential_hexagons": len([s for s in scores if s > 0.75]),
                "low_potential_hexagons": len([s for s in scores if s < 0.25]),
            },
            "module_summaries": {},
        }

        for module_name in self.modules.keys():
            valid_hex_count = sum(
                1
                for hex_data in self.unified_data.values()
                if hex_data.get(module_name)
            )
            summary["module_summaries"][module_name] = {
                "processed_hexagons": valid_hex_count,
                "coverage": (
                    round(valid_hex_count / len(self.target_hexagons) * 100, 2)
                    if self.target_hexagons
                    else 0
                ),
            }

        return summary

    def export_unified_data(
        self, output_path: str, export_format: str = "geojson"
    ) -> None:
        """
        Export unified data and scores to a specified format.

        Args:
            output_path: Output file path.
            export_format: 'geojson', 'csv', or 'json'.
        """
        if not self.unified_data:
            raise ValueError(
                "No unified data to export. Please run the analysis first."
            )

        # Combine unified data with analysis scores
        export_data = {}
        for h3_index, data in self.unified_data.items():
            export_data[h3_index] = data.copy()
            export_data[h3_index]["analysis_potential"] = self.analysis_scores.get(
                h3_index, {}
            )

        if export_format == "geojson":
            self._export_geojson(export_data, output_path)
        elif export_format == "csv":
            self._export_csv(export_data, output_path)
        elif export_format == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        else:
            logger.error(f"Unsupported export format: {export_format}")
            return
        logger.info(f"Successfully exported unified data to {output_path}")

    def _export_geojson(self, data_to_export: dict, output_path: str) -> None:
        """Exports the unified dataset to a GeoJSON file."""
        features = []
        for hex_id, properties in data_to_export.items():
            # Get geometry for the hexagon
            boundary = h3.cell_to_boundary(hex_id)
            ring = [[lng, lat] for lat, lng in boundary]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": properties,
                }
            )

        feature_collection = {"type": "FeatureCollection", "features": features}

        with open(output_path, "w") as f:
            json.dump(feature_collection, f, cls=NumpyEncoder)

    def _export_csv(self, data_to_export: dict, output_path: str) -> None:
        """Exports the unified dataset to a CSV file."""
        # This will flatten the nested dictionary structure
        flat_data = []
        for hex_id, props in data_to_export.items():
            row = {"h3_index": hex_id}
            for key, value in props.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False)

    def generate_interactive_dashboard(self, output_path: str) -> None:
        """
        Generate an interactive Folium dashboard with multiple data overlays.

        Args:
            output_path: Path to save the HTML dashboard file.
        """
        if not self.unified_data:
            logger.error("No unified data available to generate a dashboard.")
            return

        # Calculate map center
        if not self.target_hexagons:
            map_center = [0, 0]  # Default
        else:
            centroids = [h3.cell_to_latlng(h) for h in self.target_hexagons]
            lats, lons = zip(*centroids)
            map_center = [np.mean(lats), np.mean(lons)]

        logger.info(f"Generating interactive dashboard centered at {map_center}...")
        m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

        # Add a title to the map
        title_html = """
            <h3 style="text-align: center; color: #333; padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-family: 'Arial', sans-serif;">
                Unified Geospatial Analysis Dashboard
            </h3>
        """
        m.get_root().header.add_child(  # type: ignore[attr-defined]
            folium.Element(title_html)
        )

        folium.TileLayer("Stamen Terrain", attr="Stamen").add_to(m)

        # --- Create Feature Groups for each layer ---
        groups = {
            "analysis": folium.FeatureGroup(name="Analysis Potential", show=True),
        }

        # --- Populate Layers ---
        for h3_index, hex_data in self.unified_data.items():
            boundary = hex_data.get("boundary")
            if not boundary:
                continue

            # Analysis Layer
            score_data = self.analysis_scores.get(h3_index, {})
            score = score_data.get("composite_score", 0)
            popup_html = f"<b>H3:</b> {h3_index}<br><b>Score:</b> {score:.3f}"
            folium.Polygon(
                locations=boundary,
                color=self._get_color_for_score(score),
                fill_color=self._get_color_for_score(score),
                weight=1,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html),
                tooltip=f"Analysis Score: {score:.3f}",
            ).add_to(groups["analysis"])

        # --- Add layers to map ---
        for group in groups.values():
            group.add_to(m)

        # --- Add Heatmap ---
        heat_data = [
            [
                self.unified_data[h]["centroid"][0],
                self.unified_data[h]["centroid"][1],
                self.analysis_scores.get(h, {}).get("composite_score", 0),
            ]
            for h in self.target_hexagons
            if h in self.unified_data
            and "centroid" in self.unified_data[h]
            and self.unified_data[h]["centroid"]
        ]

        if heat_data:
            HeatMap(heat_data, name="Analysis Heatmap", show=False).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        try:
            m.save(output_path)
            logger.info(
                f"Successfully generated interactive dashboard at {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save interactive dashboard to {output_path}: {e}")

    def _get_color_for_score(self, score: float, theme: str = "default") -> str:
        """Helper to get a color based on a score from 0 to 1."""
        if not isinstance(score, (float, int)):
            return "#808080"  # type: ignore[unreachable]

        # Default: Green-Yellow-Red
        if score > 0.75:
            return "#2ca25f"
        if score > 0.5:
            return "#99d8c9"
        if score > 0.25:
            return "#fed976"
        return "#e31a1c"
