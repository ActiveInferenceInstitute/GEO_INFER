#!/usr/bin/env python3
"""
Spatial Microbiome-Climate-Soil Integration Script

Generates seeded synthetic microbiome, climate and soil datasets (following
Earth Microbiome Project, WorldClim and ISRIC SoilGrids value ranges), indexes
them on an H3 v4 grid and writes a multi-layer interactive folium map. Real
dataset loaders live in ``geo_infer_bio.microbiome``, ``geo_infer_bio.climate``
and ``geo_infer_bio.soil`` and require locally downloaded source files.

Usage:
    python run_spatial_integration.py --h3_resolution=7 --output_format="interactive"
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import folium
import h3
import numpy as np
from folium.plugins import MarkerCluster

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spatial_microbiome_integration")


class SpatialMicrobiomeIntegrator:
    """
    Main integration class for spatial microbiome-climate-soil analysis.

    This class demonstrates minimal orchestration of GEO-INFER modules to create
    sophisticated spatial biological analyses without novel algorithmic development.
    """

    def __init__(self, output_dir: str = "output", h3_resolution: int = 7):
        """
        Initialize the spatial integrator.

        Args:
            output_dir: Directory for output files
            h3_resolution: H3 hexagonal grid resolution (0-15)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h3_resolution = h3_resolution

        logger.info("🚀 SpatialMicrobiomeIntegrator initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔷 H3 resolution: {h3_resolution}")

    def load_biological_datasets(
        self,
        region_bbox: tuple[float, float, float, float] = (-130, 25, -65, 55),
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """
        Generate seeded synthetic microbiome, climate and soil datasets.

        The datasets follow Earth Microbiome Project, WorldClim and SoilGrids
        value ranges; no external data are downloaded.

        Args:
            region_bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            max_samples: Maximum number of microbiome samples

        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("=== Generating Synthetic Biological Datasets ===")

        np.random.seed(42)  # For reproducible results
        min_lon, min_lat, max_lon, max_lat = region_bbox

        # Create clustered coordinates to demonstrate H3 clustering
        cluster_centers = [
            (40.7589, -73.9851),  # New York area
            (34.0522, -118.2437),  # Los Angeles area
            (41.8781, -87.6298),  # Chicago area
            (29.7604, -95.3698),  # Houston area
            (47.6062, -122.3321),  # Seattle area
            (25.7617, -80.1918),  # Miami area
            (39.7392, -104.9903),  # Denver area
            (33.4484, -112.0740),  # Phoenix area
        ]

        coordinates = []
        cluster_assignments = []

        # Generate clustered data points
        samples_per_cluster = max_samples // len(cluster_centers)

        for i, (center_lat, center_lon) in enumerate(cluster_centers):
            # Create dense clusters around each center
            for _ in range(samples_per_cluster):
                # Add random offset within ~50km radius
                lat_offset = np.random.normal(0, 0.5)  # ~50km at mid-latitudes
                lon_offset = np.random.normal(0, 0.5)

                new_lat = center_lat + lat_offset
                new_lon = center_lon + lon_offset

                # Keep within bounds
                new_lat = max(min_lat, min(max_lat, new_lat))
                new_lon = max(min_lon, min(max_lon, new_lon))

                coordinates.append((new_lat, new_lon))
                cluster_assignments.append(i)

        # Add some scattered points between clusters
        remaining_samples = max_samples - len(coordinates)
        for _ in range(remaining_samples):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            coordinates.append((lat, lon))
            cluster_assignments.append(-1)  # No cluster

        # Create demo microbiome data with cluster-specific characteristics
        demo_microbiome = {
            "coordinates": coordinates,
            "cluster_assignments": cluster_assignments,
            "diversity_metrics": {},
            "data_source": "Demo Earth Microbiome Project (Clustered)",
        }

        # Generate diversity metrics with cluster patterns
        for i, cluster_id in enumerate(cluster_assignments):
            # Different diversity patterns for each cluster
            if cluster_id == 0:  # NYC - Urban, lower diversity
                base_diversity = 1.2
                base_species = 120
            elif cluster_id == 1:  # LA - Mediterranean, medium diversity
                base_diversity = 2.1
                base_species = 160
            elif cluster_id == 2:  # Chicago - Continental, variable diversity
                base_diversity = 1.8
                base_species = 140
            elif cluster_id == 3:  # Houston - Subtropical, high diversity
                base_diversity = 2.8
                base_species = 180
            elif cluster_id == 4:  # Seattle - Temperate rainforest, very high diversity
                base_diversity = 3.2
                base_species = 200
            elif cluster_id == 5:  # Miami - Tropical, high diversity
                base_diversity = 3.0
                base_species = 190
            elif cluster_id == 6:  # Denver - Mountain, low diversity
                base_diversity = 1.5
                base_species = 110
            elif cluster_id == 7:  # Phoenix - Desert, very low diversity
                base_diversity = 0.8
                base_species = 90
            else:  # Scattered points
                base_diversity = 2.0
                base_species = 150

            # Add random variation
            diversity = np.random.gamma(base_diversity, 0.3)
            species = int(np.random.poisson(base_species))

            demo_microbiome["diversity_metrics"][f"sample_{i}"] = {
                "shannon_diversity": diversity,
                "observed_species": species,
                "simpson_diversity": np.random.beta(5, 2),
                "cluster_id": cluster_id,
            }

        # Generate climate data using real climate data patterns (WorldClim-style)
        logger.info("🌡️ Generating realistic climate data with real-world patterns...")
        demo_climate = {
            "variables": ["bio1", "bio12", "bio15"],
            "climate_data": [],
            "data_source": "WorldClim-style Climate Data (Realistic Patterns)",
            "coordinates": [],
        }

        # Generate climate stations (more sparse than microbiome samples)
        climate_coordinates = []

        # Add major climate stations near cluster centers
        for center_lat, center_lon in cluster_centers:
            # Add 3-5 climate stations per region
            stations_per_region = np.random.randint(3, 6)
            for _ in range(stations_per_region):
                # Climate stations within ~100km of center
                lat_offset = np.random.normal(0, 1.0)
                lon_offset = np.random.normal(0, 1.0)

                station_lat = center_lat + lat_offset
                station_lon = center_lon + lon_offset

                # Keep within bounds
                station_lat = max(min_lat, min(max_lat, station_lat))
                station_lon = max(min_lon, min(max_lon, station_lon))

                climate_coordinates.append((station_lat, station_lon))

        # Add some additional scattered climate stations
        additional_stations = 20
        for _ in range(additional_stations):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            climate_coordinates.append((lat, lon))

        demo_climate["coordinates"] = climate_coordinates

        # Generate realistic climate data for each station
        for lat, lon in climate_coordinates:
            # Real-world temperature patterns (WorldClim Bio1 - Annual Mean Temperature)
            # Base temperature on latitude (realistic North American pattern)
            if lat > 45:  # Northern regions
                temp_base = 50 + np.random.normal(0, 30)  # 5°C ± 3°C
            elif lat > 35:  # Mid-latitude
                temp_base = 120 + np.random.normal(0, 40)  # 12°C ± 4°C
            elif lat > 25:  # Southern regions
                temp_base = 200 + np.random.normal(0, 30)  # 20°C ± 3°C
            else:  # Tropical
                temp_base = 260 + np.random.normal(0, 20)  # 26°C ± 2°C

            # Adjust for longitude (continental effects)
            if -100 < lon < -80:  # Central plains - more extreme
                temp_base += np.random.normal(0, 20)
            elif lon < -120:  # West coast - maritime moderation
                temp_base += np.random.normal(10, 15)
            elif lon > -80:  # East coast - some maritime influence
                temp_base += np.random.normal(5, 15)

            # Real-world precipitation patterns (WorldClim Bio12 - Annual Precipitation)
            # Base precipitation on geography
            coast_distance_west = abs(lon + 120)  # Distance from west coast
            coast_distance_east = abs(lon + 75)  # Distance from east coast
            min_coast_distance = min(coast_distance_west, coast_distance_east)

            # Precipitation patterns
            if lat > 45 and lon < -120:  # Pacific Northwest - high precipitation
                precip_base = 1500 + np.random.gamma(2, 200)
            elif lat > 40 and -100 < lon < -80:  # Great Plains - low precipitation
                precip_base = 400 + np.random.gamma(2, 100)
            elif lat < 30 and lon > -90:  # Southeast - high precipitation
                precip_base = 1200 + np.random.gamma(2, 150)
            elif lat < 35 and -115 < lon < -105:  # Southwest desert - very low
                precip_base = 200 + np.random.gamma(1, 50)
            else:  # General continental pattern
                precip_base = 800 - min_coast_distance * 15 + np.random.gamma(2, 100)

            precip_base = max(100, precip_base)  # Minimum 100mm

            # Precipitation seasonality (WorldClim Bio15)
            # Higher seasonality inland, lower near coasts
            seasonality_base = 30 + min_coast_distance * 2
            seasonality = np.random.normal(seasonality_base, 15)
            seasonality = max(5, min(100, seasonality))  # Realistic range

            demo_climate["climate_data"].append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "bio1": temp_base,  # Annual mean temperature (°C * 10)
                    "bio12": precip_base,  # Annual precipitation (mm)
                    "bio15": seasonality,  # Precipitation seasonality (CV)
                }
            )

        # Generate soil data with overlapping and non-overlapping areas
        logger.info(
            "🌱 Generating synthetic soil data with realistic spatial patterns..."
        )
        demo_soil = {
            "soil_properties": ["phh2o", "soc", "clay", "sand"],
            "depths": ["0-5cm", "5-15cm"],
            "soil_data": {
                "phh2o_0-5cm": {"coordinates": []},
                "soc_0-5cm": {"coordinates": []},
                "clay_0-5cm": {"coordinates": []},
                "sand_0-5cm": {"coordinates": []},
            },
            "data_source": "ISRIC SoilGrids-style Synthetic Data",
            "coordinates": [],
        }

        soil_coordinates = []

        # Soil sampling strategy:
        # 1. Some samples co-located with microbiome samples (50%)
        # 2. Some samples on a systematic grid (30%)
        # 3. Some samples randomly distributed (20%)

        # Co-located samples with microbiome data
        colocated_indices = np.random.choice(
            len(coordinates), size=len(coordinates) // 2, replace=False
        )
        for idx in colocated_indices:
            lat, lon = coordinates[idx]
            # Add small offset to simulate nearby but not exact sampling
            lat_offset = np.random.normal(0, 0.01)  # ~1km offset
            lon_offset = np.random.normal(0, 0.01)
            soil_coordinates.append((lat + lat_offset, lon + lon_offset))

        # Systematic grid sampling
        grid_points = 60
        lat_steps = np.linspace(min_lat + 1, max_lat - 1, int(np.sqrt(grid_points)))
        lon_steps = np.linspace(min_lon + 1, max_lon - 1, int(np.sqrt(grid_points)))

        for lat in lat_steps:
            for lon in lon_steps:
                # Add some random offset to make it look more natural
                lat_jitter = np.random.normal(0, 0.5)
                lon_jitter = np.random.normal(0, 0.5)
                soil_coordinates.append((lat + lat_jitter, lon + lon_jitter))

        # Random additional samples
        random_samples = 40
        for _ in range(random_samples):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            soil_coordinates.append((lat, lon))

        demo_soil["coordinates"] = soil_coordinates

        # Generate realistic soil properties for each location
        for lat, lon in soil_coordinates:
            # pH patterns based on climate and geology
            # Generally: acidic in high-precipitation areas, alkaline in arid areas

            # Estimate precipitation at this location (simplified)
            coast_distance = min(abs(lon + 70), abs(lon + 120))
            estimated_precip = 1200 - coast_distance * 15
            estimated_precip = max(200, estimated_precip)

            # pH based on precipitation and latitude
            if estimated_precip > 1000:  # High precipitation - more acidic
                ph_base = 5.8
            elif estimated_precip > 600:  # Medium precipitation
                ph_base = 6.5
            else:  # Low precipitation - more alkaline
                ph_base = 7.2

            # Regional adjustments
            if lat > 45:  # Northern areas - slightly more acidic
                ph_base -= 0.3
            elif (
                lat < 30 and lon > -100
            ):  # Southeast - more acidic due to precipitation
                ph_base -= 0.5
            elif lat < 35 and -115 < lon < -105:  # Southwest - more alkaline
                ph_base += 0.8

            ph = np.random.normal(ph_base, 0.6)
            ph = max(4.0, min(9.0, ph))  # Realistic pH range

            # Organic carbon - higher in cooler, wetter areas
            temp_factor = max(0.3, (50 - lat) / 20)  # Higher at northern latitudes
            precip_factor = estimated_precip / 1000
            oc_base = 25 * precip_factor * temp_factor
            oc = np.random.gamma(2, oc_base / 2)
            oc = max(2, min(80, oc))  # Realistic range 2-80 g/kg

            # Clay content - varies by region and geology
            if -100 < lon < -95 and 35 < lat < 45:  # Great Plains - high clay
                clay_base = 45
            elif lon < -120:  # West coast - variable
                clay_base = 25
            elif lat > 45:  # Northern areas - moderate clay
                clay_base = 30
            else:  # General
                clay_base = 20

            clay = np.random.gamma(2, clay_base / 2)
            clay = max(5, min(70, clay))  # Realistic range 5-70%

            # Sand content (inversely related to clay, but not perfectly)
            sand_base = 70 - clay * 0.8
            sand = np.random.gamma(2, sand_base / 2)
            sand = max(10, min(85, sand))  # Realistic range 10-85%

            # Add to soil data
            demo_soil["soil_data"]["phh2o_0-5cm"]["coordinates"].append(
                {"latitude": lat, "longitude": lon, "value": ph}
            )
            demo_soil["soil_data"]["soc_0-5cm"]["coordinates"].append(
                {"latitude": lat, "longitude": lon, "value": oc}
            )
            demo_soil["soil_data"]["clay_0-5cm"]["coordinates"].append(
                {"latitude": lat, "longitude": lon, "value": clay}
            )
            demo_soil["soil_data"]["sand_0-5cm"]["coordinates"].append(
                {"latitude": lat, "longitude": lon, "value": sand}
            )

        # Create unified coordinate structure with all data
        unified_coordinates = []

        # Add microbiome data
        for i, (lat, lon) in enumerate(coordinates):
            cluster_id = cluster_assignments[i] if i < len(cluster_assignments) else -1
            cluster_info = f"Cluster {cluster_id}" if cluster_id >= 0 else "Scattered"
            unified_coordinates.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "data_type": "microbiome",
                    "shannon_diversity": np.random.uniform(1.5, 4.5),
                    "species_richness": np.random.randint(50, 200),
                    "temperature": np.random.normal(150, 50),  # Scaled temperature
                    "ph": np.random.uniform(4.5, 8.5),
                    "cluster_info": cluster_info,
                }
            )

        # Add climate data
        for lat, lon in climate_coordinates:
            # Temperature varies by latitude (realistic patterns)
            temp_base = 25 - (lat - 25) * 0.4  # Decreases with latitude
            temp_base += np.random.normal(0, 3)  # Add variation
            temp_celsius = max(5, min(26, temp_base))

            # Precipitation varies by region
            if lat > 45:  # Northern regions
                precip_base = 600
            elif -100 < lon < -90 and 30 < lat < 40:  # Southeast
                precip_base = 1200
            elif lon < -115:  # West coast
                precip_base = 400
            else:  # General
                precip_base = 800

            precipitation = max(200, min(1500, np.random.normal(precip_base, 200)))
            seasonality = np.random.uniform(15, 95)

            unified_coordinates.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "data_type": "climate",
                    "temperature": temp_celsius * 10,  # Match the expected scale
                    "precipitation": int(precipitation),
                    "precipitation_seasonality": seasonality,
                }
            )

        # Add soil data
        for lat, lon in soil_coordinates:
            # pH varies by region (realistic patterns)
            ph_base = 6.5
            if lat > 45:  # Northern areas - slightly more acidic
                ph_base -= 0.3
            elif lat < 30 and lon > -100:  # Southeast - more acidic
                ph_base -= 0.5
            elif lat < 35 and -115 < lon < -105:  # Southwest - more alkaline
                ph_base += 0.8

            ph = max(4.0, min(9.0, np.random.normal(ph_base, 0.6)))
            organic_carbon = max(2, min(80, np.random.gamma(2, 12)))
            bulk_density = np.random.uniform(1.0, 1.8)

            unified_coordinates.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "data_type": "soil",
                    "ph": ph,
                    "organic_carbon": organic_carbon,
                    "bulk_density": bulk_density,
                }
            )

        datasets = {
            "microbiome": demo_microbiome,
            "climate": demo_climate,
            "soil": demo_soil,
            "coordinates": unified_coordinates,
            "cluster_assignments": cluster_assignments,
            "region_bbox": region_bbox,
        }

        logger.info(f"✅ Generated clustered demo datasets with {max_samples} samples")
        logger.info(f"🏙️ Created {len(cluster_centers)} geographic clusters")
        logger.info(f"🌡️ Generated {len(climate_coordinates)} climate stations")
        logger.info(f"🌱 Generated {len(soil_coordinates)} soil sampling sites")
        return datasets

    def create_interactive_h3_visualization(
        self,
        biological_data: dict[str, Any],
        map_center: tuple[float, float] = (40.0, -97.5),
        output_format: str = "interactive",
    ) -> str:
        """
        Create interactive H3 visualization with clustering and biological overlays
        """
        logger.info("=== Creating Interactive H3 Visualization ===")
        logger.info(f"🗺️ Creating interactive map centered at {map_center}")

        # Create base map with professional styling
        m = folium.Map(
            location=map_center,
            zoom_start=5,
            tiles="CartoDB positron",
            attr="© CartoDB, © OpenStreetMap contributors",
        )

        logger.info("🎛️ Creating layer groups for toggle control...")

        # Create properly named feature groups for layer control
        h3_group = folium.FeatureGroup(name="🔷 H3 Spatial Grid", show=True)
        microbiome_group = folium.FeatureGroup(name="🧬 Microbiome Data", show=True)
        climate_group = folium.FeatureGroup(name="🌡️ Climate Stations", show=True)
        soil_group = folium.FeatureGroup(name="🌱 Soil Properties", show=True)

        # Add enhanced H3 hexagonal overlay with mouseover highlighting
        logger.info("🔷 Adding enhanced H3 hexagonal overlay...")
        h3_sample_mapping = {}

        for coord in biological_data["coordinates"]:
            h3_index = h3.latlng_to_cell(coord["lat"], coord["lon"], self.h3_resolution)
            if h3_index not in h3_sample_mapping:
                h3_sample_mapping[h3_index] = []
            h3_sample_mapping[h3_index].append(coord)

        # Create H3 visualization with proper mouseover highlighting
        for h3_index, samples in h3_sample_mapping.items():
            try:
                boundary = h3.cell_to_boundary(h3_index)
                coords = [[lat, lon] for lat, lon in boundary]

                sample_count = len(samples)
                avg_diversity = (
                    sum(s["shannon_diversity"] for s in samples) / sample_count
                )
                avg_temp = sum(s["temperature"] for s in samples) / sample_count

                # Enhanced color coding based on sample density
                if sample_count >= 3:
                    color, fill_color, opacity = (
                        "#d73027",
                        "#d73027",
                        0.8,
                    )  # High density - red
                elif sample_count == 2:
                    color, fill_color, opacity = (
                        "#fc8d59",
                        "#fc8d59",
                        0.7,
                    )  # Medium - orange
                else:
                    color, fill_color, opacity = (
                        "#4575b4",
                        "#4575b4",
                        0.6,
                    )  # Low density - blue

                # Detailed cluster information
                cluster_info = []
                for city in [
                    "New York",
                    "Los Angeles",
                    "Chicago",
                    "Houston",
                    "Seattle",
                    "Miami",
                    "Denver",
                    "Phoenix",
                ]:
                    city_count = sum(
                        1
                        for s in samples
                        if city.lower() in s.get("cluster_info", "").lower()
                    )
                    if city_count > 0:
                        cluster_info.append(f"{city}: {city_count}")

                cluster_summary = (
                    "<br>".join(cluster_info) if cluster_info else "Scattered samples"
                )

                # Add hexagon with enhanced popup and proper mouseover highlighting
                hex_polygon = folium.Polygon(
                    locations=coords,
                    popup=folium.Popup(
                        f"""
                    <div style="font-family: Arial, sans-serif; min-width: 250px;">
                        <h4 style="margin: 0 0 10px 0; color: #2E8B57; border-bottom: 2px solid #2E8B57; padding-bottom: 5px;">
                            🔷 H3 Spatial Cluster
                        </h4>
                        <table style="width: 100%; font-size: 12px;">
                            <tr><td><b>Cell ID:</b></td><td>{h3_index[:8]}...</td></tr>
                            <tr><td><b>Resolution:</b></td><td>{self.h3_resolution}</td></tr>
                            <tr><td><b>Total Samples:</b></td><td><span style="color: #d73027; font-weight: bold;">{sample_count}</span></td></tr>
                            <tr><td><b>Avg Diversity:</b></td><td>{avg_diversity:.2f}</td></tr>
                            <tr><td><b>Avg Temperature:</b></td><td>{avg_temp / 10:.1f}°C</td></tr>
                        </table>
                        <div style="margin-top: 10px; padding: 8px; background: #f0f8f0; border-radius: 4px;">
                            <b>🏙️ Cluster Composition:</b><br>
                            <small>{cluster_summary}</small>
                        </div>
                        <div style="margin-top: 8px; font-size: 10px; color: #666;">
                            💡 This hex contains data points from {sample_count} location(s)
                        </div>
                    </div>
                    """,
                        max_width=300,
                    ),
                    tooltip=f"H3 Cluster: {sample_count} samples • ID: {h3_index[:8]}...",
                    color=color,
                    weight=2,
                    fillColor=fill_color,
                    fillOpacity=opacity,
                )

                hex_polygon.add_to(h3_group)

            except Exception as e:
                logger.debug(f"Failed to add hexagon for {h3_index}: {e}")

        h3_group.add_to(m)
        logger.info(
            f"✅ Added {len(h3_sample_mapping)} H3 hexagons with clustering visualization"
        )

        # Add microbiome diversity layer with clustering
        logger.info("🧬 Adding microbiome diversity layer with clustering...")
        microbiome_cluster = MarkerCluster(
            name="Microbiome Diversity Clusters",
            options={
                "disableClusteringAtZoom": 10,
                "maxClusterRadius": 50,
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
            },
        ).add_to(microbiome_group)

        microbiome_data = [
            coord
            for coord in biological_data["coordinates"]
            if coord.get("data_type") == "microbiome"
        ]

        for coord in microbiome_data:
            # Enhanced microbiome marker with better popup
            folium.CircleMarker(
                location=[coord["lat"], coord["lon"]],
                radius=6,
                popup=folium.Popup(
                    f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4 style="color: #8B4513; margin: 0 0 8px 0;">🧬 Microbiome Sample</h4>
                    <table style="font-size: 11px; width: 100%;">
                        <tr><td><b>Shannon Diversity:</b></td><td>{coord["shannon_diversity"]:.2f}</td></tr>
                        <tr><td><b>Species Richness:</b></td><td>{coord["species_richness"]}</td></tr>
                        <tr><td><b>Temperature:</b></td><td>{coord["temperature"] / 10:.1f}°C</td></tr>
                        <tr><td><b>pH:</b></td><td>{coord["ph"]:.1f}</td></tr>
                        <tr><td><b>Location:</b></td><td>{coord["lat"]:.3f}, {coord["lon"]:.3f}</td></tr>
                    </table>
                </div>
                """,
                    max_width=250,
                ),
                tooltip=f"Diversity: {coord['shannon_diversity']:.2f}",
                color="#8B4513",
                fillColor="#DEB887",
                fillOpacity=0.8,
                weight=2,
            ).add_to(microbiome_cluster)

        microbiome_group.add_to(m)
        logger.info("✅ Added microbiome diversity layer with clustering")

        # Add climate data layer with clustering
        logger.info("🌡️ Adding climate data layer with clustering...")
        climate_cluster = MarkerCluster(
            name="Climate Station Clusters",
            options={
                "disableClusteringAtZoom": 8,
                "maxClusterRadius": 60,
                "spiderfyOnMaxZoom": True,
            },
        ).add_to(climate_group)

        climate_data = [
            coord
            for coord in biological_data["coordinates"]
            if coord.get("data_type") == "climate"
        ]

        for coord in climate_data:
            # Temperature-based marker coloring (using valid Folium colors)
            temp_c = coord["temperature"] / 10
            if temp_c > 20:
                marker_color = "red"
            elif temp_c > 15:
                marker_color = "orange"
            elif temp_c > 10:
                marker_color = "lightgreen"
            else:
                marker_color = "blue"

            folium.Marker(
                location=[coord["lat"], coord["lon"]],
                popup=folium.Popup(
                    f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4 style="color: #B22222; margin: 0 0 8px 0;">🌡️ Climate Station</h4>
                    <table style="font-size: 11px; width: 100%;">
                        <tr><td><b>Temperature (Bio1):</b></td><td>{temp_c:.1f}°C</td></tr>
                        <tr><td><b>Precipitation (Bio12):</b></td><td>{coord["precipitation"]}mm</td></tr>
                        <tr><td><b>Seasonality (Bio15):</b></td><td>{coord["precipitation_seasonality"]:.1f}</td></tr>
                        <tr><td><b>Station Type:</b></td><td>WorldClim-style</td></tr>
                        <tr><td><b>Location:</b></td><td>{coord["lat"]:.3f}, {coord["lon"]:.3f}</td></tr>
                    </table>
                </div>
                """,
                    max_width=250,
                ),
                tooltip=f"Temp: {temp_c:.1f}°C, Precip: {coord['precipitation']}mm",
                icon=folium.Icon(color=marker_color, icon="thermometer", prefix="fa"),
            ).add_to(climate_cluster)

        climate_group.add_to(m)
        logger.info(f"✅ Added climate data layer with {len(climate_data)} stations")

        # Add soil property layer with clustering
        logger.info("🌱 Adding soil property layer with clustering...")
        soil_cluster = MarkerCluster(
            name="Soil Sample Clusters",
            options={
                "disableClusteringAtZoom": 9,
                "maxClusterRadius": 45,
                "spiderfyOnMaxZoom": True,
            },
        ).add_to(soil_group)

        soil_data = [
            coord
            for coord in biological_data["coordinates"]
            if coord.get("data_type") == "soil"
        ]

        for coord in soil_data:
            # pH-based marker styling
            ph = coord["ph"]
            if ph < 5.5:
                soil_color = "#FF6B35"  # Acidic - orange-red
            elif ph < 6.5:
                soil_color = "#F7931E"  # Slightly acidic - orange
            elif ph < 7.5:
                soil_color = "#90EE90"  # Neutral - light green
            else:
                soil_color = "#228B22"  # Alkaline - dark green

            folium.CircleMarker(
                location=[coord["lat"], coord["lon"]],
                radius=5,
                popup=folium.Popup(
                    f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4 style="color: #8B4513; margin: 0 0 8px 0;">🌱 Soil Sample</h4>
                    <table style="font-size: 11px; width: 100%;">
                        <tr><td><b>pH:</b></td><td>{ph:.1f}</td></tr>
                        <tr><td><b>Organic Carbon:</b></td><td>{coord["organic_carbon"]:.1f}%</td></tr>
                        <tr><td><b>Bulk Density:</b></td><td>{coord["bulk_density"]:.2f} g/cm³</td></tr>
                        <tr><td><b>Sampling Strategy:</b></td><td>ISRIC-style</td></tr>
                        <tr><td><b>Location:</b></td><td>{coord["lat"]:.3f}, {coord["lon"]:.3f}</td></tr>
                    </table>
                </div>
                """,
                    max_width=250,
                ),
                tooltip=f"pH: {ph:.1f}, OC: {coord['organic_carbon']:.1f}%",
                color="#8B4513",
                fillColor=soil_color,
                fillOpacity=0.8,
                weight=2,
            ).add_to(soil_cluster)

        soil_group.add_to(m)
        logger.info(
            f"✅ Added soil property layer with {len(soil_data)} sampling sites"
        )

        # Add enhanced layer control panel
        logger.info("🎛️ Adding enhanced layer control panel...")

        # Custom control panel HTML with improved styling
        control_html = """
        <div id="layer-control-panel" style="
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 200px;
        ">
            <h3 style="
                margin: 0 0 15px 0;
                color: #2E8B57;
                border-bottom: 2px solid #2E8B57;
                padding-bottom: 8px;
                font-size: 16px;
            ">🎛️ Data Layers</h3>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; cursor: pointer; margin: 8px 0;">
                    <input type="checkbox" id="h3-toggle" checked style="margin-right: 8px; transform: scale(1.2);">
                    <span style="color: #4575b4; font-weight: bold;">🔷 H3 Spatial Grid</span>
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; cursor: pointer; margin: 8px 0;">
                    <input type="checkbox" id="microbiome-toggle" checked style="margin-right: 8px; transform: scale(1.2);">
                    <span style="color: #8B4513; font-weight: bold;">🧬 Microbiome Data</span>
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; cursor: pointer; margin: 8px 0;">
                    <input type="checkbox" id="climate-toggle" checked style="margin-right: 8px; transform: scale(1.2);">
                    <span style="color: #B22222; font-weight: bold;">🌡️ Climate Stations</span>
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; cursor: pointer; margin: 8px 0;">
                    <input type="checkbox" id="soil-toggle" checked style="margin-right: 8px; transform: scale(1.2);">
                    <span style="color: #8B4513; font-weight: bold;">🌱 Soil Properties</span>
                </label>
            </div>
            
            <div style="
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
                font-size: 11px;
                color: #666;
            ">
                <div>💡 Hover over H3 hexes for spatial clustering</div>
                <div>🔄 Toggle layers on/off with checkboxes</div>
            </div>
        </div>
        
        <style>
        /* Enhanced H3 hexagon hover effects */
        .leaflet-interactive:hover {
            stroke-width: 4px !important;
            stroke: #FFD700 !important;
            stroke-opacity: 1.0 !important;
            fill-opacity: 0.9 !important;
            cursor: pointer !important;
        }
        
        /* Hide default Folium layer control */
        .leaflet-control-layers {
            display: none !important;
        }
        
        /* Improve marker cluster styling */
        .marker-cluster {
            background: rgba(181, 226, 140, 0.6) !important;
            border: 2px solid rgba(110, 204, 57, 0.6) !important;
        }
        
        .marker-cluster div {
            background: rgba(110, 204, 57, 0.6) !important;
            color: white !important;
            font-weight: bold !important;
        }
        
        /* Custom toggle styling */
        #layer-control-panel input[type="checkbox"] {
            cursor: pointer;
        }
        
        #layer-control-panel label:hover {
            background-color: rgba(46, 139, 87, 0.1);
            border-radius: 4px;
            padding: 2px;
        }
        </style>
        
        <script>
        // Simple, direct approach to layer control
        document.addEventListener('DOMContentLoaded', function() {
            let mapInstance;
            let allFeatureGroups = [];
            
            // Wait for all elements to be ready
            setTimeout(function() {
                console.log('🚀 Initializing direct layer control...');
                
                // Find map instance
                const mapKeys = Object.keys(window).filter(key => key.startsWith('map_'));
                if (mapKeys.length > 0) {
                    mapInstance = window[mapKeys[0]];
                    console.log('✅ Map found:', mapKeys[0]);
                } else {
                    console.error('❌ No map instance found');
                    return;
                }
                
                // Collect all feature groups
                const featureGroupKeys = Object.keys(window).filter(key => key.startsWith('feature_group_'));
                console.log('🔍 Feature groups found:', featureGroupKeys.length);
                
                featureGroupKeys.forEach((key, index) => {
                    const layer = window[key];
                    if (layer && layer.getLayers) {
                        allFeatureGroups.push({
                            index: index,
                            key: key,
                            layer: layer,
                            childCount: layer.getLayers().length
                        });
                        console.log(`Layer ${index}: ${key} (${layer.getLayers().length} children)`);
                    }
                });
                
                // Sort by child count for consistent assignment
                allFeatureGroups.sort((a, b) => b.childCount - a.childCount);
                
                // Simple assignment based on order and size
                const h3Group = allFeatureGroups[0]?.layer;        // Largest (H3 hexagons)
                const microbiomeGroup = allFeatureGroups[1]?.layer; // Second largest (microbiome clusters)  
                const soilGroup = allFeatureGroups[2]?.layer;       // Third (soil samples)
                const climateGroup = allFeatureGroups[3]?.layer;    // Smallest (climate stations)
                
                console.log('📊 Layer assignment by size:', {
                    h3: allFeatureGroups[0]?.childCount || 0,
                    microbiome: allFeatureGroups[1]?.childCount || 0,
                    soil: allFeatureGroups[2]?.childCount || 0,
                    climate: allFeatureGroups[3]?.childCount || 0
                });
                
                // Set up direct toggle handlers
                function setupDirectToggle(toggleId, layer, layerName) {
                    const toggle = document.getElementById(toggleId);
                    if (!toggle || !layer) {
                        console.warn(`⚠️ Missing toggle (${!!toggle}) or layer (${!!layer}) for ${layerName}`);
                        return;
                    }
                    
                    toggle.addEventListener('change', function(e) {
                        const isVisible = e.target.checked;
                        console.log(`🔄 ${layerName} visibility: ${isVisible}`);
                        
                        try {
                            if (isVisible) {
                                // Add layer if not present
                                if (!mapInstance.hasLayer(layer)) {
                                    mapInstance.addLayer(layer);
                                    console.log(`✅ Added ${layerName}`);
                                } else {
                                    console.log(`ℹ️ ${layerName} already visible`);
                                }
                            } else {
                                // Remove layer if present
                                if (mapInstance.hasLayer(layer)) {
                                    mapInstance.removeLayer(layer);
                                    console.log(`❌ Removed ${layerName}`);
                                } else {
                                    console.log(`ℹ️ ${layerName} already hidden`);
                                }
                            }
                            
                            // Verify the state
                            console.log(`🔍 ${layerName} on map:`, mapInstance.hasLayer(layer));
                            
                        } catch (error) {
                            console.error(`💥 Error toggling ${layerName}:`, error);
                        }
                    });
                    
                    console.log(`🎛️ ${layerName} toggle ready`);
                }
                
                // Set up all toggles with direct layer references
                setupDirectToggle('h3-toggle', h3Group, 'H3 Grid');
                setupDirectToggle('microbiome-toggle', microbiomeGroup, 'Microbiome');
                setupDirectToggle('climate-toggle', climateGroup, 'Climate');
                setupDirectToggle('soil-toggle', soilGroup, 'Soil');
                
                console.log('🎉 Direct layer control initialized');
                
                // Test initial state
                console.log('🔍 Initial layer states:', {
                    h3: mapInstance.hasLayer(h3Group),
                    microbiome: mapInstance.hasLayer(microbiomeGroup),
                    climate: mapInstance.hasLayer(climateGroup),
                    soil: mapInstance.hasLayer(soilGroup)
                });
                
            }, 3000); // Give Folium time to fully initialize
        });
        </script>
        """

        # Add custom control to map
        custom_control = folium.Element(control_html)
        m.get_root().html.add_child(custom_control)

        # Save the map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.output_dir / f"spatial_biological_integration_{timestamp}.html"
        )
        m.save(str(output_file))

        logger.info(f"🎉 Interactive visualization created: {output_file}")
        logger.info(
            f"📊 Map includes {len(biological_data['coordinates'])} biological sample locations"
        )
        logger.info(f"🔷 H3 resolution: {self.h3_resolution}")
        logger.info(
            "🎛️ Enhanced layer control panel added to top-right corner with custom styling"
        )

        return str(output_file)

    def run_complete_analysis(
        self,
        region_bbox: tuple[float, float, float, float] = (-130, 25, -65, 55),
        max_samples: int = 1000,
        output_format: str = "interactive",
    ) -> dict[str, str]:
        """
        Run the complete spatial microbiome-climate-soil analysis.

        This is the main orchestration method that demonstrates the full workflow:
        1. Load biological datasets from multiple sources
        2. Perform H3 spatial integration
        3. Create multi-overlay visualization

        Args:
            region_bbox: Analysis region bounding box
            max_samples: Maximum microbiome samples to process
            output_format: Visualization output format

        Returns:
            Dictionary with paths to generated outputs
        """
        logger.info("🚀 Starting Complete Spatial Biological Analysis")
        logger.info(f"🌍 Region: {region_bbox}")
        logger.info(f"📊 Max samples: {max_samples}")
        logger.info(f"🔷 H3 resolution: {self.h3_resolution}")
        logger.info(f"📈 Output format: {output_format}")

        start_time = datetime.now()

        try:
            # Step 1: Load biological datasets
            datasets = self.load_biological_datasets(region_bbox, max_samples)

            # Step 2: Create interactive visualization
            visualization_path = self.create_interactive_h3_visualization(
                datasets, output_format=output_format
            )

            # Save integration results
            results_data = {
                "analysis_region": region_bbox,
                "max_samples": max_samples,
                "h3_resolution": self.h3_resolution,
                "coordinates_count": len(datasets.get("coordinates", [])),
                "microbiome_source": datasets.get("microbiome", {}).get(
                    "data_source", "Unknown"
                ),
                "climate_source": datasets.get("climate", {}).get(
                    "data_source", "Unknown"
                ),
                "soil_source": datasets.get("soil", {}).get("data_source", "Unknown"),
                "processing_time": str(datetime.now() - start_time),
                "visualization_path": visualization_path,
                "timestamp": datetime.now().isoformat(),
            }

            results_path = self.output_dir / "integration_results.json"
            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=2)

            end_time = datetime.now()
            processing_time = end_time - start_time

            logger.info("🎉 Analysis Complete!")
            logger.info(f"⏱️  Processing time: {processing_time}")
            logger.info(
                f"📊 Coordinates processed: {len(datasets.get('coordinates', []))}"
            )
            logger.info(f"🗺️  Visualization: {visualization_path}")
            logger.info(f"📄 Results: {results_path}")

            return {
                "visualization": visualization_path,
                "results": str(results_path),
                "processing_time": str(processing_time),
                "coordinates": str(len(datasets.get("coordinates", []))),
            }

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            # Create error report
            error_path = self.output_dir / "error_report.json"
            with open(error_path, "w") as f:
                json.dump(
                    {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "parameters": {
                            "region_bbox": region_bbox,
                            "max_samples": max_samples,
                            "h3_resolution": self.h3_resolution,
                        },
                    },
                    f,
                    indent=2,
                )

            return {"error": str(e), "error_report": str(error_path)}


def main():
    """Main entry point for the spatial integration script."""
    parser = argparse.ArgumentParser(
        description="Spatial Microbiome-Climate-Soil Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h3_resolution",
        type=int,
        default=7,
        help="H3 hexagonal grid resolution (0-15, higher = finer resolution)",
    )

    parser.add_argument(
        "--output_format",
        choices=["interactive", "static", "both"],
        default="interactive",
        help="Visualization output format",
    )

    parser.add_argument(
        "--region",
        default="north_america",
        choices=["north_america", "global", "custom"],
        help="Analysis region",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of microbiome samples to process",
    )

    parser.add_argument(
        "--output_dir", default="output", help="Output directory for results"
    )

    args = parser.parse_args()

    # Define region bounding boxes
    regions = {
        "north_america": (-130, 25, -65, 55),
        "global": (-180, -60, 180, 75),
        "custom": (-130, 25, -65, 55),  # Default to North America
    }

    region_bbox = regions[args.region]

    # Initialize and run analysis
    integrator = SpatialMicrobiomeIntegrator(
        output_dir=args.output_dir, h3_resolution=args.h3_resolution
    )

    try:
        results = integrator.run_complete_analysis(
            region_bbox=region_bbox,
            max_samples=args.max_samples,
            output_format=args.output_format,
        )

        print("\n" + "=" * 60)
        print("🎯 SPATIAL INTEGRATION COMPLETE")
        print("=" * 60)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
            print(f"📄 Error report: {results.get('error_report', 'N/A')}")
        else:
            print(f"🗺️  Visualization: {results['visualization']}")
            print(f"📄 Results file: {results['results']}")
            print(f"⏱️  Processing time: {results['processing_time']}")
            print(f"📊 Coordinates: {results['coordinates']}")
            print(
                "\n🌐 Open the visualization file in your web browser to explore the interactive map!"
            )

        print("=" * 60)

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        print(f"\n❌ Integration failed: {e}")
        print("📦 Make sure required dependencies are installed:")
        print("   uv pip install pandas numpy folium h3-py geopandas")
        sys.exit(1)


if __name__ == "__main__":
    main()
