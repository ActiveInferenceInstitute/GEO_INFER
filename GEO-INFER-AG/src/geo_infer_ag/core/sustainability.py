"""
Sustainability assessment functionality for agricultural applications.
"""

from typing import Any
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


class SustainabilityAssessment:
    """
    Assess sustainability aspects of agricultural practices.

    This class provides methods for assessing environmental, economic, and social
    sustainability of agricultural practices, with a focus on carbon sequestration,
    water usage, soil health, and biodiversity.

    Attributes:
        field_data: Spatial data of agricultural fields
        metrics: Dictionary of computed sustainability metrics
    """

    def __init__(
        self,
        field_data: gpd.GeoDataFrame | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the sustainability assessment.

        Args:
            field_data: Optional GeoDataFrame of agricultural fields
            config: Optional configuration parameters
        """
        self.field_data = field_data
        self.config = config or {}
        self.metrics: dict[str, Any] = {}

    def assess_carbon_sequestration(
        self,
        field_data: gpd.GeoDataFrame | None = None,
        crop_type_column: str = "crop_type",
        soil_carbon_column: str | None = None,
        biomass_column: str | None = None,
        management_practices: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """
        Assess carbon sequestration potential of agricultural fields.

        Args:
            field_data: Optional GeoDataFrame (uses self.field_data if None)
            crop_type_column: Column name for crop types
            soil_carbon_column: Column name for soil carbon content
            biomass_column: Column name for biomass data
            management_practices: Dictionary mapping field IDs to practices

        Returns:
            Dictionary of carbon sequestration metrics

        Raises:
            ValueError: If field_data is not provided and not set in constructor, or if management_practices are provided but the field data lacks a 'field_id' column
        """
        if field_data is None:
            if self.field_data is None:
                raise ValueError("No field data provided")
            field_data = self.field_data

        # Make a copy to avoid modifying the original
        result_data = field_data.copy()

        # Carbon sequestration rates by crop type (metric tons CO2e/ha/year)
        # These are example values - real implementation would use more detailed models
        carbon_rates = {
            "corn": 1.8,
            "wheat": 1.2,
            "rice": 0.7,
            "soybean": 1.5,
            "cotton": 0.9,
            "alfalfa": 2.3,
            "grass": 2.7,
            "forest": 5.0,
            "vegetables": 1.0,
            "orchard": 3.5,
            "cover_crop": 2.1,
        }

        # Management practice modifiers
        practice_modifiers = {
            "no_till": 1.3,
            "reduced_till": 1.15,
            "conventional_till": 0.9,
            "cover_crops": 1.25,
            "crop_rotation": 1.1,
            "organic_fertilizer": 1.15,
            "agroforestry": 1.4,
            "precision_agriculture": 1.05,
            "residue_management": 1.2,
        }

        # Default carbon rate (t/ha/yr); refine per crop when the crop type
        # column is available. Initializing unconditionally guarantees the
        # column exists even when crop data is absent.
        result_data["carbon_rate"] = 1.0
        if crop_type_column in result_data.columns:
            for crop, rate in carbon_rates.items():
                mask = result_data[crop_type_column].str.lower() == crop
                result_data.loc[mask, "carbon_rate"] = rate

        # Apply management practice modifiers if provided
        if management_practices:
            if "field_id" not in result_data.columns:
                raise ValueError(
                    "field data must include 'field_id' when management_practices "
                    "are provided"
                )

            # Create management modifier column with default value of 1.0
            result_data["management_modifier"] = 1.0

            for field_id, practices in management_practices.items():
                modifier = 1.0
                for practice in practices:
                    if practice in practice_modifiers:
                        modifier *= practice_modifiers[practice]

                # Apply modifier to the field
                mask = result_data["field_id"] == field_id
                result_data.loc[mask, "management_modifier"] = modifier
        else:
            result_data["management_modifier"] = 1.0

        # Calculate total carbon sequestration
        result_data["carbon_sequestration"] = (
            result_data["carbon_rate"]
            * result_data["management_modifier"]
            * result_data["area_ha"]
        )

        # Prepare metrics
        metrics = {
            "total_carbon_sequestration": result_data["carbon_sequestration"].sum(),
            "mean_carbon_sequestration_per_ha": (
                result_data["carbon_sequestration"].sum() / result_data["area_ha"].sum()
            ),
            "field_data": result_data,
        }
        if crop_type_column in result_data.columns:
            metrics["carbon_sequestration_by_crop"] = (
                result_data.groupby(crop_type_column)["carbon_sequestration"]
                .sum()
                .to_dict()
            )

        # Store results
        self.metrics["carbon_sequestration"] = metrics
        return metrics

    def assess_water_usage(
        self,
        field_data: gpd.GeoDataFrame | None = None,
        water_data: pd.DataFrame | None = None,
        crop_type_column: str = "crop_type",
        precipitation_column: str | None = None,
        irrigation_column: str | None = None,
        evapotranspiration_column: str | None = None,
        reference_period: str | None = None,
    ) -> dict[str, Any]:
        """
        Assess water usage and efficiency of agricultural fields.

        Args:
            field_data: Optional GeoDataFrame (uses self.field_data if None)
            water_data: Optional water data DataFrame
            crop_type_column: Column name for crop types
            precipitation_column: Column name for precipitation data
            irrigation_column: Column name for irrigation data
            evapotranspiration_column: Column name for evapotranspiration data
            reference_period: Optional period for comparison

        Returns:
            Dictionary of water usage metrics

        Raises:
            ValueError: If field_data is not provided and not set in constructor
        """
        if field_data is None:
            if self.field_data is None:
                raise ValueError("No field data provided")
            field_data = self.field_data

        # Make a copy to avoid modifying the original
        result_data = field_data.copy()

        # Typical crop water requirements (mm/year)
        crop_water_requirements = {
            "corn": 550,
            "wheat": 450,
            "rice": 1200,
            "soybean": 500,
            "cotton": 700,
            "alfalfa": 900,
            "grass": 700,
            "vegetables": 400,
            "orchard": 800,
        }

        # Default water requirement (mm/yr); refine per crop when the crop
        # type column is available. Initializing unconditionally guarantees
        # the column exists even when crop data is absent.
        result_data["water_requirement_mm"] = 500
        if crop_type_column in result_data.columns:
            for crop, req in crop_water_requirements.items():
                mask = result_data[crop_type_column].str.lower() == crop
                result_data.loc[mask, "water_requirement_mm"] = req

        # Calculate total water requirement in cubic meters
        result_data["water_requirement_m3"] = (
            result_data["water_requirement_mm"]
            * result_data["area_ha"]
            * 10  # Convert mm*ha to m3
        )

        # If actual water data is provided, use it
        if water_data is not None:
            if "field_id" in water_data.columns:
                water_summary = water_data.groupby("field_id").mean(numeric_only=True)
                for col in water_summary.columns:
                    if col not in result_data.columns:
                        result_data = result_data.merge(
                            water_summary[[col]],
                            left_on="field_id",
                            right_index=True,
                            how="left",
                        )

        # If precipitation data is available, calculate water efficiency
        if precipitation_column in result_data.columns:
            result_data["water_efficiency"] = (
                result_data[precipitation_column] / result_data["water_requirement_mm"]
            )

        # If irrigation data is available, calculate irrigation efficiency
        if irrigation_column in result_data.columns:
            result_data["irrigation_efficiency"] = result_data[
                "water_requirement_mm"
            ] / (
                result_data[irrigation_column] + 0.1
            )  # Add small value to avoid division by zero

        # Calculate water footprint
        result_data["water_footprint"] = result_data["water_requirement_m3"]

        # Prepare metrics
        metrics = {
            "total_water_requirement": result_data["water_requirement_m3"].sum(),
            "mean_water_requirement_per_ha": (
                result_data["water_requirement_m3"].sum() / result_data["area_ha"].sum()
            ),
            "field_data": result_data,
        }
        if crop_type_column in result_data.columns:
            metrics["water_requirement_by_crop"] = (
                result_data.groupby(crop_type_column)["water_requirement_m3"]
                .sum()
                .to_dict()
            )

        # Add efficiency metrics if available
        if "water_efficiency" in result_data.columns:
            metrics["mean_water_efficiency"] = result_data["water_efficiency"].mean()

        if "irrigation_efficiency" in result_data.columns:
            metrics["mean_irrigation_efficiency"] = result_data[
                "irrigation_efficiency"
            ].mean()

        # Store results
        self.metrics["water_usage"] = metrics
        return metrics

    def assess_soil_health(
        self,
        field_data: gpd.GeoDataFrame | None = None,
        soil_data: gpd.GeoDataFrame | None = None,
        organic_matter_column: str | None = None,
        ph_column: str | None = None,
        erosion_column: str | None = None,
        management_practices: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """
        Assess soil health of agricultural fields.

        Args:
            field_data: Optional GeoDataFrame (uses self.field_data if None)
            soil_data: Optional soil data GeoDataFrame
            organic_matter_column: Column name for organic matter content
            ph_column: Column name for soil pH
            erosion_column: Column name for erosion rate
            management_practices: Dictionary mapping field IDs to practices

        Returns:
            Dictionary of soil health metrics

        Raises:
            ValueError: If field_data is not provided and not set in constructor
        """
        if field_data is None:
            if self.field_data is None:
                raise ValueError("No field data provided")
            field_data = self.field_data

        # Make a copy to avoid modifying the original
        result_data = field_data.copy()

        # If soil data is provided separately, join with field data
        if soil_data is not None:
            # Join soil observations to fields: aggregate per field_id when
            # present, otherwise spatially overlay soil geometries onto the
            # field boundaries and average numeric attributes per field.
            if "field_id" in soil_data.columns:
                # If field_id is in soil data, use it to join
                soil_summary = soil_data.groupby("field_id").mean(numeric_only=True)
                result_data = result_data.merge(
                    soil_summary, left_on="field_id", right_index=True, how="left"
                )
            else:
                # Spatial join: overlay soil points/polygons onto field boundaries
                try:
                    joined = gpd.sjoin(
                        result_data, soil_data, how="left", predicate="intersects"
                    )
                    numeric_cols = joined.select_dtypes(include="number").columns
                    result_data = (
                        joined.groupby(joined.index)
                        .agg({c: "mean" for c in numeric_cols})
                        .combine_first(result_data)
                    )
                except Exception as exc:
                    logger.warning(
                        "Spatial join of soil data failed; proceeding without "
                        "soil overlay: %s",
                        exc,
                    )

        # Auto-detect column names from merged data when not explicitly provided
        if organic_matter_column is None:
            for candidate in ["organic_matter", "om", "soc", "soil_organic_carbon"]:
                if candidate in result_data.columns:
                    organic_matter_column = candidate
                    break
        if ph_column is None:
            for candidate in ["ph", "ph_value", "soil_ph"]:
                if candidate in result_data.columns:
                    ph_column = candidate
                    break
        if erosion_column is None:
            for candidate in ["erosion", "erosion_rate", "soil_loss"]:
                if candidate in result_data.columns:
                    erosion_column = candidate
                    break

        # Management practice modifiers for soil health
        practice_modifiers = {
            "no_till": 1.3,
            "reduced_till": 1.15,
            "conventional_till": 0.8,
            "cover_crops": 1.25,
            "crop_rotation": 1.1,
            "organic_fertilizer": 1.2,
            "residue_management": 1.15,
            "erosion_control": 1.3,
            "compost_application": 1.25,
            "bio_char": 1.4,
        }

        # Calculate soil health score if data is available
        health_components = {}

        # Organic matter component (weight: 0.4)
        if organic_matter_column in result_data.columns:
            # Score from 0 to 10 based on organic matter
            result_data["organic_matter_score"] = np.clip(
                result_data[organic_matter_column] * 2, 0, 10
            )
            health_components["organic_matter"] = 0.4

        # pH component (weight: 0.3)
        if ph_column in result_data.columns:
            # Score from 0 to 10 based on pH (optimal: 6.0-7.0)
            result_data["ph_score"] = 10 - np.clip(
                np.abs(result_data[ph_column] - 6.5) * 3, 0, 10
            )
            health_components["ph"] = 0.3

        # Erosion component (weight: 0.3)
        if erosion_column in result_data.columns:
            # Score from 0 to 10 based on erosion (lower is better)
            # Assuming erosion in tons/ha/year
            result_data["erosion_score"] = 10 - np.clip(
                result_data[erosion_column], 0, 10
            )
            health_components["erosion"] = 0.3

        # Calculate weighted soil health score if components exist
        if health_components:
            # Normalize weights
            weight_sum = sum(health_components.values())
            normalized_weights = {
                k: v / weight_sum for k, v in health_components.items()
            }

            # Calculate weighted score
            result_data["soil_health_score"] = 0

            if "organic_matter" in normalized_weights:
                result_data["soil_health_score"] += (
                    result_data["organic_matter_score"]
                    * normalized_weights["organic_matter"]
                )

            if "ph" in normalized_weights:
                result_data["soil_health_score"] += (
                    result_data["ph_score"] * normalized_weights["ph"]
                )

            if "erosion" in normalized_weights:
                result_data["soil_health_score"] += (
                    result_data["erosion_score"] * normalized_weights["erosion"]
                )

        # Apply management practice modifiers if provided
        if management_practices:
            # Create management modifier column with default value of 1.0
            result_data["soil_management_modifier"] = 1.0

            for field_id, practices in management_practices.items():
                modifier = 1.0
                for practice in practices:
                    if practice in practice_modifiers:
                        modifier *= practice_modifiers[practice]

                # Apply modifier to the field
                mask = result_data["field_id"] == field_id
                result_data.loc[mask, "soil_management_modifier"] = modifier

            # Apply modifier to soil health score if it exists
            if "soil_health_score" in result_data.columns:
                result_data["soil_health_score"] = (
                    result_data["soil_health_score"]
                    * result_data["soil_management_modifier"]
                )
                # Cap at 10
                result_data["soil_health_score"] = np.clip(
                    result_data["soil_health_score"], 0, 10
                )

        # Prepare metrics
        metrics = {"field_data": result_data}

        # Add summary metrics if soil health score exists
        if "soil_health_score" in result_data.columns:
            metrics["mean_soil_health_score"] = result_data["soil_health_score"].mean()
            metrics["min_soil_health_score"] = result_data["soil_health_score"].min()
            metrics["max_soil_health_score"] = result_data["soil_health_score"].max()

            # Add healthy/unhealthy area statistics
            healthy_threshold = 7.0
            healthy_fields = result_data[
                result_data["soil_health_score"] >= healthy_threshold
            ]
            unhealthy_fields = result_data[
                result_data["soil_health_score"] < healthy_threshold
            ]

            metrics["healthy_soil_area_ha"] = healthy_fields["area_ha"].sum()
            metrics["unhealthy_soil_area_ha"] = unhealthy_fields["area_ha"].sum()
            metrics["healthy_soil_percentage"] = (
                metrics["healthy_soil_area_ha"]
                / (metrics["healthy_soil_area_ha"] + metrics["unhealthy_soil_area_ha"])
                * 100
            )

        # Store results
        self.metrics["soil_health"] = metrics
        return metrics

    def assess_biodiversity(
        self,
        field_data: gpd.GeoDataFrame | None = None,
        biodiversity_data: gpd.GeoDataFrame | None = None,
        edge_habitat_buffer: float = 10.0,
        protected_areas: gpd.GeoDataFrame | None = None,
        management_practices: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """
        Assess biodiversity impact of agricultural fields.

        Args:
            field_data: Optional GeoDataFrame (uses self.field_data if None)
            biodiversity_data: Optional biodiversity metrics GeoDataFrame
            edge_habitat_buffer: Buffer distance (m) for edge habitat calculation
            protected_areas: Optional GeoDataFrame of protected areas
            management_practices: Dictionary mapping field IDs to practices

        Returns:
            Dictionary of biodiversity metrics

        Raises:
            ValueError: If field_data is not provided and not set in constructor
        """
        if field_data is None:
            if self.field_data is None:
                raise ValueError("No field data provided")
            field_data = self.field_data

        # Make a copy to avoid modifying the original
        result_data = field_data.copy()

        # Calculate edge habitat length and area
        # Edge habitats often support higher biodiversity
        metric_data = (
            result_data.to_crs("EPSG:3857")
            if result_data.crs and result_data.crs.is_geographic
            else result_data
        )
        result_data["perimeter_m"] = metric_data.geometry.length
        result_data["edge_habitat_area_ha"] = (
            metric_data.geometry.buffer(edge_habitat_buffer).area
            - metric_data.geometry.area
        ) / 10000  # Convert m² to ha

        # Calculate edge density (m/ha) - higher values generally better for biodiversity
        result_data["edge_density"] = (
            result_data["perimeter_m"] / result_data["area_ha"]
        )

        # Management practice modifiers for biodiversity
        practice_modifiers = {
            "no_till": 1.2,
            "organic_farming": 1.4,
            "crop_rotation": 1.2,
            "cover_crops": 1.3,
            "hedgerows": 1.5,
            "agroforestry": 1.8,
            "pollinator_habitat": 1.7,
            "reduced_pesticide": 1.4,
            "wildlife_corridors": 1.6,
            "wetland_conservation": 1.5,
        }

        # Calculate biodiversity score based on landscape metrics and practices
        # Start with a base score derived from edge density
        result_data["biodiversity_base_score"] = np.clip(
            result_data["edge_density"] / 10, 0, 10
        )

        # Apply management practice modifiers if provided
        if management_practices:
            # Create management modifier column with default value of 1.0
            result_data["biodiversity_modifier"] = 1.0

            for field_id, practices in management_practices.items():
                modifier = 1.0
                for practice in practices:
                    if practice in practice_modifiers:
                        modifier *= practice_modifiers[practice]

                # Apply modifier to the field
                mask = result_data["field_id"] == field_id
                result_data.loc[mask, "biodiversity_modifier"] = modifier
        else:
            result_data["biodiversity_modifier"] = 1.0

        # Calculate final biodiversity score
        result_data["biodiversity_score"] = (
            result_data["biodiversity_base_score"]
            * result_data["biodiversity_modifier"]
        )
        # Cap at 10
        result_data["biodiversity_score"] = np.clip(
            result_data["biodiversity_score"], 0, 10
        )

        # Check proximity to protected areas if provided
        if protected_areas is not None:
            # Reproject to a local metric CRS (UTM) so distances are true
            # meters; in a geographic CRS (e.g. EPSG:4326) shapely distances
            # would be measured in degrees.
            metric_crs = result_data.estimate_utm_crs()
            fields_metric = (
                result_data.to_crs(metric_crs)
                if metric_crs is not None
                else result_data
            )
            areas_metric = (
                protected_areas.to_crs(metric_crs)
                if metric_crs is not None and protected_areas.crs is not None
                else protected_areas
            )

            distances = []
            for _, field in fields_metric.iterrows():
                field_geom = field.geometry
                min_distance = float("inf")

                for _, area in areas_metric.iterrows():
                    distance = field_geom.distance(area.geometry)
                    min_distance = min(min_distance, distance)

                distances.append(min_distance)

            result_data["distance_to_protected_m"] = distances

            # Add proximity score (0-10, higher for closer proximity)
            result_data["proximity_score"] = np.clip(
                10 - (result_data["distance_to_protected_m"] / 1000), 0, 10
            )

            # Update biodiversity score with proximity influence
            result_data["biodiversity_score"] = (
                result_data["biodiversity_score"] * 0.7
                + result_data["proximity_score"] * 0.3
            )

        # Prepare metrics
        metrics = {
            "mean_biodiversity_score": result_data["biodiversity_score"].mean(),
            "mean_edge_density": result_data["edge_density"].mean(),
            "total_edge_habitat_ha": result_data["edge_habitat_area_ha"].sum(),
            "field_data": result_data,
        }

        # Add proximity metrics if available
        if "distance_to_protected_m" in result_data.columns:
            metrics["mean_distance_to_protected_m"] = result_data[
                "distance_to_protected_m"
            ].mean()
            metrics["min_distance_to_protected_m"] = result_data[
                "distance_to_protected_m"
            ].min()

        # Store results
        self.metrics["biodiversity"] = metrics
        return metrics

    def calculate_sustainability_index(
        self, weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Calculate overall sustainability index from individual metrics.

        Args:
            weights: Optional weight dictionary for each sustainability component

        Returns:
            Dictionary with sustainability index metrics

        Raises:
            ValueError: If no metrics have been calculated
        """
        if not self.metrics:
            raise ValueError("No sustainability metrics have been calculated")

        # Default weights if not provided
        if weights is None:
            weights = {
                "carbon_sequestration": 0.25,
                "water_usage": 0.25,
                "soil_health": 0.25,
                "biodiversity": 0.25,
            }

        # Normalize weights
        weight_sum = sum(weights.values())
        normalized_weights = {k: v / weight_sum for k, v in weights.items()}

        # Check which metrics are available
        available_metrics = []

        if "carbon_sequestration" in self.metrics:
            carbon_data = self.metrics["carbon_sequestration"]["field_data"]
            if "carbon_sequestration" in carbon_data.columns:
                # Normalize carbon sequestration to 0-10 scale
                max_carbon = carbon_data["carbon_sequestration"].max()
                carbon_data["carbon_score"] = np.clip(
                    carbon_data["carbon_sequestration"] / (max_carbon / 10), 0, 10
                )
                available_metrics.append("carbon_sequestration")

        if "water_usage" in self.metrics:
            water_data = self.metrics["water_usage"]["field_data"]
            if "water_efficiency" in water_data.columns:
                # Higher efficiency is better (0-10 scale)
                water_data["water_score"] = np.clip(
                    water_data["water_efficiency"] * 10, 0, 10
                )
                available_metrics.append("water_usage")
            elif "water_requirement_mm" in water_data.columns:
                # Fallback score when water efficiency data is unavailable
                water_data["water_score"] = np.full(len(water_data), 5.0)
                available_metrics.append("water_usage")

        if "soil_health" in self.metrics:
            soil_data = self.metrics["soil_health"]["field_data"]
            if "soil_health_score" in soil_data.columns:
                # Already on 0-10 scale
                soil_data["soil_score"] = soil_data["soil_health_score"]
                available_metrics.append("soil_health")

        if "biodiversity" in self.metrics:
            biodiv_data = self.metrics["biodiversity"]["field_data"]
            if "biodiversity_score" in biodiv_data.columns:
                # Already on 0-10 scale
                biodiv_data["biodiv_score"] = biodiv_data["biodiversity_score"]
                available_metrics.append("biodiversity")

        # If no metrics available, raise error
        if not available_metrics:
            raise ValueError("No scorable sustainability metrics available")

        # Re-normalize weights based on available metrics
        available_weight_sum = sum(normalized_weights[m] for m in available_metrics)
        final_weights = {
            m: normalized_weights[m] / available_weight_sum for m in available_metrics
        }

        # Create copy of field data for sustainability index
        if self.field_data is not None:
            result_data = self.field_data.copy()
        else:
            # Use the first available metric's field data
            result_data = self.metrics[available_metrics[0]]["field_data"].copy()

        # Add scores from individual metrics
        for metric in available_metrics:
            score_column = f"{metric.split('_')[0]}_score"
            metric_data = self.metrics[metric]["field_data"]
            result_data[score_column] = metric_data[score_column]

        # Calculate weighted sustainability index
        result_data["sustainability_index"] = 0

        if "carbon_sequestration" in available_metrics:
            result_data["sustainability_index"] += (
                result_data["carbon_score"] * final_weights["carbon_sequestration"]
            )

        if "water_usage" in available_metrics:
            result_data["sustainability_index"] += (
                result_data["water_score"] * final_weights["water_usage"]
            )

        if "soil_health" in available_metrics:
            result_data["sustainability_index"] += (
                result_data["soil_score"] * final_weights["soil_health"]
            )

        if "biodiversity" in available_metrics:
            result_data["sustainability_index"] += (
                result_data["biodiversity_score"] * final_weights["biodiversity"]
            )

        # Prepare sustainability index metrics
        sustainability_metrics = {
            "mean_sustainability_index": result_data["sustainability_index"].mean(),
            "min_sustainability_index": result_data["sustainability_index"].min(),
            "max_sustainability_index": result_data["sustainability_index"].max(),
            "weights": final_weights,
            "available_components": available_metrics,
            "field_data": result_data,
        }

        # Add categorization of fields
        high_threshold = 7.5
        medium_threshold = 5.0

        sustainability_metrics["high_sustainability_area_ha"] = result_data[
            result_data["sustainability_index"] >= high_threshold
        ]["area_ha"].sum()

        sustainability_metrics["medium_sustainability_area_ha"] = result_data[
            (result_data["sustainability_index"] >= medium_threshold)
            & (result_data["sustainability_index"] < high_threshold)
        ]["area_ha"].sum()

        sustainability_metrics["low_sustainability_area_ha"] = result_data[
            result_data["sustainability_index"] < medium_threshold
        ]["area_ha"].sum()

        # Calculate percentages
        total_area = result_data["area_ha"].sum()

        sustainability_metrics["high_sustainability_percentage"] = (
            sustainability_metrics["high_sustainability_area_ha"] / total_area * 100
            if total_area > 0
            else 0
        )

        sustainability_metrics["medium_sustainability_percentage"] = (
            sustainability_metrics["medium_sustainability_area_ha"] / total_area * 100
            if total_area > 0
            else 0
        )

        sustainability_metrics["low_sustainability_percentage"] = (
            sustainability_metrics["low_sustainability_area_ha"] / total_area * 100
            if total_area > 0
            else 0
        )

        # Store results
        self.metrics["sustainability_index"] = sustainability_metrics
        return sustainability_metrics

    def plot_sustainability_metrics(
        self, ax: Any = None, metric_type: str = "sustainability_index"
    ) -> Any:
        """
        Plot sustainability metrics on a map.

        Args:
            ax: Optional matplotlib axis for plotting
            metric_type: Type of metric to plot ('sustainability_index', 'carbon', 'water', 'soil', 'biodiversity')

        Returns:
            The plot axis

        Raises:
            ValueError: If metrics or field data are not available
        """
        if (
            metric_type == "sustainability_index"
            and "sustainability_index" not in self.metrics
        ):
            raise ValueError(
                "Sustainability index not calculated. Run calculate_sustainability_index first"
            )

        # Map short metric_type names to actual self.metrics keys
        _metric_key_map = {
            "carbon": "carbon_sequestration",
            "water": "water_usage",
            "soil": "soil_health",
            "biodiversity": "biodiversity",
        }
        if metric_type != "sustainability_index":
            full_key = _metric_key_map.get(metric_type, metric_type)
            if full_key not in self.metrics:
                raise ValueError(
                    f"Metric {metric_type} not found in calculated metrics"
                )

        # Get data for plotting
        if metric_type == "sustainability_index":
            plot_data = self.metrics["sustainability_index"]["field_data"]
            plot_column = "sustainability_index"
            title = "Sustainability Index"
            cmap = "RdYlGn"
        elif metric_type == "carbon":
            plot_data = self.metrics["carbon_sequestration"]["field_data"]
            plot_column = "carbon_score"
            title = "Carbon Sequestration Score"
            cmap = "Greens"
        elif metric_type == "water":
            plot_data = self.metrics["water_usage"]["field_data"]
            plot_column = "water_score"
            title = "Water Efficiency Score"
            cmap = "Blues"
        elif metric_type == "soil":
            plot_data = self.metrics["soil_health"]["field_data"]
            plot_column = "soil_score"
            title = "Soil Health Score"
            cmap = "YlOrBr"
        elif metric_type == "biodiversity":
            plot_data = self.metrics["biodiversity"]["field_data"]
            plot_column = "biodiv_score"
            title = "Biodiversity Score"
            cmap = "viridis"

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the data
        plot_data.plot(
            column=plot_column,
            ax=ax,
            legend=True,
            cmap=cmap,
            vmin=0,
            vmax=10,
            legend_kwds={"label": f"{title} (0-10 scale)"},
        )

        ax.set_title(title)
        ax.set_axis_off()

        # Continuous GeoPandas plots expose a colorbar rather than a legend.
        # Add one explicit, labeled handle so exported figures retain a
        # discoverable metric key without invoking an empty legend warning.
        if ax.get_legend() is None:
            ax.legend(
                handles=[Line2D([], [], color="none", label=f"{title} (0-10 scale)")],
                loc="upper right",
            )

        return ax
