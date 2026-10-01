"""
Agricultural analysis core functionality providing methods for analyzing agricultural data.
"""

from typing import Any
import pandas as pd
import geopandas as gpd
from datetime import datetime

from geo_infer_ag.models.base import AgricultureModel


class AgriculturalAnalysis:
    """
    Core class for performing agricultural analysis and modeling.

    This class provides the main functionality for agricultural data analysis,
    including field-level assessments, crop monitoring, and prediction.

    Attributes:
        model: The agricultural model to use for analysis
        config: Configuration settings for the analysis
    """

    def __init__(
        self, model: AgricultureModel, config: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize the agricultural analysis with a model.

        Args:
            model: Agricultural model instance for analysis
            config: Optional configuration parameters
        """
        self.model = model
        self.config = config or {}
        self.results: AgriculturalResults | None = None

    def run(
        self,
        field_data: gpd.GeoDataFrame,
        weather_data: pd.DataFrame | None = None,
        soil_data: gpd.GeoDataFrame | None = None,
        management_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> "AgriculturalResults":
        """
        Run agricultural analysis using the provided data sources.

        Args:
            field_data: GeoDataFrame containing field boundaries and attributes
            weather_data: Time series weather data
            soil_data: Spatial soil properties data
            management_data: Management practices data
            **kwargs: Additional data sources or parameters

        Returns:
            AgriculturalResults object containing analysis results
        """
        # Validate inputs
        self._validate_inputs(field_data, weather_data, soil_data, management_data)

        # Prepare data for model
        prepared_data = self._prepare_data(
            field_data, weather_data, soil_data, management_data, **kwargs
        )

        # Run the agricultural model
        model_results = self.model.predict(prepared_data)

        # Process and format results
        self.results = AgriculturalResults(
            model_results=model_results,
            field_data=field_data,
            model_metadata=self.model.metadata,
        )

        return self.results

    def _validate_inputs(
        self,
        field_data: gpd.GeoDataFrame,
        weather_data: pd.DataFrame | None = None,
        soil_data: gpd.GeoDataFrame | None = None,
        management_data: pd.DataFrame | None = None,
    ) -> None:
        """
        Validate input data for correctness and compatibility.

        Args:
            field_data: GeoDataFrame containing field boundaries
            weather_data: Weather time series data
            soil_data: Soil properties spatial data
            management_data: Management practices data

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Check field data has geometry
        if not isinstance(field_data, gpd.GeoDataFrame):
            raise ValueError("Field data must be a GeoDataFrame with geometry")

        # Check model compatibility with inputs
        required_inputs = self.model.required_inputs

        if "weather" in required_inputs and weather_data is None:
            raise ValueError(f"Model {self.model.name} requires weather data")

        if "soil" in required_inputs and soil_data is None:
            raise ValueError(f"Model {self.model.name} requires soil data")

        if "management" in required_inputs and management_data is None:
            raise ValueError(f"Model {self.model.name} requires management data")

    def _prepare_data(
        self,
        field_data: gpd.GeoDataFrame,
        weather_data: pd.DataFrame | None = None,
        soil_data: gpd.GeoDataFrame | None = None,
        management_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare and integrate data sources for model input.

        Args:
            field_data: GeoDataFrame containing field boundaries
            weather_data: Weather time series data
            soil_data: Soil properties spatial data
            management_data: Management practices data
            **kwargs: Additional data sources

        Returns:
            Prepared data dictionary for model input
        """
        prepared_data = {
            "field_data": field_data,
        }

        # Add optional data if provided
        if weather_data is not None:
            prepared_data["weather_data"] = weather_data

        if soil_data is not None:
            prepared_data["soil_data"] = soil_data

        if management_data is not None:
            prepared_data["management_data"] = management_data

        # Add any additional data from kwargs
        prepared_data.update(kwargs)

        return prepared_data


class AgriculturalResults:
    """
    Container for agricultural analysis results with visualization methods.

    Attributes:
        results: Dictionary of result values and metrics
        field_data: Field boundary data with results attached
        metadata: Metadata about the analysis
    """

    def __init__(
        self,
        model_results: dict[str, Any],
        field_data: gpd.GeoDataFrame,
        model_metadata: dict[str, Any],
    ) -> None:
        """
        Initialize results container.

        Args:
            model_results: Results from the agricultural model
            field_data: Original field data used in analysis
            model_metadata: Metadata about the model used
        """
        self.results = model_results
        self.field_data = self._merge_results_with_field_data(field_data, model_results)
        self.metadata: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": model_metadata,
        }

    def _merge_results_with_field_data(
        self, field_data: gpd.GeoDataFrame, model_results: dict[str, Any]
    ) -> gpd.GeoDataFrame:
        """
        Merge model results with field geometries.

        Args:
            field_data: Original field GeoDataFrame
            model_results: Model output results

        Returns:
            GeoDataFrame with results attached to fields
        """
        # Create a copy to avoid modifying the original
        merged_data = field_data.copy()

        # Skip if spatial_results not in model_results
        if "spatial_results" not in model_results:
            return merged_data

        # Add each result column to the field data
        for col, values in model_results["spatial_results"].items():
            if len(values) == len(merged_data):
                merged_data[col] = values

        return merged_data

    def get_metric(self, name: str) -> Any:
        """
        Get a specific metric from the results.

        Args:
            name: Name of the metric to retrieve

        Returns:
            The metric value

        Raises:
            KeyError: If the metric doesn't exist
        """
        if name not in self.results:
            raise KeyError(f"Metric '{name}' not found in results")

        return self.results[name]

    def plot_spatial_distribution(
        self,
        variable: str,
        cmap: str = "viridis",
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot spatial distribution of a result variable.

        Args:
            variable: Name of variable to plot
            cmap: Colormap to use
            title: Plot title
            **kwargs: Additional arguments for plotting

        Returns:
            The plot axis

        Raises:
            ValueError: If variable doesn't exist in results
        """
        if variable not in self.field_data.columns:
            raise ValueError(f"Variable '{variable}' not found in spatial results")

        ax = self.field_data.plot(column=variable, cmap=cmap, legend=True, **kwargs)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Spatial Distribution of {variable}")

        return ax

    def summary(self) -> dict[str, Any]:
        """
        Generate a summary of the analysis results.

        Returns:
            Dictionary with summary statistics
        """
        summary_dict = {
            "timestamp": self.metadata["timestamp"],
            "model_name": self.metadata["model"].get("name", "Unknown"),
            "field_count": len(self.field_data),
        }

        # Add summary statistics for numerical result columns
        for col in self.field_data.select_dtypes(include=["number"]).columns:
            if col in self.results.get("spatial_results", {}):
                summary_dict[f"{col}_mean"] = self.field_data[col].mean()
                summary_dict[f"{col}_std"] = self.field_data[col].std()
                summary_dict[f"{col}_min"] = self.field_data[col].min()
                summary_dict[f"{col}_max"] = self.field_data[col].max()

        return summary_dict
