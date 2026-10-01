"""
Datashader-based visualization for Cascadia H3 data.
Provides efficient rendering of large geospatial datasets.
"""

import datashader as ds
import pandas as pd
import colorcet
import holoviews as hv
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class CascadiaDatashaderVisualizer:
    """
    Efficient visualization of Cascadia H3 data using Datashader.
    Handles large datasets with minimal memory usage and maximum performance.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def prepare_h3_dataframe(self, unified_data: dict, redevelopment_scores: dict) -> pd.DataFrame:
        """
        Convert H3 unified data to pandas DataFrame for Datashader processing.

        Args:
            unified_data: H3 hexagon data from unified backend
            redevelopment_scores: Redevelopment potential scores

        Returns:
            DataFrame with longitude, latitude, and analysis data
        """
        logger.info("Preparing H3 data for Datashader visualization...")

        # Import H3 utilities
        from h3 import cell_to_latlng

        data_rows = []

        for h3_id, hex_data in unified_data.items():
            try:
                # Get centroid coordinates
                lat, lng = cell_to_latlng(h3_id)

                # Get redevelopment score
                score_data = redevelopment_scores.get(h3_id, {})
                composite_score = (
                    score_data.get("composite_score", 0.0) if isinstance(score_data, dict) else 0.0
                )

                # Extract module data
                module_scores = {}
                for module_name in ["zoning", "current_use", "ownership", "improvements"]:
                    module_data = hex_data.get(module_name, {})
                    if isinstance(module_data, dict):
                        module_scores[f"{module_name}_score"] = module_data.get("score", 0.0)
                    else:
                        module_scores[f"{module_name}_score"] = 0.0

                row = {
                    "h3_id": h3_id,
                    "longitude": lng,
                    "latitude": lat,
                    "redevelopment_score": composite_score,
                    **module_scores,
                }
                data_rows.append(row)

            except Exception as e:
                logger.warning(f"Could not process H3 cell {h3_id}: {e}")
                continue

        df = pd.DataFrame(data_rows)
        logger.info(f"Prepared DataFrame with {len(df)} H3 cells")
        return df

    def create_redevelopment_heatmap(
        self, df: pd.DataFrame, plot_width: int = 1200, plot_height: int = 800
    ) -> hv.Image:
        """
        Create an efficient heatmap of redevelopment potential scores.

        Args:
            df: DataFrame with H3 data
            plot_width: Width of the plot in pixels
            plot_height: Height of the plot in pixels

        Returns:
            HoloViews Image object
        """
        logger.info("Creating redevelopment potential heatmap...")

        # Create canvas for efficient rendering
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

        # Aggregate points by redevelopment score
        agg = cvs.points(df, "longitude", "latitude", ds.mean("redevelopment_score"))

        # Create heatmap with logarithmic color scaling
        img = ds.tf.shade(agg, cmap=colorcet.fire, how="log")

        # Convert to HoloViews for interactive display
        hv_img = hv.Image(img)
        hv_img.opts(
            title="Cascadia Agricultural Redevelopment Potential",
            xlabel="Longitude",
            ylabel="Latitude",
            colorbar=True,
            cmap="fire",
        )

        logger.info("Redevelopment heatmap created successfully")
        return hv_img

    def create_module_coverage_plot(
        self, df: pd.DataFrame, module_name: str, plot_width: int = 1200, plot_height: int = 800
    ) -> hv.Image:
        """
        Create coverage plot for a specific module.

        Args:
            df: DataFrame with H3 data
            module_name: Name of the module to visualize
            plot_width: Width of the plot in pixels
            plot_height: Height of the plot in pixels

        Returns:
            HoloViews Image object
        """
        logger.info(f"Creating {module_name} coverage plot...")

        # Filter for cells with data
        score_col = f"{module_name}_score"
        if score_col not in df.columns:
            logger.warning(f"Column {score_col} not found in DataFrame")
            return None

        # Create canvas
        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

        # Aggregate by module score
        agg = cvs.points(df, "longitude", "latitude", ds.mean(score_col))

        # Create visualization
        img = ds.tf.shade(agg, cmap=colorcet.viridis, how="log")

        # Convert to HoloViews
        hv_img = hv.Image(img)
        hv_img.opts(
            title=f"{module_name.replace('_', ' ').title()} Coverage",
            xlabel="Longitude",
            ylabel="Latitude",
            colorbar=True,
            cmap="viridis",
        )

        logger.info(f"{module_name} coverage plot created successfully")
        return hv_img

    def create_comprehensive_dashboard(self, unified_data: dict, redevelopment_scores: dict) -> str:
        """
        Create a comprehensive Datashader-based dashboard.

        Args:
            unified_data: H3 unified data
            redevelopment_scores: Redevelopment scores

        Returns:
            Path to the generated HTML dashboard
        """
        logger.info("Creating comprehensive Datashader dashboard...")

        # Prepare data
        df = self.prepare_h3_dataframe(unified_data, redevelopment_scores)

        # Create main redevelopment heatmap
        redevelopment_plot = self.create_redevelopment_heatmap(df)

        # Create module coverage plots
        module_plots = {}
        for module in ["zoning", "current_use", "ownership", "improvements"]:
            plot = self.create_module_coverage_plot(df, module)
            if plot is not None:
                module_plots[module] = plot

        # Combine plots into dashboard
        dashboard = hv.Layout(
            [redevelopment_plot, hv.Layout(list(module_plots.values())).cols(2)]
        ).opts(title="Cascadia Agricultural Analysis Dashboard")

        # Save to HTML
        output_path = self.output_dir / "cascadia_datashader_dashboard.html"
        hv.save(dashboard, output_path, fmt="html")

        logger.info(f"Datashader dashboard saved to: {output_path}")
        return str(output_path)

    def create_lightweight_json_export(self, unified_data: dict, redevelopment_scores: dict) -> str:
        """
        Create a lightweight JSON export for web-based visualization.

        Args:
            unified_data: H3 unified data
            redevelopment_scores: Redevelopment scores

        Returns:
            Path to the JSON file
        """
        logger.info("Creating lightweight JSON export...")

        # Sample data for performance (every 10th cell)
        sampled_data = {}
        count = 0

        for h3_id, hex_data in unified_data.items():
            if count % 10 == 0:  # Sample every 10th cell
                try:
                    from h3 import cell_to_latlng

                    lat, lng = cell_to_latlng(h3_id)

                    score_data = redevelopment_scores.get(h3_id, {})
                    composite_score = (
                        score_data.get("composite_score", 0.0)
                        if isinstance(score_data, dict)
                        else 0.0
                    )

                    sampled_data[h3_id] = {
                        "longitude": lng,
                        "latitude": lat,
                        "redevelopment_score": composite_score,
                        "modules": {
                            module: hex_data.get(module, {}).get("score", 0.0)
                            for module in ["zoning", "current_use", "ownership", "improvements"]
                        },
                    }
                except Exception as e:
                    logger.warning(f"Could not process H3 cell {h3_id}: {e}")

            count += 1

        # Save to JSON
        output_path = self.output_dir / "cascadia_lightweight_data.json"
        with open(output_path, "w") as f:
            json.dump(sampled_data, f, indent=2)

        logger.info(f"Lightweight JSON export saved to: {output_path}")
        return str(output_path)


def create_datashader_visualization(backend, output_dir: Path) -> dict[str, str]:
    """
    Create efficient Datashader visualizations for Cascadia data.

    Args:
        backend: Unified backend with processed data
        output_dir: Output directory for visualizations

    Returns:
        Dictionary with paths to generated visualizations
    """
    logger.info("Creating Datashader visualizations...")

    # Initialize visualizer
    visualizer = CascadiaDatashaderVisualizer(output_dir)

    # Get data from backend
    unified_data = backend.unified_data
    redevelopment_scores = backend.calculate_agricultural_redevelopment_potential()

    # Create visualizations
    results = {}

    try:
        # Create comprehensive dashboard
        dashboard_path = visualizer.create_comprehensive_dashboard(
            unified_data, redevelopment_scores
        )
        results["datashader_dashboard"] = dashboard_path

        # Create lightweight JSON export
        json_path = visualizer.create_lightweight_json_export(unified_data, redevelopment_scores)
        results["lightweight_json"] = json_path

        logger.info("Datashader visualizations created successfully")

    except Exception as e:
        logger.error(f"Failed to create Datashader visualizations: {e}")
        results["error"] = str(e)

    return results
