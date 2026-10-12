"""
Geospatial data anonymization techniques.

This module provides implementation of various anonymization techniques
for geospatial data to protect privacy while maintaining utility.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import h3


class GeospatialAnonymizer:
    """
    Provides methods for anonymizing geospatial data while preserving utility.
    """

    def __init__(self, seed: int | np.random.Generator | None = None) -> None:
        """
        Initialize the anonymizer.

        Args:
            seed: Integer seed or ``np.random.Generator`` used for all random
                draws, for reproducible anonymization.
        """
        self.rng = np.random.default_rng(seed)

    def location_perturbation(
        self,
        gdf: gpd.GeoDataFrame,
        epsilon: float = 100.0,
        geometry_col: str = "geometry",
    ) -> gpd.GeoDataFrame:
        """
        Apply random perturbation to point locations.

        Args:
            gdf: GeoDataFrame with point geometries to anonymize
            epsilon: Maximum displacement distance in meters
            geometry_col: Name of the geometry column

        Returns:
            GeoDataFrame with perturbed geometries

        Raises:
            ValueError: If any geometry is not a Point, or if the GeoDataFrame
                declares a CRS other than EPSG:4326.

        Note:
            Displacement assumes EPSG:4326 geographic coordinates: meters are
            converted to degrees with a flat 111,000 m/deg factor. This is an
            approximation whose accuracy degrades away from the equator;
            projected CRS input is rejected rather than silently mis-scaled.
        """
        if not all(isinstance(geom, Point) for geom in gdf[geometry_col]):
            raise ValueError("All geometries must be Point objects")

        crs = getattr(gdf, "crs", None)
        if crs is not None and not crs.equals("EPSG:4326"):
            raise ValueError(
                "location_perturbation assumes EPSG:4326 coordinates; "
                f"got {crs.to_string()}. Reproject before anonymizing."
            )

        result = gdf.copy()
        for idx, row in result.iterrows():
            point = row[geometry_col]
            # Generate random angle and distance
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = self.rng.uniform(0, epsilon)

            # Convert displacement to degrees uniformly
            # 111,000 meters ~= 1 degree at the equator
            # Apply displacement in a coordinate space where both axes
            # represent equal metric distances, then convert back to degrees
            meters_per_deg = 111000.0
            dx_deg = (distance * np.cos(angle)) / meters_per_deg
            dy_deg = (distance * np.sin(angle)) / meters_per_deg

            # Create new point
            new_point = Point(point.x + dx_deg, point.y + dy_deg)
            result.loc[idx, geometry_col] = new_point

        return result

    def spatial_k_anonymity(
        self,
        gdf: gpd.GeoDataFrame,
        k: int = 5,
        h3_resolution: int = 9,
        geometry_col: str = "geometry",
    ) -> gpd.GeoDataFrame:
        """
        Apply spatial k-anonymity by aggregating points into H3 cells.

        Args:
            gdf: GeoDataFrame with point geometries to anonymize
            k: Minimum number of points required in each H3 cell
            h3_resolution: H3 grid resolution (0-15, where higher is more precise)
            geometry_col: Name of the geometry column

        Returns:
            GeoDataFrame with k-anonymized data
        """
        if not all(isinstance(geom, Point) for geom in gdf[geometry_col]):
            raise ValueError("All geometries must be Point objects")

        result = gdf.copy()
        current_resolution = h3_resolution

        # Iteratively reduce resolution until all cells meet the k threshold
        # or we reach resolution 0
        while current_resolution >= 0:
            # Add H3 cell IDs at current resolution
            result["h3_cell"] = [
                h3.latlng_to_cell(geom.y, geom.x, current_resolution)
                for geom in result[geometry_col]
            ]

            # Count records per cell
            cell_counts = result["h3_cell"].value_counts()

            # Check if any cell is below k
            small_cells = cell_counts[cell_counts < k].index.tolist()

            if not small_cells:
                # All cells meet the k threshold
                break

            if current_resolution == 0:
                # Cannot reduce further - merge remaining small cells into the
                # largest cell so we still group them
                largest_cell = cell_counts.idxmax()
                for sc in small_cells:
                    result.loc[result["h3_cell"] == sc, "h3_cell"] = largest_cell
                break

            # Reduce resolution and retry
            current_resolution -= 1
            result = result.drop(columns=["h3_cell"])

        # Replace coordinates with cell centroids
        for cell_id in result["h3_cell"].unique():
            cell_center = h3.cell_to_latlng(cell_id)
            result.loc[result["h3_cell"] == cell_id, geometry_col] = Point(
                cell_center[1], cell_center[0]
            )

        # Drop H3 cell column
        result = result.drop(columns=["h3_cell"])

        return result

    def geographic_masking(
        self,
        gdf: gpd.GeoDataFrame,
        attribute_cols: list[str] | None = None,
        admin_boundaries: gpd.GeoDataFrame | None = None,
        admin_id_col: str = "admin_id",
        geometry_col: str = "geometry",
    ) -> gpd.GeoDataFrame:
        """
        Apply geographic masking by aggregating data to administrative boundaries.

        Args:
            gdf: GeoDataFrame with point geometries to anonymize
            attribute_cols: List of columns with attributes to aggregate
            admin_boundaries: GeoDataFrame with administrative boundaries
            admin_id_col: Column name for administrative area identifier
            geometry_col: Name of the geometry column

        Returns:
            GeoDataFrame with geographically masked data
        """
        if attribute_cols is None or admin_boundaries is None:
            raise ValueError("attribute_cols and admin_boundaries must be provided")

        # Spatial join: assign each point to the admin boundary it falls within
        joined = gpd.sjoin(gdf, admin_boundaries, how="inner", predicate="within")

        # Aggregate data by admin area index
        aggregated = {}
        for col in attribute_cols:
            if pd.api.types.is_numeric_dtype(gdf[col]):
                aggregated[col] = joined.groupby("index_right")[col].mean()
            else:
                aggregated[col] = joined.groupby("index_right")[col].agg(
                    lambda x: x.value_counts().index[0] if len(x) > 0 else None
                )

        agg_df = pd.DataFrame(aggregated)

        # Build result from admin_boundaries so ALL boundaries appear,
        # even those with no matching points (they will get NaN attributes).
        # Only keep geometry and the requested attribute columns.
        agg_gdf = gpd.GeoDataFrame(
            {geometry_col: admin_boundaries[geometry_col]},
            geometry=geometry_col,
            crs=admin_boundaries.crs,
        )
        for col in attribute_cols:
            if col in agg_df.columns:
                agg_gdf[col] = agg_df[col].reindex(admin_boundaries.index).values
            else:
                agg_gdf[col] = np.nan

        return agg_gdf
