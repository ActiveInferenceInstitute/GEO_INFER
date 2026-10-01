"""
Field boundary management functionality for agricultural applications.
"""

from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from rasterio.features import shapes
from pyproj import CRS

# WGS 84 / Equal Area (EPSG:6933): true equal-area projection for computing
# field areas in hectares without the scale distortion of Web Mercator.
EQUAL_AREA_CRS = CRS.from_epsg(6933)


class FieldBoundaryManager:
    """
    Manages agricultural field boundaries and their properties.

    This class provides functionality for creating, editing, and analyzing
    field boundaries for agricultural applications.

    Attributes:
        fields: GeoDataFrame containing field boundaries
        crs: Coordinate reference system for spatial data
    """

    def __init__(self, fields: gpd.GeoDataFrame | None = None, crs: str = "EPSG:4326"):
        """
        Initialize the field boundary manager.

        Args:
            fields: Optional GeoDataFrame containing field boundaries
            crs: Coordinate reference system for spatial data
        """
        if fields is None:
            self.fields = gpd.GeoDataFrame(
                columns=["field_id", "name", "area_ha", "crop_type"],
                geometry=[],
                crs=crs,
            )
        else:
            self.fields = fields.copy()
            # Ensure the GeoDataFrame has the required CRS
            if self.fields.crs is None:
                self.fields.set_crs(crs, inplace=True)
            elif str(self.fields.crs) != str(crs):
                self.fields = self.fields.to_crs(crs)

        # Ensure required columns exist
        for col in ["field_id", "name", "area_ha", "crop_type"]:
            if col not in self.fields.columns:
                self.fields[col] = None

        # Calculate areas if not already present
        if self.fields["area_ha"].isna().any():
            self._calculate_areas()

    def add_field(
        self,
        geometry: Polygon | MultiPolygon,
        field_id: str | None = None,
        name: str | None = None,
        crop_type: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a new field boundary.

        Args:
            geometry: Shapely Polygon or MultiPolygon representing the field boundary
            field_id: Optional ID for the field (generated if not provided)
            name: Optional name for the field
            crop_type: Optional crop type grown in the field
            attributes: Optional dictionary of additional attributes

        Returns:
            The ID of the newly added field

        Raises:
            ValueError: If geometry is not a valid Polygon or MultiPolygon
        """
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise ValueError("Field geometry must be a Polygon or MultiPolygon")

        # Generate field ID if not provided
        if field_id is None:
            field_id = f"field_{len(self.fields) + 1}"

        # Create new field entry
        new_field = {
            "field_id": field_id,
            "name": name,
            "crop_type": crop_type,
            "geometry": geometry,
        }

        # Add additional attributes if provided
        if attributes:
            for key, value in attributes.items():
                if key not in new_field:
                    new_field[key] = value

        # Append to GeoDataFrame
        self.fields = pd.concat(
            [self.fields, gpd.GeoDataFrame([new_field], crs=self.fields.crs)],
            ignore_index=True,
        )

        # Calculate area for the new field
        self._calculate_areas()

        return field_id

    def remove_field(self, field_id: str) -> bool:
        """
        Remove a field by its ID.

        Args:
            field_id: ID of the field to remove

        Returns:
            True if field was removed, False if not found
        """
        if field_id not in self.fields["field_id"].values:
            return False

        self.fields = self.fields[self.fields["field_id"] != field_id].reset_index(
            drop=True
        )
        return True

    def update_field(
        self,
        field_id: str,
        geometry: Polygon | MultiPolygon | None = None,
        name: str | None = None,
        crop_type: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a field's properties.

        Args:
            field_id: ID of the field to update
            geometry: Optional new geometry for the field
            name: Optional new name for the field
            crop_type: Optional new crop type
            attributes: Optional dictionary of additional attributes to update

        Returns:
            True if field was updated, False if not found

        Raises:
            ValueError: If geometry is not a valid Polygon or MultiPolygon
        """
        if field_id not in self.fields["field_id"].values:
            return False

        # Get index of the field
        idx = self.fields[self.fields["field_id"] == field_id].index[0]

        # Update geometry if provided
        if geometry is not None:
            if not isinstance(geometry, (Polygon, MultiPolygon)):
                raise ValueError("Field geometry must be a Polygon or MultiPolygon")
            self.fields.loc[idx, "geometry"] = geometry

        # Update name if provided
        if name is not None:
            self.fields.loc[idx, "name"] = name

        # Update crop type if provided
        if crop_type is not None:
            self.fields.loc[idx, "crop_type"] = crop_type

        # Update additional attributes if provided
        if attributes:
            for key, value in attributes.items():
                self.fields.loc[idx, key] = value

        # Recalculate area if geometry changed
        if geometry is not None:
            self._calculate_areas()

        return True

    def get_field(self, field_id: str) -> gpd.GeoSeries | None:
        """
        Get a field by its ID.

        Args:
            field_id: ID of the field to retrieve

        Returns:
            GeoSeries containing the field data, or None if not found
        """
        if field_id not in self.fields["field_id"].values:
            return None

        return self.fields[self.fields["field_id"] == field_id].iloc[0]

    def get_fields_by_crop(self, crop_type: str) -> gpd.GeoDataFrame:
        """
        Get all fields with a specific crop type.

        Args:
            crop_type: Crop type to filter by

        Returns:
            GeoDataFrame containing fields with the specified crop type
        """
        return self.fields[self.fields["crop_type"] == crop_type].copy()

    def get_neighboring_fields(
        self, field_id: str, buffer_distance: float = 10.0
    ) -> gpd.GeoDataFrame:
        """
        Get fields that neighbor the specified field.

        Args:
            field_id: ID of the field to find neighbors for
            buffer_distance: Distance in meters to consider for neighboring

        Returns:
            GeoDataFrame containing neighboring fields

        Raises:
            ValueError: If field_id is not found
        """
        field = self.get_field(field_id)
        if field is None:
            raise ValueError(f"Field with ID {field_id} not found")

        # Interpret buffer_distance as meters: buffer in a local metric CRS
        # (UTM) when the fields are stored in a geographic CRS. Buffering a
        # geographic geometry would treat the distance as degrees (~111 km/m).
        metric_crs = (
            self.fields.estimate_utm_crs()
            if self.fields.crs is not None and self.fields.crs.is_geographic
            else None
        )
        if metric_crs is not None:
            metric_fields = self.fields.to_crs(metric_crs)
            metric_field = metric_fields[metric_fields["field_id"] == field_id]
            buffer_geom = metric_field.geometry.iloc[0].buffer(buffer_distance)
            mask = (
                metric_fields["field_id"] != field_id
            ) & metric_fields.geometry.intersects(buffer_geom)
            # Return neighbors in the manager's original CRS
            return self.fields.loc[metric_fields.index[mask]].copy()

        # Projected (metric) CRS: buffer distance is in CRS units
        buffer_geom = field.geometry.buffer(buffer_distance)

        # Find fields that intersect with the buffer (excluding the original field)
        neighbors = self.fields[
            (self.fields["field_id"] != field_id)
            & (self.fields.geometry.intersects(buffer_geom))
        ].copy()

        return neighbors

    def extract_fields_from_raster(
        self,
        raster_path: str,
        value_field: str | None = None,
        min_area: float = 0.1,
        simplify_tolerance: float | None = None,
    ) -> int:
        """
        Extract field boundaries from a classified raster image.

        Args:
            raster_path: Path to the raster file
            value_field: Raster band value representing fields
            min_area: Minimum area in hectares for a valid field
            simplify_tolerance: Optional tolerance for boundary simplification

        Returns:
            Number of fields extracted

        Raises:
            ValueError: If raster file cannot be opened
        """
        try:
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)
                raster_mask = np.ones_like(raster_data, dtype=bool)

                # If specific value provided, create mask for that value
                if value_field is not None:
                    raster_mask = raster_data == value_field

                # Get polygons from raster
                results = (
                    {"properties": {"raster_val": v}, "geometry": s}
                    for i, (s, v) in enumerate(
                        shapes(raster_data, mask=raster_mask, transform=src.transform)
                    )
                )

                # Convert to GeoDataFrame
                gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)

                # Convert area units and filter by minimum area, computed in a
                # true equal-area projection (EPSG:6933) so hectares are not
                # distorted by Web Mercator scale error.
                area_factor = 0.0001  # Convert m² to hectares
                equal_area_gdf = (
                    gdf.to_crs(EQUAL_AREA_CRS) if gdf.crs is not None else gdf
                )
                gdf["area_ha"] = equal_area_gdf.geometry.area * area_factor
                gdf = gdf[gdf["area_ha"] >= min_area]

                # Simplify geometries if tolerance is provided
                if simplify_tolerance is not None:
                    gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance)

                # Add to fields
                orig_count = len(self.fields)
                for idx, row in gdf.iterrows():
                    self.add_field(
                        geometry=row.geometry,
                        field_id=f"field_r_{idx + 1}",
                        name=f"Field R{idx + 1}",
                        attributes={"source": "raster_extraction"},
                    )

                return len(self.fields) - orig_count

        except Exception as e:
            raise ValueError(f"Error extracting fields from raster: {str(e)}")

    def export_to_file(self, output_path: str, driver: str = "ESRI Shapefile") -> None:
        """
        Export field boundaries to a file.

        Args:
            output_path: Path to output file
            driver: OGR driver name for output format

        Raises:
            ValueError: If export fails
        """
        try:
            export_fields = self.fields.copy()
            # Shapefile numeric fields have a narrow, driver-defined width.
            # Preserve large projected areas without pyogrio truncation by
            # serializing only this export-only attribute as text.
            if driver == "ESRI Shapefile" and "area_ha" in export_fields:
                export_fields["area_ha"] = export_fields["area_ha"].map(
                    lambda value: f"{float(value):.12g}"
                )
            export_fields.to_file(output_path, driver=driver)
        except Exception as e:
            raise ValueError(f"Error exporting fields: {str(e)}")

    def _calculate_areas(self) -> None:
        """Calculate area in hectares for all fields."""
        # Compute areas in a true equal-area projection (EPSG:6933). Web
        # Mercator inflates areas by 1/cos(latitude), so hectares computed
        # there are wrong away from the equator.
        if self.fields.crs is not None:
            area_gdf = self.fields.to_crs(EQUAL_AREA_CRS)
            areas = area_gdf.geometry.area / 10000  # Convert m² to hectares
        else:
            # No CRS set: interpret coordinates as planar meters
            areas = self.fields.geometry.area / 10000

        self.fields["area_ha"] = areas.values
