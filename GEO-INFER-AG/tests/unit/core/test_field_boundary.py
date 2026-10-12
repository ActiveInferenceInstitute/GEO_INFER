"""Unit tests for the Field Boundary Manager functionality."""

import os
import pytest
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import tempfile

from geo_infer_ag.core.field_boundary import FieldBoundaryManager


class TestFieldBoundaryManager:
    """Test suite for FieldBoundaryManager class."""

    def test_initialization(self):
        """Test initialization of FieldBoundaryManager."""
        # Test empty initialization
        fbm = FieldBoundaryManager()
        assert isinstance(fbm.fields, gpd.GeoDataFrame)
        assert len(fbm.fields) == 0
        assert "field_id" in fbm.fields.columns
        assert "name" in fbm.fields.columns
        assert "area_ha" in fbm.fields.columns
        assert "crop_type" in fbm.fields.columns
        assert fbm.fields.crs == "EPSG:4326"

        # Test initialization with existing GeoDataFrame
        sample_gdf = gpd.GeoDataFrame(
            {
                "field_id": ["field_1"],
                "name": ["Test Field"],
                "geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])],
            },
            crs="EPSG:4326",
        )
        fbm = FieldBoundaryManager(fields=sample_gdf)
        assert len(fbm.fields) == 1
        assert "area_ha" in fbm.fields.columns
        assert fbm.fields["area_ha"].iloc[0] > 0

        # Test initialization with different CRS
        sample_gdf = gpd.GeoDataFrame(
            {
                "field_id": ["field_1"],
                "name": ["Test Field"],
                "geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])],
            },
            crs="EPSG:3857",
        )
        fbm = FieldBoundaryManager(fields=sample_gdf, crs="EPSG:4326")
        assert str(fbm.fields.crs) == "EPSG:4326"

    def test_add_field(self):
        """Test adding a field."""
        fbm = FieldBoundaryManager()

        # Test adding a field with minimal parameters
        field_id = fbm.add_field(geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]))
        assert len(fbm.fields) == 1
        assert fbm.fields["field_id"].iloc[0] == field_id
        assert fbm.fields["area_ha"].iloc[0] > 0

        # Test adding a field with all parameters
        field_id = fbm.add_field(
            geometry=Polygon([(20, 0), (20, 10), (30, 10), (30, 0)]),
            field_id="custom_id",
            name="Custom Field",
            crop_type="wheat",
            attributes={"soil_type": "clay", "irrigated": True},
        )
        assert len(fbm.fields) == 2
        assert field_id == "custom_id"
        assert fbm.fields["name"].iloc[1] == "Custom Field"
        assert fbm.fields["crop_type"].iloc[1] == "wheat"
        assert fbm.fields["soil_type"].iloc[1] == "clay"
        assert bool(fbm.fields["irrigated"].iloc[1]) is True

        # Test adding a MultiPolygon
        multi_geom = MultiPolygon(
            [
                Polygon([(40, 0), (40, 5), (45, 5), (45, 0)]),
                Polygon([(40, 10), (40, 15), (45, 15), (45, 10)]),
            ]
        )
        field_id = fbm.add_field(geometry=multi_geom, name="Multi Field")
        assert len(fbm.fields) == 3
        assert isinstance(fbm.fields.geometry.iloc[2], MultiPolygon)

        # Test invalid geometry
        with pytest.raises(ValueError):
            fbm.add_field(geometry="not a geometry")

    def test_remove_field(self):
        """Test removing a field."""
        fbm = FieldBoundaryManager()

        # Add some fields
        fbm.add_field(
            geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]), field_id="field_1"
        )
        fbm.add_field(
            geometry=Polygon([(20, 0), (20, 10), (30, 10), (30, 0)]), field_id="field_2"
        )

        # Test removing a field
        assert len(fbm.fields) == 2
        assert fbm.remove_field("field_1") is True
        assert len(fbm.fields) == 1
        assert "field_1" not in fbm.fields["field_id"].values

        # Test removing nonexistent field
        assert fbm.remove_field("nonexistent") is False
        assert len(fbm.fields) == 1

    def test_update_field(self):
        """Test updating a field."""
        fbm = FieldBoundaryManager()

        # Add a field
        fbm.add_field(
            geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            field_id="field_1",
            name="Original Name",
            crop_type="corn",
        )

        # Test updating field properties
        assert (
            fbm.update_field(
                field_id="field_1",
                name="Updated Name",
                crop_type="wheat",
                attributes={"status": "active"},
            )
            is True
        )

        field = fbm.get_field("field_1")
        assert field["name"] == "Updated Name"
        assert field["crop_type"] == "wheat"
        assert field["status"] == "active"

        # Test updating geometry
        original_area = field["area_ha"]
        new_geom = Polygon([(0, 0), (0, 20), (20, 20), (20, 0)])

        assert fbm.update_field(field_id="field_1", geometry=new_geom) is True

        field = fbm.get_field("field_1")
        assert field["area_ha"] > original_area

        # Test updating nonexistent field
        assert fbm.update_field(field_id="nonexistent") is False

        # Test invalid geometry
        with pytest.raises(ValueError):
            fbm.update_field(field_id="field_1", geometry="not a geometry")

    def test_get_field(self):
        """Test retrieving a field."""
        fbm = FieldBoundaryManager()

        # Add some fields
        fbm.add_field(
            geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            field_id="field_1",
            name="Field 1",
            crop_type="corn",
        )
        fbm.add_field(
            geometry=Polygon([(20, 0), (20, 10), (30, 10), (30, 0)]),
            field_id="field_2",
            name="Field 2",
            crop_type="wheat",
        )

        # Test retrieving a field
        field = fbm.get_field("field_1")
        assert field is not None
        assert field["name"] == "Field 1"
        assert field["crop_type"] == "corn"

        # Test retrieving nonexistent field
        assert fbm.get_field("nonexistent") is None

    def test_get_fields_by_crop(self):
        """Test retrieving fields by crop type."""
        fbm = FieldBoundaryManager()

        # Add some fields with different crops
        fbm.add_field(
            geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            field_id="field_1",
            crop_type="corn",
        )
        fbm.add_field(
            geometry=Polygon([(20, 0), (20, 10), (30, 10), (30, 0)]),
            field_id="field_2",
            crop_type="wheat",
        )
        fbm.add_field(
            geometry=Polygon([(40, 0), (40, 10), (50, 10), (50, 0)]),
            field_id="field_3",
            crop_type="corn",
        )

        # Test retrieving fields by crop
        corn_fields = fbm.get_fields_by_crop("corn")
        assert len(corn_fields) == 2
        assert set(corn_fields["field_id"].tolist()) == {"field_1", "field_3"}

        wheat_fields = fbm.get_fields_by_crop("wheat")
        assert len(wheat_fields) == 1
        assert wheat_fields["field_id"].iloc[0] == "field_2"

        # Test retrieving nonexistent crop
        empty_fields = fbm.get_fields_by_crop("soybean")
        assert len(empty_fields) == 0

    def test_get_neighboring_fields(self):
        """Test retrieving neighboring fields."""
        fbm = FieldBoundaryManager()

        # Geometry is in EPSG:4326; buffer_distance is interpreted as meters,
        # so use field separations that are meaningful at metric scale.
        # 0.001 deg longitude ~ 111 m at the equator.
        fbm.add_field(
            geometry=Polygon([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]),
            field_id="field_1",
        )
        fbm.add_field(
            geometry=Polygon(
                [(0.011, 0.0), (0.021, 0.0), (0.021, 0.01), (0.011, 0.01)]
            ),
            field_id="field_2",  # ~111 m from field_1
        )
        fbm.add_field(
            geometry=Polygon([(0.5, 0.0), (0.51, 0.0), (0.51, 0.01), (0.5, 0.01)]),
            field_id="field_3",  # ~54 km from field_1
        )

        # Tiny buffer: no neighbors within 2 m
        neighbors = fbm.get_neighboring_fields(field_id="field_1", buffer_distance=2.0)
        assert len(neighbors) == 0

        # Buffer covering the ~111 m gap reaches only field_2
        neighbors = fbm.get_neighboring_fields(
            field_id="field_1", buffer_distance=200.0
        )
        assert len(neighbors) == 1
        assert neighbors["field_id"].iloc[0] == "field_2"

        # Very large buffer reaches field_3 as well
        neighbors = fbm.get_neighboring_fields(
            field_id="field_1", buffer_distance=100000.0
        )
        assert len(neighbors) == 2
        assert set(neighbors["field_id"].tolist()) == {"field_2", "field_3"}

        # Neighbors are returned in the manager's CRS
        assert str(neighbors.crs) == "EPSG:4326"

        # Test nonexistent field
        with pytest.raises(ValueError):
            fbm.get_neighboring_fields(field_id="nonexistent")

    def test_extract_fields_from_raster(self):
        """Test extracting fields from raster data."""
        fbm = FieldBoundaryManager()

        # Create a temporary raster file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            raster_path = tmp.name

            # Create a simple raster with field-like patterns
            data = np.zeros((100, 100), dtype=np.uint8)
            data[10:30, 10:30] = 1  # Field 1
            data[50:80, 50:80] = 1  # Field 2

            transform = from_origin(0, 100, 1, 1)

            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs="+proj=latlong",
                transform=transform,
            ) as dst:
                dst.write(data, 1)

        try:
            # Test extracting fields
            num_fields = fbm.extract_fields_from_raster(
                raster_path=raster_path, value_field=1, min_area=0.01
            )

            assert num_fields == 2
            assert len(fbm.fields) == 2
            assert "source" in fbm.fields.columns
            assert fbm.fields["source"].iloc[0] == "raster_extraction"

            # Test with simplification
            fbm = FieldBoundaryManager()
            num_fields = fbm.extract_fields_from_raster(
                raster_path=raster_path,
                value_field=1,
                min_area=0.01,
                simplify_tolerance=1.0,
            )

            assert num_fields == 2

            # Test with invalid path
            with pytest.raises(ValueError):
                fbm.extract_fields_from_raster(
                    raster_path="nonexistent.tif", value_field=1
                )

        finally:
            # Clean up the temporary file
            if os.path.exists(raster_path):
                os.unlink(raster_path)

    def test_export_to_file(self):
        """Test exporting fields to file."""
        fbm = FieldBoundaryManager()

        # Add some fields
        fbm.add_field(
            geometry=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]), field_id="field_1"
        )
        fbm.add_field(
            geometry=Polygon([(20, 0), (20, 10), (30, 10), (30, 0)]), field_id="field_2"
        )

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Test exporting to shapefile
            fbm.export_to_file(output_path=output_path)

            # Verify the file was created and can be read
            exported = gpd.read_file(output_path)
            assert len(exported) == 2
            assert set(exported["field_id"].tolist()) == {"field_1", "field_2"}

            # Test export with invalid path
            with pytest.raises(ValueError):
                fbm.export_to_file(output_path="/nonexistent/directory/fields.shp")

        finally:
            # Clean up temporary files
            if os.path.exists(output_path):
                # Need to remove all associated files (.dbf, .shx, etc.)
                base_path = output_path[:-4]  # Remove .shp extension
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    if os.path.exists(base_path + ext):
                        os.unlink(base_path + ext)

    def test_calculate_areas(self):
        """Test area calculation."""
        fbm = FieldBoundaryManager()

        # Add a field without area
        fbm.fields = gpd.GeoDataFrame(
            {
                "field_id": ["field_1"],
                "name": ["Test Field"],
                "geometry": [Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])],
            },
            crs="EPSG:4326",
        )
        fbm.fields["area_ha"] = None

        # Calculate areas
        fbm._calculate_areas()

        # Check that area was calculated
        assert fbm.fields["area_ha"].iloc[0] is not None
        assert fbm.fields["area_ha"].iloc[0] > 0

    def test_area_ha_uses_equal_area_projection(self):
        """Identical 1-degree fields must scale with cos(latitude) of true area.

        Web Mercator (EPSG:3857) stretches x and y by 1/cos(lat), so a
        1-degree cell gets the SAME projected area at any latitude. True
        ground area of a 1x1 degree cell at 60N is ~cos(60) = 0.5 of the
        equatorial cell.
        """
        fbm = FieldBoundaryManager()

        fbm.add_field(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            field_id="equator",
        )
        fbm.add_field(
            geometry=Polygon([(0, 60), (1, 60), (1, 61), (0, 61)]),
            field_id="high_latitude",
        )

        area_equator = fbm.get_field("equator")["area_ha"]
        area_high = fbm.get_field("high_latitude")["area_ha"]

        # A 1x1 degree cell at 60N has ~cos(60) = 0.5 of the equatorial
        # cell area; Web Mercator would yield a ratio of exactly 1.0.
        assert area_high < area_equator
        ratio = area_high / area_equator
        assert 0.45 < ratio < 0.55, f"unexpected area ratio: {ratio}"
