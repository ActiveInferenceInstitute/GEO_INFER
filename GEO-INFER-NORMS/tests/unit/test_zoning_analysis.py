"""
Tests for the zoning analysis functionality in the GEO-INFER-NORMS module.

This module tests the ZoningAnalyzer and LandUseClassifier classes and their methods
for analyzing zoning districts and land use patterns.
"""

import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import os
import tempfile

from geo_infer_norms.core.zoning_analysis import ZoningAnalyzer, LandUseClassifier
from geo_infer_norms.models.zoning import ZoningCode, ZoningDistrict, LandUseType


class TestZoningAnalyzer:
    """Test cases for the ZoningAnalyzer class."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create test zoning codes
        self.residential_code = ZoningCode(
            code="R-1",
            name="Single Family Residential",
            description="Low-density residential zoning for single-family homes",
            category="residential",
            jurisdiction_id="jur1",
            allowed_uses=["single_family_dwelling", "parks", "schools"],
        )

        self.commercial_code = ZoningCode(
            code="C-1",
            name="Neighborhood Commercial",
            description="Small-scale commercial uses serving neighborhood needs",
            category="commercial",
            jurisdiction_id="jur1",
            allowed_uses=["retail", "restaurants", "offices", "personal_services"],
        )

        self.industrial_code = ZoningCode(
            code="I-1",
            name="Light Industrial",
            description="Light manufacturing and industrial uses",
            category="industrial",
            jurisdiction_id="jur1",
            allowed_uses=["manufacturing", "warehousing", "research_facilities"],
        )

        self.mixed_use_code = ZoningCode(
            code="MU-1",
            name="Mixed Use",
            description="Mix of residential and commercial uses",
            category="mixed_use",
            jurisdiction_id="jur1",
            allowed_uses=["multi_family_dwelling", "retail", "offices", "restaurants"],
        )

        # Create test zoning districts with simple geometries
        self.residential_district = ZoningDistrict(
            id="dist1",
            name="North Residential District",
            zoning_code="R-1",
            jurisdiction_id="jur1",
            geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        )

        self.commercial_district = ZoningDistrict(
            id="dist2",
            name="Central Commercial District",
            zoning_code="C-1",
            jurisdiction_id="jur1",
            geometry=Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)]),
        )

        self.industrial_district = ZoningDistrict(
            id="dist3",
            name="East Industrial District",
            zoning_code="I-1",
            jurisdiction_id="jur1",
            geometry=Polygon([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
        )

        self.mixed_use_district = ZoningDistrict(
            id="dist4",
            name="South Mixed Use District",
            zoning_code="MU-1",
            jurisdiction_id="jur1",
            geometry=Polygon([(1, -1), (1, 0), (2, 0), (2, -1), (1, -1)]),
        )

        # Initialize the ZoningAnalyzer with the test data
        self.analyzer = ZoningAnalyzer(
            zoning_districts=[
                self.residential_district,
                self.commercial_district,
                self.industrial_district,
                self.mixed_use_district,
            ],
            zoning_codes=[
                self.residential_code,
                self.commercial_code,
                self.industrial_code,
                self.mixed_use_code,
            ],
        )

    def test_get_district_by_id(self):
        """Test retrieving a zoning district by ID."""
        district = self.analyzer.get_district_by_id("dist1")
        assert district == self.residential_district

        district = self.analyzer.get_district_by_id("nonexistent")
        assert district is None

    def test_get_code_by_id(self):
        """Test retrieving a zoning code by ID."""
        code = self.analyzer.get_code_by_id("R-1")
        assert code == self.residential_code

        code = self.analyzer.get_code_by_id("nonexistent")
        assert code is None

    def test_get_zoning_at_point(self):
        """Test retrieving zoning districts at a specific point."""
        # Point in residential district
        point = Point(0.5, 0.5)
        districts = self.analyzer.get_zoning_at_point(point)
        assert len(districts) == 1
        assert districts[0].id == "dist1"

        # Point at the boundary between residential and commercial districts
        point = Point(1, 0.5)
        districts = self.analyzer.get_zoning_at_point(point)
        assert len(districts) == 0  # Points on boundaries aren't contained

        # Point outside any district
        point = Point(5, 5)
        districts = self.analyzer.get_zoning_at_point(point)
        assert len(districts) == 0

    def test_calculate_compatibility(self):
        """Test calculating compatibility between zoning codes."""
        # Same code should be fully compatible
        assert self.analyzer.calculate_compatibility("R-1", "R-1") == 1.0

        # Residential and industrial should have low compatibility
        assert self.analyzer.calculate_compatibility("R-1", "I-1") == 0.1

        # Commercial and mixed use should have medium-high compatibility (same category)
        compatibility = self.analyzer.calculate_compatibility("C-1", "MU-1")
        assert compatibility >= 0.5

        # Unknown codes should return default compatibility
        assert self.analyzer.calculate_compatibility("R-1", "nonexistent") == 0.5

    def test_analyze_zoning_boundaries(self):
        """Test analyzing zoning district boundaries for potential conflicts."""
        results = self.analyzer.analyze_zoning_boundaries()

        assert results["status"] == "success"
        assert "adjacency_count" in results
        assert "average_compatibility" in results
        assert "potential_conflicts" in results

        # Check that residential-industrial adjacency is flagged as a potential conflict
        for conflict in results["potential_conflicts"]:
            district1_id, district2_id, score = conflict
            if (district1_id == "dist1" and district2_id == "dist3") or (
                district1_id == "dist3" and district2_id == "dist1"
            ):
                assert score < 0.3

    def test_evaluate_zoning_change(self):
        """Test evaluating the impact of a zoning change."""
        # Test changing residential to mixed use
        results = self.analyzer.evaluate_zoning_change("dist1", "MU-1")

        assert results["status"] == "success"
        assert results["district_id"] == "dist1"
        assert results["current_code"] == "R-1"
        assert results["proposed_code"] == "MU-1"

        # Compatibility should improve when changing from residential to mixed use
        # when adjacent to commercial
        assert results["compatibility_change"] > 0

        # Test with nonexistent district
        results = self.analyzer.evaluate_zoning_change("nonexistent", "C-1")
        assert results["status"] == "error"

        # Test with nonexistent code
        results = self.analyzer.evaluate_zoning_change("dist1", "nonexistent")
        assert results["status"] == "error"

    def test_export_districts_to_geodataframe(self):
        """Test exporting zoning districts to a GeoDataFrame."""
        gdf = self.analyzer.export_districts_to_geodataframe()

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4
        assert set(gdf.columns) >= {"id", "name", "zoning_code", "category", "geometry"}
        assert gdf.crs == "EPSG:4326"

    def test_generate_zoning_report_renders_formatted_values(self):
        """Report lines carry computed values, never bare format specs."""
        report = self.analyzer.generate_zoning_report()

        assert "Total Area: " in report
        assert " ha" in report
        assert "Population Density: " in report
        lines = [line.strip() for line in report.splitlines()]
        assert not any(line in {".1f", ".2f", ".3f"} for line in lines)

    def test_visualize_zoning(self):
        """Test visualization of zoning districts."""
        # Basic visualization test
        fig = self.analyzer.visualize_zoning()
        assert isinstance(fig, plt.Figure)

        # Visualization with highlighted district
        fig = self.analyzer.visualize_zoning(highlight_district="dist1")
        assert isinstance(fig, plt.Figure)

        # Test with temp file saving
        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            fig = self.analyzer.visualize_zoning(save_path=temp.name)
            assert os.path.exists(temp.name)
            assert os.path.getsize(temp.name) > 0


class TestLandUseClassifier:
    """Test cases for the LandUseClassifier class."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create test land use types
        self.residential_type = LandUseType(
            id="lu1",
            name="Residential",
            category="residential",
            subcategory="single_family",
            description="Single-family residential land use",
            typical_zoning_codes=["R-1", "R-2"],
        )

        self.commercial_type = LandUseType(
            id="lu2",
            name="Commercial",
            category="commercial",
            subcategory="retail",
            description="Retail commercial land use",
            typical_zoning_codes=["C-1", "C-2", "MU-1"],
        )

        self.industrial_type = LandUseType(
            id="lu3",
            name="Industrial",
            category="industrial",
            subcategory="light_manufacturing",
            description="Light manufacturing industrial land use",
            typical_zoning_codes=["I-1"],
        )

        # Initialize LandUseClassifier
        self.classifier = LandUseClassifier(
            land_use_types=[
                self.residential_type,
                self.commercial_type,
                self.industrial_type,
            ]
        )

    def test_get_land_use_type_by_id(self):
        """Test retrieving a land use type by ID."""
        land_use = self.classifier.get_land_use_type_by_id("lu1")
        assert land_use == self.residential_type

        land_use = self.classifier.get_land_use_type_by_id("nonexistent")
        assert land_use is None

    def test_calculate_compatibility(self):
        """Test calculating compatibility between land use categories."""
        # Check default compatibility matrix values
        res_res = self.classifier.calculate_compatibility("residential", "residential")
        res_com = self.classifier.calculate_compatibility("residential", "commercial")
        res_ind = self.classifier.calculate_compatibility("residential", "industrial")

        assert res_res > 0.8  # Same category should be highly compatible
        assert (
            res_com > 0.3
        )  # Residential and commercial should have medium compatibility
        assert res_ind < 0.3  # Residential and industrial should have low compatibility

        # Test with unknown category
        unknown = self.classifier.calculate_compatibility("residential", "nonexistent")
        assert unknown == 0.5  # Default medium compatibility

    def test_analyze_land_use_pattern(self):
        """Test analyzing land use patterns in a GeoDataFrame."""
        # Create a simple GeoDataFrame with land use data
        geometries = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
            Polygon([(0, 1), (0, 2), (1, 2), (1, 1), (0, 1)]),
            Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]),
        ]

        categories = [
            "residential",
            "commercial",
            "industrial",
            "residential",
            "commercial",
        ]

        data = {"category": categories, "geometry": geometries}

        land_use_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Test analysis
        results = self.classifier.analyze_land_use_pattern(land_use_gdf)

        assert results["status"] == "success"
        assert "total_area" in results
        assert "area_by_category" in results
        assert "percentage_by_category" in results
        assert "adjacency_matrix" in results
        assert "average_compatibility" in results

        # Check the category counts
        assert results["category_count"] == 3
        assert set(results["area_by_category"].keys()) == {
            "residential",
            "commercial",
            "industrial",
        }

        # Test with missing category column
        bad_gdf = gpd.GeoDataFrame(
            {"wrong_column": [1, 2], "geometry": geometries[:2]}, crs="EPSG:4326"
        )
        results = self.classifier.analyze_land_use_pattern(
            bad_gdf, category_column="category"
        )
        assert results["status"] == "error"

    def test_classify_land_use(self):
        """Test classifying land use based on features."""
        # Create a simple GeoDataFrame with parcel features
        geometries = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
        ]

        data = {
            "building_count": [15, 3, 2],
            "population_density": [6000, 1000, 500],
            "business_count": [2, 8, 1],
            "farmland_percentage": [0, 0, 85],
            "park_area": [5000, 2000, 1000],
            "geometry": geometries,
        }

        features_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Test classification
        result_gdf = self.classifier.classify_land_use(
            features_gdf,
            feature_columns=[
                "building_count",
                "population_density",
                "business_count",
                "farmland_percentage",
                "park_area",
            ],
        )

        assert "land_use_category" in result_gdf.columns
        assert "land_use_confidence" in result_gdf.columns

        # Check that classification worked as expected based on our rules
        assert (
            result_gdf.loc[0, "land_use_category"] == "residential"
        )  # High building count and population density
        assert (
            result_gdf.loc[2, "land_use_category"] == "agricultural"
        )  # High farmland percentage

    def test_visualize_land_use(self):
        """Test visualization of land use data."""
        # Create a simple GeoDataFrame with land use categories
        geometries = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
        ]

        data = {
            "land_use_category": ["residential", "commercial", "industrial"],
            "geometry": geometries,
        }

        land_use_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Basic visualization test
        fig = self.classifier.visualize_land_use(land_use_gdf)
        assert isinstance(fig, plt.Figure)

        # Test with custom colormap
        fig = self.classifier.visualize_land_use(land_use_gdf, cmap="viridis")
        assert isinstance(fig, plt.Figure)

        # Test with temp file saving
        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            fig = self.classifier.visualize_land_use(land_use_gdf, save_path=temp.name)
            assert os.path.exists(temp.name)
            assert os.path.getsize(temp.name) > 0

        # Test with nonexistent column
        fig = self.classifier.visualize_land_use(
            land_use_gdf, category_column="nonexistent"
        )
        assert isinstance(fig, plt.Figure)

        # Test with empty GeoDataFrame
        empty_gdf = gpd.GeoDataFrame()
        fig = self.classifier.visualize_land_use(empty_gdf)
        assert isinstance(fig, plt.Figure)
