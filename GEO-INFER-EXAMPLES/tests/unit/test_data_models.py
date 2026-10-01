"""Tests for domain-specific data models and DataFormatConverter."""

import pytest
from datetime import datetime

from geo_infer_examples.models.integration_models import (
    ModuleType,
    ModuleSpec,
    DataFormat,
    SpatialTemporalData,
    AnalysisResult,
    IntegrationResult,
    HealthSurveillanceData,
    AgriculturalData,
    UrbanPlanningData,
    ClimateData,
    DataFormatConverter,
    GEO_INFER_MODULES,
)


def _make_temporal_range():
    return (datetime(2024, 1, 1), datetime(2024, 6, 30))


def _make_spatial_bounds():
    return (-74.0, 40.0, -73.5, 40.5)


class TestSpatialTemporalData:
    """Tests for the base SpatialTemporalData dataclass."""

    def test_create(self):
        data = SpatialTemporalData(
            features=[{"type": "Feature", "geometry": {"type": "Point"}}],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
        )
        assert data.coordinate_system == "EPSG:4326"
        assert data.temporal_resolution is None
        assert len(data.features) == 1

    def test_to_geojson(self):
        data = SpatialTemporalData(
            features=[{"type": "Feature"}],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            spatial_resolution=100.0,
        )
        gj = data.to_geojson()
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 1
        assert "temporal_range" in gj["metadata"]
        assert gj["metadata"]["spatial_resolution"] == 100.0

    def test_to_geojson_temporal_range_iso(self):
        tr = _make_temporal_range()
        data = SpatialTemporalData(
            features=[],
            temporal_range=tr,
            spatial_bounds=_make_spatial_bounds(),
        )
        gj = data.to_geojson()
        assert gj["metadata"]["temporal_range"][0] == tr[0].isoformat()
        assert gj["metadata"]["temporal_range"][1] == tr[1].isoformat()


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_create_minimal(self):
        result = AnalysisResult(data={"mean": 42.0})
        assert result.confidence is None
        assert result.method is None
        assert isinstance(result.timestamp, datetime)

    def test_to_dict(self):
        result = AnalysisResult(
            data={"prediction": [1, 2, 3]},
            confidence=0.95,
            method="random_forest",
            parameters={"n_trees": 100},
            performance_metrics={"mse": 0.01},
        )
        d = result.to_dict()
        assert d["confidence"] == 0.95
        assert d["method"] == "random_forest"
        assert d["parameters"]["n_trees"] == 100
        assert "timestamp" in d


class TestIntegrationResult:
    """Tests for IntegrationResult dataclass."""

    def test_create(self):
        ir = IntegrationResult(success=True, data={"output": "ok"})
        assert ir.success is True
        assert ir.errors == []
        assert ir.warnings == []
        assert ir.module_results == {}

    def test_add_and_get_module_result(self):
        ir = IntegrationResult(success=True, data={})
        ar = AnalysisResult(data={"v": 1}, confidence=0.9)
        ir.add_module_result("SPACE", ar)
        retrieved = ir.get_module_result("SPACE")
        assert retrieved is ar
        assert retrieved.confidence == 0.9

    def test_get_missing_module_result(self):
        ir = IntegrationResult(success=False, data={})
        assert ir.get_module_result("NONEXISTENT") is None

    def test_to_dict(self):
        ir = IntegrationResult(
            success=True,
            data={"k": "v"},
            errors=["warn1"],
            execution_time=1.5,
        )
        ar = AnalysisResult(data={"x": 1})
        ir.add_module_result("AI", ar)
        d = ir.to_dict()
        assert d["success"] is True
        assert d["execution_time"] == 1.5
        assert "AI" in d["module_results"]
        assert d["module_results"]["AI"]["data"] == {"x": 1}


class TestHealthSurveillanceData:
    """Tests for HealthSurveillanceData (extends SpatialTemporalData)."""

    def test_create(self):
        data = HealthSurveillanceData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            disease_type="influenza",
            case_data=[{"id": 1}],
        )
        assert data.disease_type == "influenza"
        assert len(data.case_data) == 1

    def test_to_health_geojson(self):
        data = HealthSurveillanceData(
            features=[{"f": 1}],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            disease_type="covid",
            case_data=[{"id": 1}, {"id": 2}],
            demographic_data=[{"age": 30}],
            severity_levels=["mild", "severe"],
        )
        gj = data.to_health_geojson()
        assert gj["type"] == "FeatureCollection"
        meta = gj["metadata"]
        assert meta["domain"] == "health_surveillance"
        assert meta["disease_type"] == "covid"
        assert meta["case_count"] == 2
        assert meta["demographic_count"] == 1


class TestAgriculturalData:
    """Tests for AgriculturalData."""

    def test_create(self):
        data = AgriculturalData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            crop_types=["corn", "wheat"],
        )
        assert len(data.crop_types) == 2

    def test_to_agricultural_geojson(self):
        data = AgriculturalData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            crop_types=["rice"],
            field_boundaries=[{"id": "f1"}],
            weather_data=[{"temp": 25}],
            management_practices=[{"irrigation": True}],
        )
        gj = data.to_agricultural_geojson()
        meta = gj["metadata"]
        assert meta["domain"] == "agriculture"
        assert meta["crop_types"] == ["rice"]
        assert meta["field_count"] == 1
        assert meta["weather_records"] == 1
        assert meta["management_practices"] == 1


class TestUrbanPlanningData:
    """Tests for UrbanPlanningData."""

    def test_create(self):
        data = UrbanPlanningData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            infrastructure={"roads": 10, "parks": 5},
        )
        assert data.infrastructure["roads"] == 10

    def test_to_urban_geojson(self):
        data = UrbanPlanningData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            zoning_data=[{"zone": "R1"}],
            infrastructure={"roads": 10, "parks": 5},
            community_input=[{"comment": "needs more parks"}],
        )
        gj = data.to_urban_geojson()
        meta = gj["metadata"]
        assert meta["domain"] == "urban_planning"
        assert meta["zoning_areas"] == 1
        assert meta["community_inputs"] == 1
        assert set(meta["infrastructure_types"]) == {"roads", "parks"}


class TestClimateData:
    """Tests for ClimateData."""

    def test_create(self):
        data = ClimateData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            variables=["temperature", "precipitation"],
            scenarios=["RCP4.5", "RCP8.5"],
            models=["CESM", "GFDL"],
        )
        assert data.ensemble_data is False
        assert data.downscaling_method is None

    def test_to_climate_geojson(self):
        data = ClimateData(
            features=[],
            temporal_range=_make_temporal_range(),
            spatial_bounds=_make_spatial_bounds(),
            variables=["temperature"],
            scenarios=["RCP8.5"],
            models=["CESM"],
            ensemble_data=True,
            downscaling_method="BCSD",
            bias_correction="quantile_mapping",
        )
        gj = data.to_climate_geojson()
        meta = gj["metadata"]
        assert meta["domain"] == "climate"
        assert meta["variables"] == ["temperature"]
        assert meta["ensemble_data"] is True
        assert meta["downscaling_method"] == "BCSD"
        assert meta["bias_correction"] == "quantile_mapping"


class TestDataFormatConverter:
    """Tests for DataFormatConverter utility class."""

    def test_same_format_returns_data(self):
        data = {"features": [{"id": 1}]}
        result = DataFormatConverter.convert_to_standard_format(
            data, DataFormat.GEOJSON, DataFormat.GEOJSON
        )
        assert result is data

    def test_geojson_to_spatial_temporal(self):
        data = {
            "features": [{"type": "Feature"}],
            "metadata": {
                "temporal_range": ["2024-01-01", "2024-06-30"],
                "spatial_bounds": [-74.0, 40.0, -73.5, 40.5],
            },
        }
        result = DataFormatConverter.convert_to_standard_format(
            data, DataFormat.GEOJSON, DataFormat.SPATIAL_TEMPORAL_JSON
        )
        assert result["features"] == data["features"]
        assert result["temporal_info"] == data["metadata"]["temporal_range"]
        assert result["spatial_bounds"] == data["metadata"]["spatial_bounds"]

    def test_spatial_temporal_to_geojson(self):
        data = {
            "features": [{"type": "Feature"}],
            "temporal_info": ["2024-01-01", "2024-06-30"],
            "spatial_bounds": [-74.0, 40.0, -73.5, 40.5],
            "metadata": {"crs": "EPSG:4326"},
        }
        result = DataFormatConverter.convert_to_standard_format(
            data, DataFormat.SPATIAL_TEMPORAL_JSON, DataFormat.GEOJSON
        )
        assert result["type"] == "FeatureCollection"
        assert result["features"] == data["features"]
        assert result["metadata"]["temporal_range"] == data["temporal_info"]

    def test_unsupported_conversion_raises(self):
        with pytest.raises(ValueError, match="No conversion available"):
            DataFormatConverter.convert_to_standard_format(
                {}, DataFormat.RASTER_ARRAY, DataFormat.HEALTH_RECORD
            )


class TestGeoInferModules:
    """Tests for pre-defined module specifications."""

    def test_data_module_exists(self):
        assert "DATA" in GEO_INFER_MODULES
        spec = GEO_INFER_MODULES["DATA"]
        assert spec.name == "GEO-INFER-DATA"
        assert spec.module_type == ModuleType.DATA_PROCESSING

    def test_space_module_exists(self):
        assert "SPACE" in GEO_INFER_MODULES
        spec = GEO_INFER_MODULES["SPACE"]
        assert spec.module_type == ModuleType.SPATIAL_TEMPORAL
        assert "MATH" in spec.dependencies

    def test_modules_have_valid_types(self):
        for spec in GEO_INFER_MODULES.values():
            assert isinstance(spec, ModuleSpec)
            assert isinstance(spec.module_type, ModuleType)
            assert spec.version is not None
