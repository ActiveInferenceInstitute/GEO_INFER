import unittest
from pathlib import Path
from geo_infer_space.core.unified_backend import UnifiedH3Backend
from geo_infer_space.core.base_module import BaseAnalysisModule
import tempfile
import shutil
import json
import pytest


class MockModule(BaseAnalysisModule):
    def acquire_raw_data(self) -> Path:
        temp_file = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
        with open(temp_file, "w") as f:
            json.dump({"test": "data"}, f)
        return temp_file

    def run_final_analysis(self, h3_data: dict) -> dict:
        return {"mock_hex": {"value": 42}}


@pytest.mark.core
class TestUnifiedH3Backend(unittest.TestCase):
    def setUp(self):
        # Target areas come from an explicit geojson_path (no CWD default).
        self._config_dir = Path(tempfile.mkdtemp())
        config_path = self._config_dir / "target_areas.geojson"
        sample_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"area": "TestRegion", "subarea": "all"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                }
            ],
        }
        with open(config_path, "w") as f:
            json.dump(sample_geojson, f)
        self.geojson_path = config_path
        self.backend_instance = UnifiedH3Backend(modules={}, resolution=8)
        self.modules = {"mock": MockModule("mock")}
        self.backend = UnifiedH3Backend(
            modules=self.modules,
            resolution=8,
            target_region="TestRegion",
            target_areas={"TestRegion": ["all"]},
            base_data_dir=Path("test_data"),
            geojson_path=config_path,
        )

    def tearDown(self):
        if self._config_dir.exists():
            shutil.rmtree(self._config_dir)

    def test_define_target_region(self):
        """Test target region definition with small real geometry."""
        # Pass a valid GeoJSON dict with [lat, lon] coordinates
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
            ],
        }
        test_geom = {"TestArea": {"all": geojson_polygon}}
        test_backend = UnifiedH3Backend.__new__(UnifiedH3Backend)
        test_backend.resolution = 8
        test_backend.target_region = "TestArea"
        test_backend.modules = {}
        test_backend.base_data_dir = Path("test_data")
        test_backend.unified_data = {}
        test_backend.analysis_scores = {}

        def mock_get_geometries(target_areas):
            return test_geom

        test_backend._get_geometries = mock_get_geometries
        hex_by_area, all_hex = test_backend._define_target_region({"TestArea": ["all"]})
        self.assertGreater(len(all_hex), 0)

    def test_run_comprehensive_analysis(self):
        """Test full analysis with small real data."""
        self.backend.target_hexagons = ["mock_hex"]
        self.backend.modules["mock"].run_analysis = lambda: {"mock_hex": {"value": 42}}
        self.backend.run_comprehensive_analysis()
        self.assertIn("mock", self.backend.unified_data.get("mock_hex", {}))

    def test_get_comprehensive_summary(self):
        """Test summary generation."""
        self.backend.run_comprehensive_analysis()
        summary = self.backend.get_comprehensive_summary()
        self.assertNotIn("error", summary)
        self.assertIn("target_region", summary)
        self.assertEqual(summary["target_region"], "TestRegion")

    def test_export_unified_data(self):
        """Test data export to JSON."""
        self.backend.run_comprehensive_analysis()
        temp_file = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
        self.backend.export_unified_data(str(temp_file), "json")
        self.assertTrue(temp_file.exists())
        with open(temp_file) as f:
            data = json.load(f)
        self.assertGreater(len(data), 0)
        temp_file.unlink()

    def test_generate_interactive_dashboard(self):
        """Test dashboard generation."""
        self.backend.run_comprehensive_analysis()
        temp_html = Path(tempfile.NamedTemporaryFile(suffix=".html", delete=False).name)
        self.backend.generate_interactive_dashboard(str(temp_html))
        self.assertTrue(temp_html.exists())
        temp_html.unlink()
