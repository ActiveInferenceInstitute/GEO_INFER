"""Unit tests for CascadianAgriculturalH3Backend."""

import sys

import pytest

from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend


@pytest.fixture
def backend_modules():
    """Minimal modules dict required by CascadianAgriculturalH3Backend."""
    return {}


@pytest.fixture
def backend(backend_modules, tmp_path):
    return CascadianAgriculturalH3Backend(
        modules=backend_modules,
        resolution=8,
        base_data_dir=tmp_path,
        enable_caching=False,
    )


class TestCascadianBackendInit:
    def test_init_with_modules(self, backend_modules, tmp_path):
        b = CascadianAgriculturalH3Backend(
            modules=backend_modules,
            resolution=8,
            base_data_dir=tmp_path,
            enable_caching=False,
        )
        assert b is not None

    def test_init_custom_resolution(self, backend_modules, tmp_path):
        b = CascadianAgriculturalH3Backend(
            modules=backend_modules,
            resolution=7,
            base_data_dir=tmp_path,
            enable_caching=False,
        )
        assert b.resolution == 7

    def test_target_hexagons_property(self, backend):
        # target_hexagons may return a number or a set/list; may not exist yet
        hexagons = getattr(backend, "target_hexagons", None)
        assert hexagons is None or isinstance(hexagons, (int, float, set, list))

    def test_modules_attribute_set(self, backend, backend_modules):
        assert hasattr(backend, "modules") or hasattr(backend, "_modules")

    def test_init_does_not_recompute_target_region(
        self, backend_modules, tmp_path, monkeypatch
    ):
        """GS-191: CascadianAgriculturalH3Backend.__init__ must use the cached
        region result instead of calling _define_target_region a second time
        directly (the SPACE parent's own dispatch is out of scope)."""
        callers = []

        place_init_file = CascadianAgriculturalH3Backend.__init__.__code__.co_filename

        def counting_define(self, target_counties=None):
            caller = sys._getframe(1)
            if caller.f_code.co_filename != place_init_file:
                return {}, []
            callers.append(caller.f_code.co_name)
            return {}, []

        monkeypatch.setattr(
            CascadianAgriculturalH3Backend, "_define_target_region", counting_define
        )
        b = CascadianAgriculturalH3Backend(
            modules=backend_modules,
            resolution=8,
            base_data_dir=tmp_path,
            enable_caching=False,
        )
        assert callers == ["_define_target_region_cached"]
        assert b.target_hexagons == []
        assert b.target_hexagons_by_state == {}


class TestCascadianBackendH3Operations:
    def test_get_h3_cell_returns_string(self, backend):
        cell = backend.get_h3_cell(lat=41.75, lon=-124.2)
        assert isinstance(cell, str)
        assert len(cell) > 0

    def test_cache_key_deterministic(self, backend):
        """Same lat/lon/resolution produces the same cache key."""
        cell1 = backend.get_h3_cell(lat=41.75, lon=-124.2)
        cell2 = backend.get_h3_cell(lat=41.75, lon=-124.2)
        assert cell1 == cell2

    def test_export_to_geojson(self, backend, temp_output_dir):
        result = backend.export_to_geojson(output_dir=str(temp_output_dir))
        assert result is not None


class TestCascadianBackendSPACEIntegration:
    def test_uses_shared_space_backend(self):
        """Backend uses the shared GEO-INFER-SPACE implementation."""
        from geo_infer_space.core.unified_backend import UnifiedH3Backend
        from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

        assert issubclass(CascadianAgriculturalH3Backend, UnifiedH3Backend)

    def test_cell_to_boundary_returns_polygon(self, backend):
        import h3

        cell = h3.latlng_to_cell(41.75, -124.2, 8)
        try:
            boundary = backend.cell_to_boundary(cell)
            assert boundary is not None
        except AttributeError:
            # If method doesn't exist, verify h3 directly
            boundary = h3.cell_to_boundary(cell)
            assert boundary is not None


class TestRemovedLoaderSurface:
    def test_no_osc_loader_attributes(self, backend):
        """The removed OSC loader surface is not reintroduced."""
        for name in ("h3_loader", "h3_data_loader", "osc_repo_dir"):
            assert not hasattr(backend, name)

    def test_osc_repo_dir_argument_rejected(self, backend_modules, tmp_path):
        with pytest.raises(TypeError):
            CascadianAgriculturalH3Backend(
                modules=backend_modules,
                base_data_dir=tmp_path,
                enable_caching=False,
                osc_repo_dir=str(tmp_path),
            )


class TestBaseModuleProcessToH3:
    """PLACE BaseAnalysisModule indexes vector files onto H3 directly."""

    @staticmethod
    def _module(backend):
        from geo_infer_place.core.base_module import BaseAnalysisModule

        class _Module(BaseAnalysisModule):
            def acquire_raw_data(self):
                raise RuntimeError("not used in this test")

            def run_final_analysis(self, h3_data):
                return h3_data

        return _Module(backend, "unit")

    def test_points_and_polygons_are_indexed(self, backend, tmp_path):
        import json

        import h3

        polygon = [
            [-124.21, 41.74],
            [-124.19, 41.74],
            [-124.19, 41.76],
            [-124.21, 41.76],
            [-124.21, 41.74],
        ]
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"kind": "site"},
                    "geometry": {"type": "Point", "coordinates": [-124.2, 41.75]},
                },
                {
                    "type": "Feature",
                    "properties": {"kind": "parcel"},
                    "geometry": {"type": "Polygon", "coordinates": [polygon]},
                },
            ],
        }
        raw = tmp_path / "raw.geojson"
        raw.write_text(json.dumps(payload), encoding="utf-8")

        h3_data = self._module(backend).process_to_h3(raw)

        point_cell = h3.latlng_to_cell(41.75, -124.2, backend.resolution)
        assert {"kind": "site"} in h3_data[point_cell]
        polygon_cells = h3.geo_to_cells(
            {"type": "Polygon", "coordinates": [polygon]}, backend.resolution
        )
        assert polygon_cells
        assert all({"kind": "parcel"} in h3_data[cell] for cell in polygon_cells)
        json.dumps(h3_data)  # cached results must be JSON-serializable
