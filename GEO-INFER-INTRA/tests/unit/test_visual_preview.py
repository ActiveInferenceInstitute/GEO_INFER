"""Unit and contract tests for spatial visual preview helpers (DOCS-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_infer_intra.core.documentation.visual_preview import (
    MODULE_PROFILES,
    SpatialPreviewArtifacts,
    generate_all_module_previews,
    generate_module_preview_suite,
    render_leaflet_html,
    render_png_card,
    render_svg_card,
)


@pytest.mark.unit
@pytest.mark.geospatial
class TestVisualPreviewContract:
    """Test suite verifying reproducible spatial widget preview cards."""

    def test_all_45_modules_registered(self) -> None:
        """Verify all 45 GEO-INFER modules are configured in the profile registry."""
        expected_modules = {
            "ACT",
            "AG",
            "AGENT",
            "AI",
            "ANT",
            "API",
            "APP",
            "ART",
            "BAYES",
            "BIO",
            "CIV",
            "CLIMATE",
            "COG",
            "COMMS",
            "DATA",
            "ECON",
            "EDU",
            "EMERGENCY",
            "ENERGY",
            "EXAMPLES",
            "FOREST",
            "GIT",
            "HEALTH",
            "INSURANCE",
            "INTRA",
            "IOT",
            "LOG",
            "MARINE",
            "MATH",
            "METAGOV",
            "NORMS",
            "OPS",
            "ORG",
            "PEP",
            "PLACE",
            "REQ",
            "RISK",
            "SEC",
            "SIM",
            "SPACE",
            "SPM",
            "TEST",
            "TIME",
            "TRANSPORT",
            "WATER",
        }
        assert set(MODULE_PROFILES.keys()) == expected_modules
        assert len(MODULE_PROFILES) == 45

    def test_module_profiles_have_valid_spatial_attributes(self) -> None:
        """Ensure each module profile defines valid coordinates, colors, and features."""
        for profile in MODULE_PROFILES.values():
            assert "name" in profile
            assert "description" in profile
            assert "category" in profile
            assert "center" in profile
            lat, lng = profile["center"]
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lng <= 180.0
            assert "zoom" in profile
            assert 1 <= profile["zoom"] <= 20
            assert profile["primary_color"].startswith("#")
            assert len(profile["primary_color"]) == 7
            assert profile["secondary_color"].startswith("#")
            assert len(profile["secondary_color"]) == 7
            assert isinstance(profile["features"], list)
            assert len(profile["features"]) >= 3

    def test_render_leaflet_html_contract(self, tmp_path: Path) -> None:
        """Verify Leaflet HTML includes its local static fallback."""
        out_file = tmp_path / "test_act_map.html"
        html_str = render_leaflet_html("ACT", out_file)

        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == html_str
        assert "<!doctype html>" in html_str.lower() or "<html" in html_str.lower()
        assert "GEO-INFER-ACT" in html_str
        assert "Active Inference Engine" in html_str
        assert len(html_str) > 500

    def test_render_svg_card_contract(self, tmp_path: Path) -> None:
        """Verify SVG card rendering produces valid vector XML with hex polygons."""
        out_file = tmp_path / "test_space_card.svg"
        svg_str = render_svg_card("SPACE", out_file)

        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == svg_str
        assert svg_str.startswith("<svg")
        assert svg_str.endswith("</svg>")
        assert 'viewBox="0 0 700 380"' in svg_str
        assert "GEO-INFER-SPACE" in svg_str
        assert "<polygon" in svg_str
        assert "<circle" in svg_str

    def test_render_png_card_contract(self, tmp_path: Path) -> None:
        """Verify PNG rendering produces a valid PNG binary with PNG header and IEND."""
        out_file = tmp_path / "test_water_card.png"
        png_bytes = render_png_card("WATER", out_file)

        assert out_file.exists()
        assert out_file.read_bytes() == png_bytes
        # Standard PNG Magic Header: 89 50 4E 47 0D 0A 1A 0A
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"IHDR" in png_bytes
        assert b"IDAT" in png_bytes
        assert b"IEND" in png_bytes
        assert len(png_bytes) > 200

    def test_generate_module_preview_suite(self, tmp_path: Path) -> None:
        """Verify the full artifact bundle generation and receipt verification."""
        artifacts = generate_module_preview_suite("BIO", tmp_path)

        assert isinstance(artifacts, SpatialPreviewArtifacts)
        assert artifacts.module_id == "BIO"
        assert artifacts.html_path.exists()
        assert artifacts.svg_path.exists()
        assert artifacts.png_path.exists()
        assert artifacts.manifest_path.exists()

        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "geo-infer-intra-visual-preview/v2"
        assert manifest["module_id"] == "GEO-INFER-BIO"
        assert manifest["name"] == "Biological Systems"
        assert len(manifest["artifacts"]) == 3
        assert manifest["accessibility"]["has_title"] is True
        assert manifest["accessibility"]["has_svg_viewbox"] is True
        assert manifest["accessibility"]["has_png_metadata"] is True

    def test_unknown_module_raises(self, tmp_path: Path) -> None:
        """Verify unknown module identifiers raise ValueError."""
        with pytest.raises(ValueError, match="Unknown module ID"):
            render_leaflet_html("INVALID_MODULE", tmp_path / "fail.html")

        with pytest.raises(ValueError, match="Unknown module ID"):
            render_svg_card("INVALID_MODULE", tmp_path / "fail.svg")

        with pytest.raises(ValueError, match="Unknown module ID"):
            render_png_card("INVALID_MODULE", tmp_path / "fail.png")

    def test_generate_all_module_previews(self, tmp_path: Path) -> None:
        """Verify batch emission produces bundles for all 45 modules deterministically."""
        all_bundles = generate_all_module_previews(tmp_path / "first")
        repeated = generate_all_module_previews(tmp_path / "second")
        assert len(all_bundles) == 45
        for mod_id, bundle in all_bundles.items():
            assert bundle.module_id == mod_id
            assert bundle.html_path.exists()
            assert bundle.svg_path.exists()
            assert bundle.png_path.exists()
            assert bundle.manifest_path.exists()
            assert bundle.html_bytes > 0
            assert bundle.svg_bytes > 0
            assert bundle.png_bytes > 0
            for attribute in ("html_path", "svg_path", "png_path", "manifest_path"):
                assert (
                    getattr(bundle, attribute).read_bytes()
                    == getattr(repeated[mod_id], attribute).read_bytes()
                )


@pytest.mark.unit
def test_preview_bytes_and_receipts_are_reproducible(tmp_path):
    """Equivalent inputs produce byte-identical artifacts and content hashes."""
    import hashlib

    first = generate_module_preview_suite("SPACE", tmp_path / "first")
    second = generate_module_preview_suite("SPACE", tmp_path / "second")
    for attribute in ("html_path", "svg_path", "png_path", "manifest_path"):
        assert (
            getattr(first, attribute).read_bytes()
            == getattr(second, attribute).read_bytes()
        )
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["provenance"]["data_kind"] == "illustrative"
    for artifact in manifest["artifacts"]:
        assert (
            artifact["sha256"]
            == hashlib.sha256(
                (first.html_path.parent / artifact["name"]).read_bytes()
            ).hexdigest()
        )


@pytest.mark.unit
def test_preview_uses_h3_cells_and_has_offline_content():
    """HTML retains an accessible static map without Leaflet or external tiles."""
    import h3

    markup = render_leaflet_html("SPACE")
    cell = h3.latlng_to_cell(*MODULE_PROFILES["SPACE"]["center"], 7)
    assert cell in markup
    assert "Illustrative" in markup
    assert 'id="static-preview"' in markup
    assert "<title>" in render_svg_card("SPACE")


@pytest.mark.unit
def test_preview_rejects_invalid_canvas_dimensions():
    """Bad dimensions fail before allocating raster memory."""
    for render in (render_leaflet_html, render_svg_card, render_png_card):
        with pytest.raises(ValueError, match="dimensions"):
            render("SPACE", width=0)
        with pytest.raises(ValueError, match="dimensions"):
            render("SPACE", height=100000)


@pytest.mark.unit
def test_preview_rejects_attribute_injection(monkeypatch):
    """Profile extensions cannot inject active SVG attributes through colors."""
    profile = dict(MODULE_PROFILES["SPACE"], secondary_color='red" onload="alert(1)')
    monkeypatch.setitem(MODULE_PROFILES, "SPACE", profile)
    with pytest.raises(ValueError, match="colors"):
        render_svg_card("SPACE")


@pytest.mark.unit
@pytest.mark.parametrize("center", [(60.0, 20.0), (45.0, 179.9999), (90.0, 0.0)])
def test_static_projection_preserves_local_geodesic_proportions(monkeypatch, center):
    """High latitude and date-line previews retain physical H3 edge proportions."""
    import math
    import h3
    from geo_infer_intra.core.documentation.visual_preview import (
        _cells,
        _projected_rings,
    )

    monkeypatch.setitem(
        MODULE_PROFILES, "SPACE", dict(MODULE_PROFILES["SPACE"], center=center)
    )
    geographic = _cells("SPACE")
    projected = _projected_rings("SPACE", 700, 380)
    scales = []
    for (_, ring), pixels in zip(geographic, projected):
        for start, end, p0, p1 in zip(ring, ring[1:], pixels, pixels[1:]):
            scales.append(
                math.dist(p0, p1) / h3.great_circle_distance(start, end, unit="km")
            )
    assert max(scales) / min(scales) < 1.00001
    assert all(0 <= x <= 700 and 0 <= y <= 380 for ring in projected for x, y in ring)


@pytest.mark.unit
def test_leaflet_longitude_unwrap_prevents_world_spanning_polygons(monkeypatch):
    """Equivalent longitudes stay adjacent when an H3 cell crosses the date line."""
    import re

    monkeypatch.setitem(
        MODULE_PROFILES, "SPACE", dict(MODULE_PROFILES["SPACE"], center=(45, 179.9999))
    )
    markup = render_leaflet_html("SPACE")
    payload = json.loads(
        re.search(
            r'<script id="preview-data" type="application/json">(.*?)</script>',
            markup,
            re.S,
        ).group(1)
    )
    longitudes = [longitude for _, ring in payload["cells"] for _, longitude in ring]
    assert max(longitudes) - min(longitudes) < 1
    assert min(longitudes) > 179
    assert 'role="region"' in markup


@pytest.mark.unit
@pytest.mark.parametrize(
    "center", [(float("nan"), 0), (0, float("inf")), (91, 0), (0, 181), (True, 0), [0]]
)
def test_invalid_profile_coordinates_fail_before_writing(monkeypatch, tmp_path, center):
    """Extension profiles cannot introduce invalid coordinates into receipts."""
    monkeypatch.setitem(
        MODULE_PROFILES, "SPACE", dict(MODULE_PROFILES["SPACE"], center=center)
    )
    for render, suffix in [
        (render_svg_card, "svg"),
        (render_png_card, "png"),
        (render_leaflet_html, "html"),
    ]:
        path = tmp_path / f"invalid.{suffix}"
        with pytest.raises(ValueError, match="center"):
            render("SPACE", path)
        assert not path.exists()


@pytest.mark.unit
def test_png_contains_named_module_description_and_projection():
    """Raster cards carry text equivalents and honest projection provenance."""
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(render_png_card("SPACE"))) as image:
        assert MODULE_PROFILES["SPACE"]["name"] in image.info["Description"]
        assert image.info["Projection"] == "Local orthographic on a unit sphere"
