"""Offline complete-source/license custody and actual bioregion rendering."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from shapely.geometry import Point, shape

from geo_infer_place.core.bioregion_visualization import create_bioregion_map
from geo_infer_place.core.cascadia_boundary import (
    LICENSE_URL,
    NOTICE,
    OUTPUT,
    RAW_FILES,
    RECEIPT,
    capture_cascadia_boundary,
    normalize_cascadia_boundary,
    verify_cascadia_boundary,
)
from geo_infer_place.core.regional_layers import _json_bytes

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "GEO-INFER-PLACE/locations/cascadia/config"
PACKAGED = ROOT / "GEO-INFER-PLACE/src/geo_infer_place/locations/cascadia/config"
NAMES = [*RAW_FILES.values(), OUTPUT, NOTICE, RECEIPT]


def captures():
    """Read exact retained publisher responses, never contacting the service."""
    return {key: (CONFIG / name).read_bytes() for key, name in RAW_FILES.items()}


def test_complete_source_replay_and_packaged_bytes():
    """Source IDs/extent and every native vertex survive deterministic replay."""
    receipt = verify_cascadia_boundary(CONFIG)
    assert receipt == verify_cascadia_boundary(PACKAGED)
    for name in NAMES:
        assert (CONFIG / name).read_bytes() == (PACKAGED / name).read_bytes()
    raw = captures()
    rebuilt = normalize_cascadia_boundary(raw)
    assert _json_bytes(rebuilt) == (CONFIG / OUTPUT).read_bytes()
    assert rebuilt["features"] == json.loads(raw["geometry"])["features"]
    assert receipt["native_ids"] == [1] and receipt["feature_count"] == 1
    geometry = shape(rebuilt["features"][0]["geometry"])
    assert geometry.geom_type == "MultiPolygon" and len(geometry.geoms) == 178
    assert geometry.covers(Point(-122.33, 47.61))  # Seattle inland reference.
    assert geometry.covers(Point(-116.20, 43.61))  # Boise / Snake watershed.
    assert not geometry.covers(Point(-121.49, 38.58))  # Sacramento is south.
    assert geometry.bounds[3] > 61 and geometry.bounds[0] < -144
    assert rebuilt["provenance"]["license_url"] == LICENSE_URL


def test_complete_source_map_fits_full_extent_and_displays_attribution(tmp_path):
    """All configured layers render, with the licensed full boundary identified."""
    output = tmp_path / "complete.html"
    create_bioregion_map(CONFIG, {}, output)
    html = output.read_text(encoding="utf-8")
    assert 'id="boundary-attribution"' in html
    assert "Aquila Flower" in html and LICENSE_URL in html
    assert "fitBounds" in html and "61.684726" in html
    assert "https://tile.openstreetmap.org/" in html
    assert "basemaps.cartocdn.com" not in html
    assert "Unavailable layers:" not in html
    report = json.loads(output.with_suffix(".layers.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert len(report["layers"]) == 4
    assert report["boundary_source"]["whole_cascadia_bioregion"] is True
    assert report["boundary_source"]["native_ids"] == [1]


@pytest.mark.parametrize(
    "case",
    [
        "license",
        "publisher",
        "service",
        "epoch",
        "count",
        "ids",
        "extent",
        "projected",
        "geometry",
        "truncated",
    ],
)
def test_wrong_or_partial_source_cannot_qualify(case):
    """Authentic captures with a changed obligation fail before publication."""
    raw = captures()
    values = {key: json.loads(value) for key, value in raw.items()}
    if case == "license":
        values["item"]["licenseInfo"] = "All rights reserved"
    elif case == "publisher":
        values["item"]["owner"] = "other-publisher"
    elif case == "service":
        values["layer"]["serviceItemId"] = "different-item"
    elif case == "epoch":
        values["layer_final"]["editingInfo"]["dataLastEditDate"] += 1
    elif case == "count":
        values["count"]["count"] = True
    elif case == "ids":
        values["ids"]["objectIds"] = [999]
    elif case == "extent":
        values["extent"]["extent"]["xmax"] -= 1
    elif case == "projected":
        values["geometry"]["crs"] = {"properties": {"name": "EPSG:3857"}}
    elif case == "geometry":
        values["geometry"]["features"][0]["geometry"] = {
            "type": "Point",
            "coordinates": [-122, 47],
        }
    else:
        values["geometry"]["exceededTransferLimit"] = True
    with pytest.raises(ValueError):
        normalize_cascadia_boundary(
            {key: _json_bytes(value) for key, value in values.items()}
        )


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}', b'{"id":NaN}', b'{"id":1e999}'])
def test_duplicate_and_nonfinite_metadata_cannot_qualify(raw):
    """Metadata decoding is strict independently of geometry validation."""
    data = captures()
    data["item"] = raw
    with pytest.raises(ValueError):
        normalize_cascadia_boundary(data)


def test_modified_boundary_cannot_render_as_captured_source(tmp_path):
    """Retaining a receipt makes even stripped-provenance substitutions fail."""
    target = tmp_path / "changed"
    shutil.copytree(CONFIG, target)
    data = json.loads((target / OUTPUT).read_text(encoding="utf-8"))
    data.pop("provenance")
    (target / OUTPUT).write_bytes(_json_bytes(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        create_bioregion_map(target, {}, tmp_path / "bad.html")
    assert not (tmp_path / "bad.html").exists()


def test_capture_never_replaces_existing_files(tmp_path):
    """A rejected capture preserves original bytes before any network work."""
    target = tmp_path / OUTPUT
    target.write_bytes(b"retain original")
    with pytest.raises(FileExistsError):
        capture_cascadia_boundary(tmp_path)
    assert target.read_bytes() == b"retain original"


def test_captured_source_survives_real_crlf_checkout(tmp_path):
    """Publisher bytes and normalized/notice/receipt hashes remain portable."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git = ["git", "-c", "core.fsmonitor=false", "-C", str(checkout)]
    for args in [("init",), ("config", "core.autocrlf", "false")]:
        subprocess.run([*git, *args], check=True, capture_output=True)
    shutil.copy2(ROOT / ".gitattributes", checkout / ".gitattributes")
    target = checkout / CONFIG.relative_to(ROOT)
    target.mkdir(parents=True)
    for name in NAMES:
        shutil.copy2(CONFIG / name, target / name)
    (checkout / "control.txt").write_bytes(b"control\n")
    subprocess.run([*git, "add", "."], check=True, capture_output=True)
    subprocess.run(
        [*git, "config", "core.autocrlf", "true"], check=True, capture_output=True
    )
    subprocess.run(
        [*git, "config", "core.eol", "crlf"], check=True, capture_output=True
    )
    for name in NAMES:
        (target / name).unlink()
    (checkout / "control.txt").unlink()
    subprocess.run(
        [*git, "checkout-index", "--all", "--force"], check=True, capture_output=True
    )
    assert (checkout / "control.txt").read_bytes() == b"control\r\n"
    for name in NAMES:
        assert (
            hashlib.sha256((target / name).read_bytes()).digest()
            == hashlib.sha256((CONFIG / name).read_bytes()).digest()
        )
    verify_cascadia_boundary(target)


def test_boundary_map_can_omit_external_basemap(tmp_path):
    """A geometry-only map avoids external tile dependencies without losing credit."""
    output = tmp_path / "no-tiles.html"
    create_bioregion_map(CONFIG, {}, output, tiles=None)
    html = output.read_text(encoding="utf-8")
    assert "L.tileLayer(" not in html
    assert 'id="boundary-attribution"' in html and "fitBounds" in html
