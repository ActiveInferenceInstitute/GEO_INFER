from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, UTC
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import zipfile

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box, mapping

from geo_infer_place.locations.california.acquisition import (
    SourceEndpoints,
    acquire_sources,
    check_sources,
)
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    ObservatoryConfig,
    digest,
    strict_bytes,
)
from geo_infer_place.locations.california.geometry import Landscape, prepare_landscape


def fixture_payloads(tmp_path: Path) -> tuple[list[dict], bytes]:
    features = [
        {
            "type": "Feature",
            "geometry": mapping(
                box(-121 + index * 0.002, 38, -121 + (index + 1) * 0.002, 38.002)
            ),
            "properties": {
                "OBJECTID": index + 1,
                "County": f"Fixture county {index + 1}",
                "FIPS": f"{index + 1:03d}",
            },
        }
        for index in range(58)
    ]
    gis = tmp_path / "shape"
    gis.mkdir()
    gpd.GeoDataFrame(
        {"DM": [1]}, geometry=[box(-121, 38, -120.884, 38.002)], crs="EPSG:4326"
    ).to_file(gis / "fixture.shp")
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "x") as stream:
        for member in gis.iterdir():
            stream.write(member, member.name)
    return features, archive.read_bytes()


@contextmanager
def source_server(features: list[dict], drought: bytes, failure: str = ""):
    calls = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path)
            queries = parse_qs(route.query)
            calls[route.path] = calls.get(route.path, 0) + 1
            if route.path.startswith("/drought/"):
                raw = drought
            elif route.path == "/events":
                year = int(queries["starttime"][0][:4])
                earthquake = {
                    "type": "Feature",
                    "id": f"fixture-{year}",
                    "geometry": {"type": "Point", "coordinates": [-120.99, 38.001, 0]},
                    "properties": {
                        "mag": 3.4,
                        "time": int(
                            datetime(year, 1, 1, tzinfo=UTC).timestamp() * 1000
                        ),
                        "type": "earthquake",
                    },
                }
                raw = json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [earthquake]
                        * (20000 if failure == "events" else 1),
                    }
                ).encode()
            elif route.path.endswith("/query"):
                counties = route.path.startswith("/counties")
                selected = (
                    features
                    if counties
                    else [
                        {
                            "type": "Feature",
                            "geometry": mapping(box(-121, 38, -120.99, 38.002)),
                            "properties": {"OBJECTID": 1, "YEAR_": 2016, "STATE": "CA"},
                        }
                    ]
                )
                if "returnIdsOnly" in queries:
                    ids = [feature["properties"]["OBJECTID"] for feature in selected]
                    if failure == "ids":
                        ids.append(ids[-1])
                    payload = {"objectIds": ids}
                else:
                    payload = {
                        "type": "FeatureCollection",
                        "features": selected[:-1] if failure == "page" else selected,
                        "exceededTransferLimit": failure == "truncation",
                    }
                raw = json.dumps(payload).encode()
            else:
                payload = {
                    "name": "fixture",
                    "objectIdField": "OBJECTID",
                    "maxRecordCount": 1000,
                    "editingInfo": {
                        "lastEditDate": calls[route.path] if failure == "drift" else 1
                    },
                }
                raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield SourceEndpoints(
            counties=base + "/counties",
            fires=base + "/fires",
            earthquakes=base + "/events",
            drought=base + "/drought",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_real_http_capture_closed_roster_and_polygon_rasterization(
    tmp_path: Path,
) -> None:
    """Capture real local HTTP files and rasterize an explicitly synthetic county fixture."""
    features, drought = fixture_payloads(tmp_path)
    config = ObservatoryConfig(
        first_year=2016,
        target_year=2018,
        rounds=1,
        seeds=(0,),
        budget=12,
        hazard_budget_floor=1,
    )
    sources = tmp_path / "sources"
    with source_server(features, drought) as endpoints:
        receipt_path = acquire_sources(config, sources, endpoints=endpoints, timeout=30)
    before = digest(receipt_path)
    receipt = check_sources(sources)
    assert len(receipt.county_ids) == 58 and receipt.fire_ids == (1,)
    assert digest(receipt_path) == before
    prepared = prepare_landscape(config, sources, tmp_path / "landscape")
    landscape = Landscape.model_validate_json(strict_bytes(prepared))
    assert landscape.uncovered_fraction <= 1e-7
    assert min(cell.raw_county_coverage_ratio for cell in landscape.cells) >= 1 - 1e-8
    assert len(landscape.counties) == 58
    assert all(
        abs(sum(share.fraction for share in cell.county_shares) - 1) < 1e-12
        for cell in landscape.cells
    )
    assert np.all(np.asarray(landscape.indicators[2].positive) == 1)
    assert np.any(np.asarray(landscape.indicators[0].values[0]) > 0)
    assert np.any(np.asarray(landscape.indicators[1].values))
    sources.joinpath("undeclared.bin").write_bytes(b"extra")
    with pytest.raises(CaliforniaError, match="roster"):
        check_sources(sources)


@pytest.mark.parametrize("failure", ["ids", "page", "truncation", "drift", "events"])
def test_failed_capture_never_seals_a_receipt(tmp_path: Path, failure: str) -> None:
    """Refuse a success receipt after incomplete pages, duplicate IDs or source drift."""
    features, drought = fixture_payloads(tmp_path)
    config = ObservatoryConfig(first_year=2016, target_year=2018)
    output = tmp_path / "failed"
    with source_server(features, drought, failure) as endpoints:
        with pytest.raises(CaliforniaError):
            acquire_sources(config, output, endpoints=endpoints, timeout=30)
    assert not output.joinpath("sources_receipt.json").exists()
    assert output.joinpath("capture_design.json").is_file()


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'"oversized"']
)
def test_strict_reader_rejects_corrupted_bytes(tmp_path: Path, raw: bytes) -> None:
    """Reject duplicate JSON fields, nonfinite numbers and byte-budget violations."""
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(CaliforniaError):
        strict_bytes(path, limit=8 if raw == b'"oversized"' else 1024)


def test_strict_reader_refuses_symlink(tmp_path: Path) -> None:
    """Refuse a JSON input that redirects custody through a symlink."""
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    with pytest.raises(CaliforniaError):
        strict_bytes(link)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest(target)
