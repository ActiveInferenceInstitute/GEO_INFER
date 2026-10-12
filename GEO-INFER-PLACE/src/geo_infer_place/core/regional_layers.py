"""Reproducible bounded US Cascadia display layers from official USGS services.

The fixed study window is not the whole Cascadia bioregion. No bioregion boundary
is inferred. Raw responses and source IDs are retained beside derived layers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode

from shapely.geometry import box, mapping, shape

from ._regional_download_worker import _MAX_BYTES, _validate_request

BOUNDS = (-124.8, 40.0, -114.5, 49.0)
TECTONIC_BOUNDS = (-130.0, 40.0, -120.0, 51.0)
MAX_BYTES = _MAX_BYTES
MAX_FEATURES = 1000
_WORKER_CLEANUP_SECONDS = 1.0
_BATCH_TIMEOUT_SECONDS = 300.0
LICENSE_URL = (
    "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
)
WBD_SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2"
TECTONIC_SERVICE = "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/Cascadia_tectonic_features/FeatureServer/5"
QUERY = dict(
    geometry=",".join(map(str, BOUNDS)),
    geometryType="esriGeometryEnvelope",
    inSR=4326,
    spatialRel="esriSpatialRelIntersects",
    outSR=4326,
    returnGeometry="true",
    f="geojson",
    resultRecordCount=MAX_FEATURES,
)
SOURCES = {
    "volcanoes": dict(
        url="https://volcanoes.usgs.gov/vsc/api/volcanoApi/geojson",
        raw="cascadia_volcanoes.source.json",
        output="cascadia_volcanoes.geojson",
        citation="https://volcanoes.usgs.gov/vsc/api/volcanoApi/",
    ),
    "watersheds": dict(
        url=WBD_SERVICE
        + "/query?"
        + urlencode(
            dict(
                QUERY,
                where="huc4 LIKE '17%' OR huc4 = '1801'",
                outFields="huc4,name,tnmid,areasqkm",
                maxAllowableOffset=0.002,
                geometryPrecision=6,
            )
        ),
        raw="cascadia_watersheds.source.json",
        output="cascadia_major_watersheds.geojson",
        citation="https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer",
    ),
    "tectonics": dict(
        url=TECTONIC_SERVICE
        + "/query?"
        + urlencode(
            dict(
                QUERY,
                geometry=",".join(map(str, TECTONIC_BOUNDS)),
                where="1=1",
                outFields="*",
            )
        ),
        raw="cascadia_tectonics.source.json",
        output="cascadia_subduction_zone.geojson",
        citation="https://www.usgs.gov/special-topics/subduction-zone-science/science/cascadia-subduction-zone-database",
    ),
}


def _json_bytes(data: dict) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _decode(raw: bytes) -> dict:
    if len(raw) > MAX_BYTES:
        raise ValueError("Regional data exceeds 20 MiB budget")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Source JSON numbers must be finite")
        return number

    def invalid_constant(value):
        raise ValueError(f"Nonfinite source JSON constant: {value}")

    data = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_float=finite_float,
        parse_constant=invalid_constant,
    )
    if (
        not isinstance(data, dict)
        or data.get("type") != "FeatureCollection"
        or not isinstance(data.get("features"), list)
    ):
        raise ValueError("Source must be a GeoJSON FeatureCollection")
    if data.get("exceededTransferLimit") or data.get("properties", {}).get(
        "exceededTransferLimit"
    ):
        raise ValueError("Source response was truncated")
    if not 1 <= len(data["features"]) <= MAX_FEATURES:
        raise ValueError("Source feature count exceeds bounds or is empty")
    crs = data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if crs not in {"EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84"}:
        raise ValueError("Source must use WGS84 coordinates")
    return data


def normalize_regional_layer(kind: str, raw: bytes) -> dict:
    """Validate and clip authoritative geometry without inventing missing layers.

    HU4 polygons retain HUC4 IDs and source areas, whose values describe whole
    source subregions. The subduction display includes only the source's named
    convergent Juan de Fuca boundary. Volcano status is not copied as a live alert.
    """
    if kind not in SOURCES:
        raise ValueError("Unknown regional source")
    data = _decode(raw)
    bounds = TECTONIC_BOUNDS if kind == "tectonics" else BOUNDS
    window = box(*bounds)
    features = []
    seen = set()
    for feature in data["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Invalid source feature")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Source properties must be an object")
        geometry = shape(feature.get("geometry"))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Invalid source geometry; no geometry repair is performed")
        west, south, east, north = geometry.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("Source geometry outside WGS84 bounds")
        if kind == "volcanoes":
            if geometry.geom_type != "Point":
                raise ValueError("Volcano source must contain points")
            identifier = str(properties["vnum"])
            display = dict(
                name=properties["volcanoName"],
                vnum=identifier,
                source_url=properties["volcanoUrl"],
                threat_level=properties.get("nvewsThreat", "Unknown").removesuffix(
                    " Threat"
                ),
                nvews_threat_source=properties.get("nvewsThreat", "Unknown"),
                last_major_eruption="Not provided by this source",
                lahar_risk_drainages=["Not provided by this source"],
            )
        elif kind == "watersheds":
            if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Watershed source must contain polygons")
            identifier = properties["huc4"]
            if (
                not isinstance(identifier, str)
                or len(identifier) != 4
                or not identifier.isdigit()
            ):
                raise ValueError("HUC4 identity must remain a four-digit string")
            if not (identifier.startswith("17") or identifier == "1801"):
                raise ValueError("Unexpected watershed outside declared HUC selection")
            display = dict(
                properties,
                source_area_sq_km=properties["areasqkm"],
                display_scope="HU4 subregion clipped to study window",
            )
        else:
            if geometry.geom_type not in {"LineString", "MultiLineString"}:
                raise ValueError("Tectonic source must contain lines")
            if (
                properties.get("Type") != "Convergent"
                or properties.get("Name") != "North America:Juan de Fuca"
            ):
                continue
            identifier = str(properties["OBJECTID"])
            display = dict(
                properties,
                name=properties["Name"],
                display_scope="USGS Juan de Fuca convergent boundary within offshore study window; not a hazard-probability surface",
            )
        if not geometry.intersects(window):
            continue
        clipped = geometry.intersection(window)
        if clipped.is_empty:
            continue
        expected_types = {
            "volcanoes": {"Point"},
            "watersheds": {"Polygon", "MultiPolygon"},
            "tectonics": {"LineString", "MultiLineString"},
        }[kind]
        if clipped.geom_type not in expected_types:
            # Mere boundary contact has no area (watersheds) or length
            # (tectonics), so it must not become a different display layer.
            lower_dimensions = {
                "volcanoes": set(),
                "watersheds": {"Point", "MultiPoint", "LineString", "MultiLineString"},
                "tectonics": {"Point", "MultiPoint"},
            }[kind]
            if clipped.geom_type in lower_dimensions:
                continue
            raise ValueError(
                f"Clipped {kind} geometry must retain its source geometry type"
            )
        if not clipped.is_valid:
            raise ValueError("Clipped source geometry is invalid")
        if identifier in seen:
            raise ValueError("Duplicate source identity")
        seen.add(identifier)
        features.append(
            dict(
                type="Feature",
                id=identifier,
                properties=display,
                geometry=mapping(clipped),
            )
        )
    if not features:
        raise ValueError("No source features overlap the requested study window")
    return dict(
        type="FeatureCollection",
        features=sorted(features, key=lambda f: f["id"]),
        provenance=dict(
            source_url=SOURCES[kind]["url"],
            source_sha256=hashlib.sha256(raw).hexdigest(),
            citation=SOURCES[kind]["citation"],
            license="USGS public domain; source attribution retained",
            license_url=LICENSE_URL,
            study_bounds=list(bounds),
            clipped=True,
            whole_cascadia_bioregion=False,
        ),
    )


def _stop_download_worker(process: subprocess.Popen[bytes]) -> None:
    """Kill the isolated worker and reap it within a separate cleanup budget."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=_WORKER_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Regional download worker could not be reaped within cleanup budget"
        ) from exc


def _download(url: str, deadline: float, remaining_bytes: int) -> bytes:
    """Supervise a private worker against the shared batch wall-clock deadline.

    Deadline expiry kills the worker instead of waiting for another HTTP chunk.
    Cleanup has a separate one-second budget. The worker starts no subprocesses;
    POSIX cleanup also kills its isolated process group. No downloaded bytes are
    returned unless the worker exits successfully within the parent deadline.
    """
    _validate_request(url, deadline, remaining_bytes)
    if time.monotonic() >= deadline:
        raise TimeoutError("Regional download exceeded its parent wall-clock deadline")
    worker = Path(__file__).with_name("_regional_download_worker.py")
    process = subprocess.Popen(
        [sys.executable, "-I", str(worker), url, repr(deadline), str(remaining_bytes)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        try:
            raw, errors = process.communicate(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired as exc:
            _stop_download_worker(process)
            raise TimeoutError(
                "Regional download exceeded its parent wall-clock deadline"
            ) from exc
        except BaseException:
            _stop_download_worker(process)
            raise
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Regional download exceeded its parent wall-clock deadline"
            )
        if process.returncode:
            try:
                message = json.loads(errors[:4096]).get(
                    "error", "Regional download worker failed"
                )
            except (ValueError, AttributeError):
                message = "Regional download worker failed"
            if process.returncode == 2:
                raise ValueError(message)
            if process.returncode == 4:
                raise TimeoutError(message)
            raise RuntimeError(message)
        if len(raw) > remaining_bytes:
            raise ValueError("Regional data exceeds 20 MiB budget")
        return raw
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def acquire_regional_layers(output_dir: str | Path, *, offline: bool = False) -> dict:
    """Fetch three fixed USGS sources or replay saved responses, with a receipt.

    Network workers are supervised against a shared five-minute wall-clock
    deadline, with at most one additional second to reap a terminated worker.
    Local geometry/serialization work checks the deadline between phases rather
    than interrupting an individual operation. Total persisted data is capped
    at 20 MiB. Output file
    replacement does not follow pre-existing file symlinks.
    The missing authentic whole-bioregion boundary remains explicitly unresolved.
    """
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    receipt = dict(
        schema_version="geo-infer-place/regional-layers/1",
        generated_at=datetime.now(UTC).isoformat(),
        study_bounds=list(BOUNDS),
        max_bytes=MAX_BYTES,
        max_features_per_source=MAX_FEATURES,
        layers={},
        tectonic_study_bounds=list(TECTONIC_BOUNDS),
        unresolved=[
            "Whole Cascadia bioregion boundary: no licensed vector source verified"
        ],
    )
    deadline = time.monotonic() + _BATCH_TIMEOUT_SECONDS
    for kind, spec in SOURCES.items():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Regional request batch exceeded its cooperative deadline"
            )
        if offline:
            with (destination / spec["raw"]).open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        else:
            raw = _download(
                spec["url"], deadline, MAX_BYTES - sum(map(len, files.values()))
            )
        normalized = normalize_regional_layer(kind, raw)
        encoded = _json_bytes(normalized)
        files[spec["raw"]] = raw
        files[spec["output"]] = encoded
        if sum(map(len, files.values())) > MAX_BYTES:
            raise ValueError("Regional data exceeds 20 MiB persisted budget")
        receipt["layers"][kind] = dict(
            source_url=spec["url"],
            raw_file=spec["raw"],
            output_file=spec["output"],
            source_sha256=hashlib.sha256(raw).hexdigest(),
            output_sha256=hashlib.sha256(encoded).hexdigest(),
            feature_count=len(normalized["features"]),
            source_feature_count=len(_decode(raw)["features"]),
        )
    files["cascadia_layers.provenance.json"] = _json_bytes(receipt)
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("Regional data exceeds 20 MiB persisted budget")
    if time.monotonic() >= deadline:
        raise TimeoutError("Regional request batch exceeded its cooperative deadline")
    for name, raw in files.items():
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
            stream.write(raw)
            temporary = Path(stream.name)
        temporary.replace(destination / name)
    return receipt


def main() -> int:
    """Acquire or replay the explicitly bounded regional display datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            acquire_regional_layers(args.output_dir, offline=args.offline), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
