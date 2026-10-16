"""Capture and verify the complete publisher-defined Cascadia watershed boundary.

This is Aquila Flower's named delineation, not a political/regulatory boundary.
Raw metadata and geometry remain byte-preserved; no clipping, simplification or
geometry repair is performed. Acquisition is explicit and bounded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import hashlib
from html import unescape
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from shapely.geometry import shape

from .regional_layers import MAX_BYTES, _decode, _download, _json_bytes

ITEM_ID = "c6497941559d433da98e286fb3f63551"
ITEM_URL = f"https://www.arcgis.com/home/item.html?id={ITEM_ID}"
SERVICE = "https://services.arcgis.com/qboYD3ru0louQq4F/arcgis/rest/services/Cascadia_Bioregion_Boundary/FeatureServer"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SCHEMA = "geo-infer-place/cascadia-boundary/1"
OUTPUT = "cascadia_bioregion_boundary.geojson"
RECEIPT = "cascadia_boundary.provenance.json"
NOTICE = "cascadia_boundary.license.md"
SOURCES = {
    "item": f"https://www.arcgis.com/sharing/rest/content/items/{ITEM_ID}?f=pjson",
    "service": SERVICE + "?f=pjson",
    "layer": SERVICE + "/0?f=pjson",
    "count": SERVICE + "/0/query?where=1%3D1&returnCountOnly=true&f=json",
    "ids": SERVICE + "/0/query?where=1%3D1&returnIdsOnly=true&f=json",
    "extent": SERVICE + "/0/query?where=1%3D1&returnExtentOnly=true&outSR=4326&f=json",
    "geometry": SERVICE
    + "/0/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson",
    "layer_final": SERVICE + "/0?f=pjson",
}
RAW_FILES = {
    key: (
        "cascadia_boundary.source.json"
        if key == "geometry"
        else f"cascadia_boundary.{key}.source.json"
    )
    for key in SOURCES
}


def _object(raw: bytes) -> dict[str, Any]:
    """Decode metadata strictly, including duplicate/nonfinite rejection."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate boundary metadata key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite boundary metadata")
        return number

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    data = json.loads(
        raw, object_pairs_hook=unique, parse_float=finite, parse_constant=invalid
    )
    if not isinstance(data, dict) or "error" in data:
        raise ValueError("Boundary source must be a successful JSON object")
    return data


def normalize_cascadia_boundary(captures: dict[str, bytes]) -> dict[str, Any]:
    """Verify the named source/license and complete IDs, retaining every vertex."""
    if set(captures) != set(SOURCES) or sum(map(len, captures.values())) > MAX_BYTES:
        raise ValueError("Boundary capture inventory or byte budget is invalid")
    metadata = {key: _object(raw) for key, raw in captures.items() if key != "geometry"}
    item, layer = metadata["item"], metadata["layer"]
    if (item.get("id"), item.get("owner"), item.get("url")) != (
        ITEM_ID,
        "aquilaflower",
        SERVICE,
    ):
        raise ValueError("Boundary publisher/item identity differs")
    license_text = " ".join(
        unescape(re.sub(r"<[^>]*>", "", item.get("licenseInfo", ""))).split()
    )
    expected = (
        "License: Attribution-NonCommercial-ShareAlike 4.0 International CC BY-NC-SA 4.0. "
        f"Details: {LICENSE_URL}"
    )
    if license_text != expected:
        raise ValueError("Boundary publisher license requires renewed review")
    for record in (metadata["service"], layer, metadata["layer_final"]):
        if record.get("serviceItemId") != ITEM_ID:
            raise ValueError("Boundary service and licensed item do not match")
    epoch = layer.get("editingInfo", {}).get("dataLastEditDate")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise ValueError("Boundary source edit epoch is unavailable")
    final = metadata["layer_final"]
    if final.get("editingInfo", {}).get("dataLastEditDate") != epoch or final.get(
        "objectIdField"
    ) != layer.get("objectIdField"):
        raise ValueError("Boundary source changed during capture")
    if (
        layer.get("id") != 0
        or final.get("id") != 0
        or layer.get("geometryType") != "esriGeometryPolygon"
    ):
        raise ValueError("Boundary layer identity or geometry kind differs")
    count = metadata["count"].get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("Boundary source count must be an integer")
    ids = metadata["ids"].get("objectIds")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 1000:
        raise ValueError("Boundary source IDs are absent or exceed the feature budget")
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("Boundary source IDs must be unique integers")
    if metadata["count"].get("count") != len(ids):
        raise ValueError("Boundary source count differs from ID inventory")
    id_field = layer.get("objectIdField")
    if (
        not isinstance(id_field, str)
        or metadata["ids"].get("objectIdFieldName") != id_field
    ):
        raise ValueError("Boundary source ID field differs")
    data = _decode(captures["geometry"])
    if len(data["features"]) != len(ids):
        raise ValueError("Boundary response does not contain the complete source")
    received = []
    bounds = []
    for feature in data["features"]:
        if feature.get("type") != "Feature" or not isinstance(
            feature.get("properties"), dict
        ):
            raise ValueError("Invalid boundary feature")
        identifier = feature["properties"].get(id_field)
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise ValueError("Invalid native boundary identity")
        received.append(identifier)
        geometry = shape(feature.get("geometry"))
        if (
            geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise ValueError("Boundary geometry must be valid nonempty polygons")
        west, south, east, north = geometry.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("Boundary geometry must be WGS84")
        bounds.append((west, south, east, north))
    if sorted(received) != sorted(ids):
        raise ValueError("Boundary source IDs are missing or duplicated")
    actual = [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]
    extent = metadata["extent"].get("extent", {})
    if extent.get("spatialReference", {}).get("wkid") != 4326:
        raise ValueError("Boundary extent must be WGS84")
    expected_bounds = [extent.get(key) for key in ("xmin", "ymin", "xmax", "ymax")]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in expected_bounds
    ):
        raise ValueError("Boundary source extent is invalid")
    if not all(
        math.isclose(a, b, abs_tol=1e-8, rel_tol=0)
        for a, b in zip(actual, expected_bounds)
    ):
        raise ValueError("Boundary geometry does not cover the full source extent")
    return {
        **data,
        "provenance": {
            "schema_version": SCHEMA,
            "publisher": "Aquila Flower",
            "citation": ITEM_URL,
            "license": "CC-BY-NC-SA-4.0",
            "license_url": LICENSE_URL,
            "source_sha256": hashlib.sha256(captures["geometry"]).hexdigest(),
            "service_data_last_edit_ms": epoch,
            "native_ids": sorted(ids),
            "bounds": actual,
            "whole_cascadia_bioregion": True,
            "scope": "Complete Aquila Flower watershed delineation; not a political or regulatory border. Other display layers retain their narrower study windows.",
        },
    }


def _notice() -> bytes:
    return (
        "# Cascadia boundary attribution\n\n"
        f"Cascadia Bioregion Boundary, data analysis by Aquila Flower. [Publisher]({ITEM_URL}).\n\n"
        f"Publisher license: [CC BY-NC-SA 4.0]({LICENSE_URL}). Retain attribution, the license, "
        "and its noncommercial/share-alike conditions when redistributing this data.\n\n"
        "The publisher bases this watershed delineation on Cascadia Institute publications, "
        "CEC 2011 watersheds and CEC 2010 political/land boundaries. The original delineation "
        "used Lambert Azimuthal Equal Area; the hosted layer is Web Mercator and the captured "
        "GeoJSON query requests WGS84. Geometry is neither clipped, simplified nor repaired. "
        "Serialization and provenance metadata are added; original responses remain intact. "
        "Source area attributes retain their original meaning and are not geodesic areas.\n"
    ).encode()


def capture_cascadia_boundary(output_dir: str | Path) -> dict[str, Any]:
    """Acquire the fixed source within 20 MiB and one five-minute worker deadline.

    Existing captured files are never overwritten. All responses validate before
    writes start; the receipt is written last. Writes are not a multi-file
    transaction, and a failed attempt without its complete receipt is unqualified.
    """
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    names = {*RAW_FILES.values(), OUTPUT, NOTICE, RECEIPT}
    if any(
        (destination / name).exists() or (destination / name).is_symlink()
        for name in names
    ):
        raise FileExistsError("Choose a fresh boundary capture destination")
    deadline = time.monotonic() + 300
    captures: dict[str, bytes] = {}
    for key, url in SOURCES.items():
        captures[key] = _download(
            url, deadline, MAX_BYTES - sum(map(len, captures.values()))
        )
    data = normalize_cascadia_boundary(captures)
    files = {RAW_FILES[key]: raw for key, raw in captures.items()}
    files.update({OUTPUT: _json_bytes(data), NOTICE: _notice()})
    receipt = {
        "schema_version": SCHEMA,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_urls": SOURCES,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        "feature_count": len(data["features"]),
        "native_ids": data["provenance"]["native_ids"],
        "bounds": data["provenance"]["bounds"],
        "service_data_last_edit_ms": data["provenance"]["service_data_last_edit_ms"],
        "max_persisted_bytes": MAX_BYTES,
    }
    files[RECEIPT] = _json_bytes(receipt)
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("Boundary persisted capture exceeds its 20 MiB budget")
    if time.monotonic() >= deadline:
        raise TimeoutError("Boundary capture deadline exceeded before writing")
    for name, raw in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(raw)
    return verify_cascadia_boundary(destination)


def verify_cascadia_boundary(config_dir: str | Path) -> dict[str, Any]:
    """Rebuild and hash-check captured metadata, license and geometry offline."""
    root = Path(config_dir)
    files: dict[str, bytes] = {}
    total = 0
    for name in [RECEIPT, *RAW_FILES.values(), OUTPUT, NOTICE]:
        path = root / name
        if path.is_symlink():
            raise ValueError("Captured boundary files must not be symlinks")
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES - total + 1)
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError("Boundary capture exceeds its persisted budget")
        files[name] = raw
    receipt = _object(files.pop(RECEIPT))
    if receipt.get("max_persisted_bytes") != MAX_BYTES:
        raise ValueError("Boundary receipt byte budget differs")
    retrieved = datetime.fromisoformat(receipt.get("retrieved_at", ""))
    if retrieved.tzinfo is None or retrieved.utcoffset() != UTC.utcoffset(retrieved):
        raise ValueError("Boundary receipt retrieval time must be UTC")
    if receipt.get("schema_version") != SCHEMA or receipt.get("source_urls") != SOURCES:
        raise ValueError("Boundary receipt source contract differs")
    if set(receipt.get("files", {})) != set(files):
        raise ValueError("Boundary receipt inventory differs")
    if any(
        hashlib.sha256(raw).hexdigest() != receipt["files"][name]
        for name, raw in files.items()
    ):
        raise ValueError("Boundary capture hash mismatch")
    normalized = normalize_cascadia_boundary(
        {key: files[name] for key, name in RAW_FILES.items()}
    )
    if files[OUTPUT] != _json_bytes(normalized) or files[NOTICE] != _notice():
        raise ValueError("Boundary output or attribution differs from source replay")
    facts = normalized["provenance"]
    if not isinstance(receipt.get("feature_count"), int) or isinstance(
        receipt.get("feature_count"), bool
    ):
        raise ValueError("Boundary receipt feature count must be an integer")
    if any(
        receipt.get(key) != facts[key]
        for key in ("native_ids", "bounds", "service_data_last_edit_ms")
    ) or receipt.get("feature_count") != len(normalized["features"]):
        raise ValueError("Boundary receipt facts differ from complete source")
    return receipt


def main() -> int:
    """Capture explicitly or check the existing publisher snapshot offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (
        capture_cascadia_boundary(args.output_dir)
        if args.capture
        else verify_cascadia_boundary(args.output_dir)
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
