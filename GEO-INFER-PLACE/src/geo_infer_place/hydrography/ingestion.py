"""Bounded, resumable snapshots of the USGS NHDPlus HR network flowline service.

A selection snapshots object IDs first; each page must return exactly its requested
IDs. This detects transfer limits and source deletions instead of certifying a
partial response as complete. Attribute edits during a download remain possible:
this service does not provide transactionally consistent historical snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit

import requests
from shapely.geometry import shape
from urllib3.exceptions import HTTPError

NHDPLUS_FLOWLINES_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/3"
)
SMITH_RIVER_HUC8 = "18010101"
SMITH_RIVER_PILOT_BBOX = (-124.22, 41.90, -124.18, 41.94)
US_CASCADIA_BBOX = (-124.8, 40.0, -114.5, 49.0)


class HydrographyError(RuntimeError):
    """A source request failed or its response violates the data contract."""


class IncompleteHydrographyError(HydrographyError):
    """An interrupted, truncated, or corrupted dataset cannot be certified."""


@dataclass(frozen=True)
class HydrographySelection:
    """Explicit WGS84 envelope and/or HUC8 reachcode prefix selection.

    A bounding box selects intersecting reaches, preserving whole geometries;
    it does not assert complete watershed coverage. HUC8 uses NHD reachcodes.
    """

    bbox: tuple[float, float, float, float] | None = None
    huc8: str | None = None

    def __post_init__(self) -> None:
        if self.bbox is None and self.huc8 is None:
            raise ValueError("An explicit bbox or huc8 selection is required")
        if self.huc8 is not None and not re.fullmatch(r"[0-9]{8}", self.huc8):
            raise ValueError("huc8 must contain exactly eight digits")
        if self.bbox is not None:
            if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
                raise ValueError("bbox must contain four finite coordinates")
            west, south, east, north = self.bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(
                    "bbox must be ordered WGS84 bounds without antimeridian crossing"
                )

    def query(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "where": f"reachcode LIKE '{self.huc8}%'" if self.huc8 else "1=1"
        }
        if self.bbox is not None:
            params.update(
                geometry=",".join(map(str, self.bbox)),
                geometryType="esriGeometryEnvelope",
                inSR=4326,
                spatialRel="esriSpatialRelIntersects",
            )
        return params


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def _write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class NHDPlusHRIngestor:
    """Fetch one selection with persisted verified pages and explicit resource caps.

    Reuse an output directory only for the identical source/selection/page size.
    One writer per output directory is required. HTTPS official USGS sources and
    loopback HTTP test servers are accepted; redirects are never followed.
    """

    def __init__(
        self,
        *,
        service_url: str = NHDPLUS_FLOWLINES_URL,
        page_size: int = 200,
        max_features: int = 10_000,
        max_bytes: int = 128 * 1024 * 1024,
        timeout: float = 45,
        max_duration: float = 180,
    ) -> None:
        url = urlsplit(service_url)
        official = url.scheme == "https" and url.hostname == "hydro.nationalmap.gov"
        local = url.scheme == "http" and url.hostname in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        if (
            not (official or local)
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Use an official USGS HTTPS layer or a loopback HTTP test server"
            )
        if (
            any(
                type(value) is not int for value in (page_size, max_features, max_bytes)
            )
            or not 1 <= page_size <= 2000
            or max_features < 1
            or max_bytes < 1
        ):
            raise ValueError(
                "page_size must be 1..2000; resource limits must be positive"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.service_url = service_url.rstrip("/")
        self.page_size = page_size
        self.max_features = max_features
        self.max_bytes = max_bytes
        if not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("max_duration must be finite and positive")
        self.timeout = timeout
        self.max_duration = max_duration

    def _request(self, params: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        started = time.monotonic()
        try:
            with requests.get(
                self.service_url + "/query",
                params=params,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise HydrographyError("USGS query returned an unexpected redirect")
                response.raise_for_status()
                payload = bytearray()
                while True:
                    if time.monotonic() - started > self.max_duration:
                        raise IncompleteHydrographyError(
                            "Request exceeded max_duration"
                        )
                    block = response.raw.read1(65536, decode_content=True)
                    if time.monotonic() - started > self.max_duration:
                        raise IncompleteHydrographyError(
                            "Request exceeded max_duration"
                        )
                    if not block:
                        break
                    payload.extend(block)
                    if len(payload) > self.max_bytes:
                        raise IncompleteHydrographyError("Response exceeds max_bytes")
                data = json.loads(payload)
        except (requests.RequestException, HTTPError, ValueError) as exc:
            raise HydrographyError(
                f"NHDPlus HR request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(data, dict) or "error" in data:
            raise HydrographyError(
                "USGS service returned an error or non-object response"
            )
        if data.get("exceededTransferLimit"):
            raise IncompleteHydrographyError(
                "USGS transfer limit exceeded; reduce page_size"
            )
        return data, bytes(payload)

    def ingest(self, selection: HydrographySelection, output_dir: str | Path) -> Path:
        """Return a complete GeoJSON path; failures retain resumable checkpoints.

        The manifest distinguishes complete, empty, failed, and incomplete. Every
        returned source field is retained. Canonical aliases are added when loaded.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        selection_data = json.loads(_json_bytes(asdict(selection)))
        identity = {
            "source_url": self.service_url,
            "selection": selection_data,
            "page_size": self.page_size,
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            if any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    "Output directory belongs to a different source, selection or page size"
                )
        else:
            manifest = {
                **identity,
                "schema_version": 1,
                "status": "incomplete",
                "acquired_at": datetime.now(UTC).isoformat(),
                "crs": "EPSG:4326",
                "pages": {},
                "coverage": "intersecting whole reaches; boundary crossings are not basin outlets",
                "snapshot_consistency": "object ID membership; attributes may change during retrieval",
            }
        try:
            if "object_ids" not in manifest:
                ids_data, _ = self._request(
                    {**selection.query(), "returnIdsOnly": "true", "f": "json"}
                )
                ids = ids_data.get("objectIds")
                if not isinstance(ids, list) or any(type(i) is not int for i in ids):
                    raise HydrographyError(
                        "USGS response is missing an integer objectIds list"
                    )
                if len(ids) != len(set(ids)):
                    raise IncompleteHydrographyError("Duplicate IDs in source snapshot")
                if not isinstance(ids_data.get("objectIdFieldName"), str):
                    raise HydrographyError("Missing object ID field name")
                manifest.update(
                    object_ids=sorted(ids),
                    object_id_field=ids_data["objectIdFieldName"],
                )
                _write(manifest_path, _json_bytes(manifest))
            ids = manifest["object_ids"]
            if len(ids) > self.max_features:
                raise IncompleteHydrographyError(
                    "Selection exceeds max_features; narrow selection or explicitly raise cap"
                )
            features = []
            consumed = 0
            for start in range(0, len(ids), self.page_size):
                requested = ids[start : start + self.page_size]
                name = f"page-{start:09d}.geojson"
                path = output / name
                previous = manifest["pages"].get(name)
                if previous:
                    if not path.is_file():
                        raise IncompleteHydrographyError(f"Cached page missing: {name}")
                    payload = path.read_bytes()
                    if _digest(payload) != previous["sha256"]:
                        raise IncompleteHydrographyError(
                            f"Cached page checksum mismatch: {name}"
                        )
                    data = json.loads(payload)
                else:
                    data, payload = self._request(
                        {
                            "objectIds": ",".join(map(str, requested)),
                            "outFields": "*",
                            "returnGeometry": "true",
                            "outSR": 4326,
                            "f": "geojson",
                        }
                    )
                consumed += len(payload)
                if consumed > self.max_bytes:
                    raise IncompleteHydrographyError("Dataset exceeds max_bytes")
                records = data.get("features")
                if data.get("type") != "FeatureCollection" or not isinstance(
                    records, list
                ):
                    raise HydrographyError("Expected GeoJSON FeatureCollection")
                actual = []
                for feature in records:
                    if (
                        not isinstance(feature, dict)
                        or feature.get("type") != "Feature"
                    ):
                        raise HydrographyError("Malformed GeoJSON feature")
                    properties = feature.get("properties")
                    geometry = feature.get("geometry")
                    if not isinstance(properties, dict) or not isinstance(
                        geometry, dict
                    ):
                        raise HydrographyError("Flowline lacks properties or geometry")
                    if geometry.get("type") not in {"LineString", "MultiLineString"}:
                        raise HydrographyError("Flowline geometry must be linear")
                    try:
                        line = shape(geometry)
                    except (ValueError, TypeError, KeyError) as exc:
                        raise HydrographyError(
                            "Malformed flowline coordinates"
                        ) from exc
                    if (
                        line.is_empty
                        or not line.is_valid
                        or not all(math.isfinite(v) for v in line.bounds)
                    ):
                        raise HydrographyError("Invalid or empty flowline geometry")
                    west, south, east, north = line.bounds
                    if not (
                        -180 <= west <= east <= 180 and -90 <= south <= north <= 90
                    ):
                        raise HydrographyError(
                            "Flowline coordinates are outside WGS84 bounds"
                        )
                    for key in ("nhdplusid", "fromnode", "tonode"):
                        value = properties.get(key)
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            or int(value) != value
                        ):
                            raise HydrographyError(
                                f"Invalid native topology attribute: {key}"
                            )
                    length = properties.get("lengthkm")
                    if (
                        isinstance(length, bool)
                        or not isinstance(length, (int, float))
                        or not math.isfinite(length)
                        or length < 0
                    ):
                        raise HydrographyError("Invalid native lengthkm")
                    order = properties.get("streamorde")
                    if order is not None and (
                        isinstance(order, bool)
                        or not isinstance(order, (int, float))
                        or not math.isfinite(order)
                        or int(order) != order
                    ):
                        raise HydrographyError("Invalid native streamorde")
                    actual.append(properties.get(manifest["object_id_field"]))
                if len(actual) != len(requested) or set(actual) != set(requested):
                    raise IncompleteHydrographyError(
                        "Page IDs differ from source snapshot"
                    )
                if not previous:
                    _write(path, payload)
                    manifest["pages"][name] = {
                        "sha256": _digest(payload),
                        "bytes": len(payload),
                        "feature_count": len(records),
                    }
                    _write(manifest_path, _json_bytes(manifest))
                features.extend(records)
            native_ids = [feature["properties"]["nhdplusid"] for feature in features]
            if len(native_ids) != len(set(native_ids)):
                raise IncompleteHydrographyError(
                    "Duplicate native NHDPlusID across pages"
                )
            result = {"type": "FeatureCollection", "features": features}
            payload = _json_bytes(result)
            if len(payload) > self.max_bytes:
                raise IncompleteHydrographyError("Serialized dataset exceeds max_bytes")
            _write(output / "flowlines.geojson", payload)
            manifest.update(
                status="complete" if features else "empty",
                feature_count=len(features),
                sha256=_digest(payload),
                dataset="flowlines.geojson",
                bytes=len(payload),
            )
            manifest.pop("error", None)
            _write(manifest_path, _json_bytes(manifest))
            return output / "flowlines.geojson"
        except (HydrographyError, OSError, ValueError, KeyError, TypeError) as exc:
            manifest.update(
                status="incomplete"
                if manifest["pages"] or isinstance(exc, IncompleteHydrographyError)
                else "failed",
                error=str(exc),
            )
            _write(manifest_path, _json_bytes(manifest))
            raise
