"""Cloud-native vector readers with optional DuckDB-Spatial acceleration.

Reads GeoParquet, FlatGeobuf, and Shapefile inputs into GeoDataFrames. When
DuckDB with the Spatial extension is installed, reads route through DuckDB for
fast cloud-native parsing; otherwise the reader transparently falls back to the
always-present GeoPandas/Fiona path. This mirrors the approach GeoLibre uses
for client-side vector import (DuckDB-WASM Spatial) while keeping GEO-INFER's
core importable and testable without the optional dependency.

The fallback keeps behaviour consistent across environments: callers read a
GeoDataFrame either way and never need to branch on which engine ran.
"""

from __future__ import annotations

import json
import logging
import hashlib
import math
import re
import tempfile
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import probe
    import duckdb as _duckdb  # type: ignore[import-not-found]

    HAS_DUCKDB: bool = True
    _DUCKDB = _duckdb
except ModuleNotFoundError as exc:  # exercised when duckdb is missing
    if exc.name != "duckdb":
        raise
    _DUCKDB = None
    HAS_DUCKDB = False


class DuckDBSpatialError(RuntimeError):
    """Raised when an explicit DuckDB-Spatial read fails."""


def _download_extension(
    url: str, target: Path, *, deadline: float, max_bytes: int
) -> dict[str, Any]:
    """Bound DNS, TLS, headers, body and decompression in one owned child."""
    from ._duckdb_extension_worker import validate_extension_url

    validate_extension_url(url)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DuckDBSpatialError("DuckDB extension download deadline exceeded")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("_duckdb_extension_worker.py")),
        url,
        str(target),
        str(remaining),
        str(max_bytes),
    ]
    try:
        result = subprocess.run(
            command,
            timeout=remaining,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for this worker before raising. The
        # stdlib-only worker spawns no descendants, including during DNS/TLS.
        diagnostics = exc.stderr or b""
        if isinstance(diagnostics, bytes):
            diagnostics = diagnostics.decode("utf-8", errors="replace")
        raise DuckDBSpatialError(
            "DuckDB extension download deadline exceeded; worker reaped\n" + diagnostics
        ) from exc
    if result.returncode != 0:
        raise DuckDBSpatialError(
            f"DuckDB extension download failed (exit {result.returncode}):\n"
            + result.stderr
        )
    try:
        evidence = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise DuckDBSpatialError("Extension worker returned invalid evidence") from exc
    if (
        not isinstance(evidence, dict)
        or evidence.get("download_url") != url
        or type(evidence.get("bytes")) is not int
        or not 0 < evidence["bytes"] <= max_bytes
        or not isinstance(evidence.get("sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", evidence["sha256"])
    ):
        raise DuckDBSpatialError("Extension worker returned inconsistent evidence")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as archive:
        while chunk := archive.read(64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise DuckDBSpatialError("Extension output exceeds byte budget")
            digest.update(chunk)
            if time.monotonic() >= deadline:
                raise DuckDBSpatialError("DuckDB extension download deadline exceeded")
    if time.monotonic() >= deadline:
        raise DuckDBSpatialError("DuckDB extension download deadline exceeded")
    if size != evidence["bytes"] or digest.hexdigest() != evidence["sha256"]:
        raise DuckDBSpatialError("Extension worker output does not match its evidence")
    return evidence


def provision_spatial_extension(
    *, download_timeout: float = 120.0, max_extension_bytes: int = 512 * 1024 * 1024
) -> dict[str, Any]:
    """Install official signed HTTPFS and Spatial for the actual runtime ABI.

    Download through Python HTTPS before local installation: a fresh DuckDB
    cannot bootstrap its HTTPS repository without HTTPFS already installed.
    Both archives must download before installation begins. Network, archive,
    signature and analytical failures propagate; no fallback certifies them.
    The shared download deadline and compressed/decompressed byte budgets are
    explicit. DuckDB enforces signatures with unsigned extensions disabled.
    """
    if (
        isinstance(download_timeout, bool)
        or not isinstance(download_timeout, (int, float))
        or not math.isfinite(download_timeout)
        or download_timeout <= 0
    ):
        raise ValueError("download_timeout must be finite and positive")
    if type(max_extension_bytes) is not int or max_extension_bytes <= 0:
        raise ValueError("max_extension_bytes must be a positive integer")
    if _DUCKDB is None:
        raise DuckDBSpatialError(
            "Install geo-infer-data[integrations] before provisioning Spatial"
        )
    with (
        _DUCKDB.connect(
            config={
                "allow_unsigned_extensions": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            }
        ) as conn,
        tempfile.TemporaryDirectory(prefix="geo-infer-duckdb-") as temporary,
    ):
        version = conn.execute("SELECT version()").fetchone()[0]
        platform = conn.execute("PRAGMA platform").fetchone()[0]
        if not isinstance(version, str) or not re.fullmatch(
            r"v[0-9]+\.[0-9]+\.[0-9]+", version
        ):
            raise DuckDBSpatialError(
                "Official provisioning requires a stable DuckDB runtime"
            )
        if not isinstance(platform, str) or not re.fullmatch(r"[a-z0-9_]+", platform):
            raise DuckDBSpatialError(
                "DuckDB reported an unsupported extension platform"
            )
        deadline = time.monotonic() + download_timeout
        extensions = []
        for name in ("httpfs", "spatial"):
            target = Path(temporary) / f"{name}.duckdb_extension"
            try:
                evidence = _download_extension(
                    f"https://extensions.duckdb.org/{version}/{platform}/{target.name}.gz",
                    target,
                    deadline=deadline,
                    max_bytes=max_extension_bytes,
                )
            except DuckDBSpatialError as exc:
                exc.runtime_version = version
                exc.platform = platform
                raise
            extensions.append({"extension": name, **evidence})
        if time.monotonic() >= deadline:
            raise DuckDBSpatialError("DuckDB extension download deadline exceeded")
        for evidence in extensions:
            name = evidence["extension"]
            conn.install_extension(
                str(Path(temporary) / f"{name}.duckdb_extension"), force_install=True
            )
            conn.load_extension(name)
        point = conn.execute("SELECT ST_AsText(ST_Point(1, 2))").fetchone()
        if point != ("POINT (1 2)",):
            raise DuckDBSpatialError(
                "Spatial extension failed its analytical smoke check"
            )
    return {
        "duckdb_version": _DUCKDB.__version__,
        "runtime_version": version,
        "platform": platform,
        "extension": "spatial",
        "extensions": extensions,
        "signature_verification": "duckdb-enforced",
        "status": "verified",
    }


def _fallback_read_vector(
    file_path: Path,
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a vector file through GeoPandas/Fiona (always available)."""
    if file_path.suffix.lower() in (".parquet", ".pq"):
        from .dependencies import require_dependency

        require_dependency("pyarrow", "integrations")
        if layer is not None:
            raise ValueError("GeoParquet does not accept a layer parameter")
        return gpd.read_parquet(str(file_path), **kwargs)
    if layer:
        return gpd.read_file(str(file_path), layer=layer, **kwargs)
    return gpd.read_file(str(file_path), **kwargs)


def _duckdb_read_vector(
    file_path: Path,
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a cloud-native vector file through DuckDB Spatial.

    GeoParquet uses native ``read_parquet`` with declared WKB geometry columns;
    other formats use Spatial's ``ST_Read``. ``layer``/extra kwargs are only
    supported by the fallback path.

    The CRS is taken from the file itself — GeoParquet key-value metadata when
    present (defaulting to OGC:CRS84 / EPSG:4326 only when the spec-default
    metadata omits ``crs``), otherwise the geometry column's SRID — so both
    read engines report the same CRS for the same file.
    """
    del layer, kwargs
    if _DUCKDB is None:  # pragma: no cover - guarded by callers
        raise DuckDBSpatialError("DuckDB is not installed")
    conn = _DUCKDB.connect()
    try:
        conn.execute("LOAD spatial;")
        # The path is untrusted input interpolated into SQL, so it is
        # passed as a bound parameter. DuckDB builds that reject parameters
        # inside table functions fall back to standard SQL string-literal
        # escaping, where a doubled quote can never terminate the literal.
        posix_path = file_path.as_posix()
        if file_path.suffix.lower() in (".parquet", ".pq"):
            return _duckdb_read_geoparquet(conn, posix_path)
        try:
            rel = conn.execute("SELECT * FROM ST_Read(?)", [posix_path])
        except _DUCKDB.Error:
            escaped = posix_path.replace("'", "''")
            rel = conn.execute(f"SELECT * FROM ST_Read('{escaped}')")
        df = rel.df()
        # ST_Read exposes a feature-id column the GeoPandas/Fiona fallback
        # does not produce; drop it so both engines return the same columns.
        df = df.drop(columns=["OGC_FID"], errors="ignore")
        # Some DuckDB versions materialise BLOB columns as bytearray objects,
        # which shapely's from_wkb rejects; normalise every element to bytes.
        geometry_column = df.pop("geom").map(_wkb_bytes)
        geometry = gpd.GeoSeries.from_wkb(geometry_column)
        return gpd.GeoDataFrame(
            df, geometry=geometry, crs=_resolve_crs(conn, posix_path)
        )
    finally:
        conn.close()


def _wkb_bytes(blob: Any) -> bytes | None:
    """Normalize DuckDB BLOB/GEOMETRY and its pandas null representation."""
    if blob is None or blob is pd.NA or (isinstance(blob, float) and math.isnan(blob)):
        return None
    return bytes(blob)


def _duckdb_read_geoparquet(conn: Any, path: str) -> gpd.GeoDataFrame:
    """Read declared WKB columns through Parquet, without depending on GDAL.

    Geometry names remain DataFrame labels, never SQL identifiers. DuckDB's
    native GEOMETRY conversion and older BLOB readers both expose WKB bytes.
    Original Arrow fields repair native type changes (notably TIMESTAMPTZ's
    microsecond precision), and pandas metadata restores index and field dtypes.
    Geometry and compatible fields still come from the native DuckDB reader.
    All declared geometry columns retain their own CRS; null CRS stays unknown.
    """
    rows = conn.execute(
        "SELECT key, value FROM parquet_kv_metadata(?)", [path]
    ).fetchall()
    metadata = None
    for key, value in rows:
        if (key.encode() if isinstance(key, str) else bytes(key)) == b"geo":
            if metadata is not None:
                raise DuckDBSpatialError("GeoParquet has duplicate geo metadata")
            metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise DuckDBSpatialError("GeoParquet requires geo metadata")
    primary = metadata.get("primary_column")
    columns = metadata.get("columns")
    if (
        not isinstance(primary, str)
        or not isinstance(columns, dict)
        or primary not in columns
    ):
        raise DuckDBSpatialError(
            "GeoParquet requires a declared primary geometry column"
        )
    df = _lossless_parquet_frame(conn, path, set(columns))
    for name, properties in columns.items():
        if (
            not isinstance(name, str)
            or name not in df
            or not isinstance(properties, dict)
        ):
            raise DuckDBSpatialError(
                "GeoParquet geometry metadata does not match its columns"
            )
        if properties.get("encoding") != "WKB":
            raise DuckDBSpatialError(
                "DuckDB GeoParquet reader supports declared WKB encoding"
            )
        crs = properties.get("crs", "default")
        resolved = (
            "OGC:CRS84"
            if crs == "default"
            else None
            if crs is None
            else _crs_from_declared(crs)
        )
        geometry = df[name].map(_wkb_bytes)
        df[name] = gpd.GeoSeries.from_wkb(geometry, crs=resolved)
    return gpd.GeoDataFrame(df, geometry=primary)


def _lossless_parquet_frame(
    conn: Any, path: str, geometry_columns: set[str]
) -> pd.DataFrame:
    """Combine native fields with exact original index/type compatibility reads.

    A single-file Parquet scan preserves insertion order explicitly. Original
    fields are selected only for physical indexes or changed Arrow types; the
    original schema's pandas metadata reconstructs labels, ranges, categories,
    nullable dtypes and timestamp zones without lossy pandas casts.
    """
    from .dependencies import require_dependency

    arrow = require_dependency("pyarrow", "integrations")
    parquet = require_dependency("pyarrow.parquet", "integrations")
    with parquet.ParquetFile(path) as original:
        schema = original.schema_arrow
        conn.execute("SET preserve_insertion_order = true")
        result = conn.execute("SELECT * FROM read_parquet(?)", [path]).arrow()
        native = result.read_all() if hasattr(result, "read_all") else result
        if (
            native.column_names != schema.names
            or len(set(schema.names)) != len(schema.names)
            or native.num_rows != original.metadata.num_rows
        ):
            raise DuckDBSpatialError(
                "Native and original Parquet rows/fields do not align"
            )
        pandas_metadata = schema.pandas_metadata
        indexes = (
            [] if pandas_metadata is None else pandas_metadata.get("index_columns", [])
        )
        if not isinstance(indexes, list):
            raise DuckDBSpatialError("Unsupported Parquet pandas index metadata")
        physical_indexes = set()
        for index in indexes:
            if (
                isinstance(index, str)
                and index in schema.names
                and index not in geometry_columns
            ):
                physical_indexes.add(index)
            elif (
                isinstance(index, dict)
                and len(indexes) == 1
                and index.get("kind") == "range"
            ):
                start, stop, step = (
                    index.get(name) for name in ("start", "stop", "step")
                )
                if (
                    any(type(value) is not int for value in (start, stop, step))
                    or step == 0
                    or len(range(start, stop, step)) != native.num_rows
                ):
                    raise DuckDBSpatialError(
                        "Parquet RangeIndex metadata does not match its rows"
                    )
            else:
                raise DuckDBSpatialError("Unsupported Parquet pandas index metadata")
        selected = [
            field.name
            for field in schema
            if field.name in physical_indexes
            or (
                field.name not in geometry_columns
                and field.type != native.schema.field(field.name).type
            )
        ]
        exact = original.read(columns=selected) if selected else None
        if exact is not None and exact.num_rows != native.num_rows:
            raise DuckDBSpatialError(
                "Original Parquet compatibility fields do not align"
            )
        arrays = [
            exact[field.name] if field.name in selected else native[field.name]
            for field in schema
        ]
        return arrow.Table.from_arrays(arrays, schema=schema).to_pandas()


def _resolve_crs(conn: Any, posix_path: str) -> Any:
    """Best-effort CRS lookup for a file already opened via ``ST_Read``.

    Returns a :class:`pyproj.CRS` when the file declares one, ``None``
    otherwise (matching what ``gpd.read_file`` would infer).
    """
    from pyproj import CRS as _PyprojCRS

    crs_meta = _geoparquet_metadata_crs(conn, posix_path)
    if crs_meta is not None:
        # GeoParquet spec: a column without an explicit "crs" entry defaults
        # to OGC:CRS84 (WGS84 lon/lat).
        return (
            _PyprojCRS("OGC:CRS84")
            if crs_meta == "default"
            else _crs_from_declared(crs_meta)
        )
    geometry_crs = _geometry_crs(conn, posix_path)
    if geometry_crs is not None:
        return geometry_crs
    return None


def _geoparquet_metadata_crs(conn: Any, posix_path: str) -> Any:
    """Return the declared CRS of a GeoParquet file, "default", or None."""
    try:
        rows = conn.execute(
            "SELECT key, value FROM parquet_kv_metadata(?)", [posix_path]
        ).fetchall()
    except Exception:
        return None
    for key, value in rows:
        key_bytes = key.encode() if isinstance(key, str) else key
        if key_bytes != b"geo":
            continue
        try:
            geo = json.loads(value)
            columns = geo.get("columns", {})
            primary = geo.get("primary_column")
            entries = (
                [columns[primary]] if primary in columns else list(columns.values())
            )
            for entry in entries:
                if "crs" in entry:
                    return entry["crs"]
                return "default"
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def _crs_from_declared(declared: Any) -> Any:
    """Build a pyproj CRS from a GeoParquet CRS declaration (PROJJSON or string)."""
    from pyproj import CRS as _PyprojCRS

    return _PyprojCRS.from_user_input(declared)


def _geometry_crs(conn: Any, posix_path: str) -> Any:
    """Resolve the CRS DuckDB Spatial reports for the geometry column.

    ``ST_CRS`` returns a string such as ``'EPSG:32610'`` on the DuckDB
    Spatial versions shipped with this module.
    """
    from pyproj import CRS as _PyprojCRS

    try:
        row = conn.execute(
            'SELECT ST_CRS("geom") FROM ST_Read(?) LIMIT 1', [posix_path]
        ).fetchone()
    except Exception:
        return None
    if row and row[0]:
        declared = row[0].strip() if isinstance(row[0], str) else row[0]
        if isinstance(declared, str):
            upper = declared.upper()
            if upper.startswith("EPSG:"):
                return _PyprojCRS.from_epsg(int(upper.split(":", 1)[1]))
        try:
            return _PyprojCRS.from_user_input(declared)
        except Exception:
            return None
    return None


def read_cloud_native_vector(
    file_path: str | Path,
    *,
    use_duckdb: bool = True,
    require_duckdb: bool = False,
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a GeoParquet / FlatGeobuf / Shapefile into a GeoDataFrame.

    Args:
        file_path: Path to the vector file.
        use_duckdb: When True (default) and DuckDB+Spatial is installed, use
            the DuckDB fast path; otherwise fall back to GeoPandas/Fiona.
        require_duckdb: Require the real fast path and propagate backend failures.
            Provision the matching extension explicitly before calling this mode.
        layer: Optional layer name (fallback path only).
        **kwargs: Extra kwargs forwarded to the reader.

    Returns:
        A GeoDataFrame with the file's features.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Not a regular file: {path}")

    if require_duckdb and not use_duckdb:
        raise ValueError("require_duckdb requires use_duckdb=True")
    if require_duckdb and not HAS_DUCKDB:
        raise DuckDBSpatialError(
            "DuckDB is not installed; use geo-infer-data[integrations]"
        )
    if layer is not None or kwargs:
        if require_duckdb:
            raise ValueError(
                "DuckDB Spatial does not support layer or reader keyword arguments"
            )
        return _fallback_read_vector(path, layer=layer, **kwargs)
    if use_duckdb and HAS_DUCKDB:
        try:
            return _duckdb_read_vector(path, layer=layer, **kwargs)
        except Exception as exc:  # pragma: no cover - engine dependent
            if require_duckdb:
                raise DuckDBSpatialError("Required DuckDB-Spatial read failed") from exc
            logger.warning("DuckDB-Spatial read failed (%s); falling back", exc)

    return _fallback_read_vector(path, layer=layer, **kwargs)


def duckdb_status() -> str:
    """Return the DuckDB-Spatial availability status for diagnostics."""
    if not HAS_DUCKDB:
        return "duckdb-spatial: unavailable (using GeoPandas/Fiona fallback)"
    try:
        return "duckdb-spatial: available"
    except Exception:  # pragma: no cover - defensive
        return "duckdb-spatial: unknown"


__all__ = [
    "HAS_DUCKDB",
    "DuckDBSpatialError",
    "provision_spatial_extension",
    "read_cloud_native_vector",
    "duckdb_status",
]
