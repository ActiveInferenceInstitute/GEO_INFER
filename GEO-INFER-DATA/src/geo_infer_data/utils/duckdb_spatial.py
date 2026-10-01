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
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import probe
    import duckdb as _duckdb  # type: ignore[import-not-found]

    HAS_DUCKDB: bool = True
    _DUCKDB = _duckdb
except ImportError:  # pragma: no cover - exercised when duckdb is missing
    _DUCKDB = None
    HAS_DUCKDB = False


class DuckDBSpatialError(RuntimeError):
    """Raised when an explicit DuckDB-Spatial read fails."""


def _fallback_read_vector(
    file_path: Path,
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a vector file through GeoPandas/Fiona (always available)."""
    if layer:
        return gpd.read_file(str(file_path), layer=layer, **kwargs)
    return gpd.read_file(str(file_path), **kwargs)


def _duckdb_read_vector(
    file_path: Path,
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a cloud-native vector file through DuckDB Spatial.

    GeoParquet and FlatGeobuf are read natively by DuckDB Spatial's ``ST_Read``
    and returned as a GeoDataFrame. ``layer``/extra kwargs are only supported
    by the fallback path.

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
        conn.execute("INSTALL spatial; LOAD spatial;")
        # The path is untrusted input interpolated into SQL, so it is
        # passed as a bound parameter. DuckDB builds that reject parameters
        # inside table functions fall back to standard SQL string-literal
        # escaping, where a doubled quote can never terminate the literal.
        posix_path = file_path.as_posix()
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
        geometry_column = df.pop("geom").map(
            lambda blob: bytes(blob) if not isinstance(blob, bytes) else blob
        )
        geometry = gpd.GeoSeries.from_wkb(geometry_column)
        return gpd.GeoDataFrame(
            df, geometry=geometry, crs=_resolve_crs(conn, posix_path)
        )
    finally:
        conn.close()


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
            _PyprojCRS(4326) if crs_meta == "default" else _crs_from_declared(crs_meta)
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
    layer: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read a GeoParquet / FlatGeobuf / Shapefile into a GeoDataFrame.

    Args:
        file_path: Path to the vector file.
        use_duckdb: When True (default) and DuckDB+Spatial is installed, use
            the DuckDB fast path; otherwise fall back to GeoPandas/Fiona.
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

    if use_duckdb and HAS_DUCKDB:
        try:
            return _duckdb_read_vector(path, layer=layer, **kwargs)
        except Exception as exc:  # pragma: no cover - engine dependent
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
    "read_cloud_native_vector",
    "duckdb_status",
]
