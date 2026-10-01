"""
Point cloud data I/O operations for various 3D point data formats.

This module provides reading and writing capabilities for point cloud data
including LAS, LAZ, PLY, XYZ, and CSV formats. LAS/LAZ support requires
the optional ``laspy`` dependency; XYZ and CSV formats are handled with
NumPy and the Python standard library for zero-dependency fallback.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: laspy
# ---------------------------------------------------------------------------
try:
    import laspy  # type: ignore[import-untyped]

    HAS_LASPY = True
except ImportError:
    laspy = None
    HAS_LASPY = False
    logger.debug(
        "laspy is not installed. LAS/LAZ support is unavailable. "
        "Install with: pip install laspy[lazrs]"
    )

# ---------------------------------------------------------------------------
# Supported point cloud formats
# ---------------------------------------------------------------------------
SUPPORTED_POINT_CLOUD_FORMATS: dict[str, str] = {
    ".las": "LAS (LASer)",
    ".laz": "LAZ (Compressed LAS)",
    ".ply": "PLY (Polygon File Format)",
    ".xyz": "XYZ (Plain Text Points)",
    ".csv": "CSV (Comma-Separated Values)",
}


class PointCloudReader:
    """Reader class for point cloud data.

    Supports LAS, LAZ, PLY, XYZ, and CSV formats.  LAS/LAZ reading
    requires the optional ``laspy`` package.  All other formats are
    handled via NumPy and the standard library.

    Example::

        reader = PointCloudReader()
        data = reader.read("terrain.las")
        points = data['points']       # Nx3 (or Nx3+) float64 array
        cls    = data['classifications']  # optional Nx1 array or None
    """

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_POINT_CLOUD_FORMATS.copy()

    def read(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read point cloud data from *file_path*.

        Args:
            file_path: Path to a point cloud file.
            **kwargs: Format-specific keyword arguments forwarded to
                the underlying reader.

        Returns:
            A dictionary with the following keys:

            * ``points`` -- :class:`numpy.ndarray` of shape ``(N, 3)``
              (or ``(N, M)`` when extra dimensions are present).
            * ``classifications`` -- :class:`numpy.ndarray` of shape
              ``(N,)`` or ``None``.
            * ``intensities`` -- :class:`numpy.ndarray` of shape
              ``(N,)`` or ``None``.
            * ``colors`` -- :class:`numpy.ndarray` of shape ``(N, 3)``
              (RGB) or ``None``.
            * ``metadata`` -- ``dict`` with information about the file
              (format, point count, bounding box, etc.).

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file format is not supported or cannot
                be parsed.
            ImportError: If LAS/LAZ is requested but ``laspy`` is not
                installed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported point cloud format: {file_ext}. "
                f"Supported: {list(self.supported_formats.keys())}"
            )

        try:
            if file_ext in (".las", ".laz"):
                return self._read_las(file_path, **kwargs)
            elif file_ext == ".ply":
                return self._read_ply(file_path, **kwargs)
            elif file_ext == ".xyz":
                return self._read_xyz(file_path, **kwargs)
            elif file_ext == ".csv":
                return self._read_csv(file_path, **kwargs)
            else:
                raise ValueError(f"No reader implemented for {file_ext}")
        except (ImportError, ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error("Failed to read point cloud file %s: %s", file_path, e)
            raise

    # ------------------------------------------------------------------
    # LAS / LAZ
    # ------------------------------------------------------------------
    def _read_las(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
        """Read a LAS or LAZ file using *laspy*."""
        if not HAS_LASPY:
            raise ImportError(
                "laspy is required to read LAS/LAZ files. "
                "Install with: pip install laspy[lazrs]"
            )

        las = laspy.read(str(file_path), **kwargs)

        # Core XYZ coordinates
        points = np.column_stack(
            (
                np.asarray(las.x),
                np.asarray(las.y),
                np.asarray(las.z),
            )
        )

        # Optional attributes
        classifications: np.ndarray | None = None
        intensities: np.ndarray | None = None
        colors: np.ndarray | None = None

        if hasattr(las, "classification"):
            classifications = np.asarray(las.classification)

        if hasattr(las, "intensity"):
            intensities = np.asarray(las.intensity)

        # RGB colours (LAS point formats 2, 3, 5, 7, 8, 10)
        if all(hasattr(las, c) for c in ("red", "green", "blue")):
            colors = np.column_stack(
                (
                    np.asarray(las.red),
                    np.asarray(las.green),
                    np.asarray(las.blue),
                )
            )

        # Metadata
        header = las.header
        metadata: dict[str, Any] = {
            "format": "LAS" if file_path.suffix.lower() == ".las" else "LAZ",
            "point_count": int(header.point_count),
            "point_format_id": int(header.point_format.id),
            "version": f"{header.version.major}.{header.version.minor}",
            "bounds": {
                "min": [float(header.x_min), float(header.y_min), float(header.z_min)],
                "max": [float(header.x_max), float(header.y_max), float(header.z_max)],
            },
            "scale": list(header.scales),
            "offset": list(header.offsets),
            "crs_wkt": str(las.header.parse_crs())
            if hasattr(las.header, "parse_crs")
            else None,
        }

        logger.info(
            "Read %d points from %s (format: LAS %s, point format %d)",
            points.shape[0],
            file_path,
            metadata["version"],
            metadata["point_format_id"],
        )

        return {
            "points": points,
            "classifications": classifications,
            "intensities": intensities,
            "colors": colors,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # PLY
    # ------------------------------------------------------------------
    def _read_ply(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
        """Read a PLY file (ASCII or binary little-endian).

        A lightweight reader that does not require ``plyfile`` as a
        dependency.  It supports the most common vertex properties
        (x, y, z, nx, ny, nz, red, green, blue, intensity,
        classification/label).
        """
        with open(file_path, "rb") as fh:
            # ---- Parse header ------------------------------------------------
            header_lines: list[str] = []
            while True:
                line = fh.readline()
                if not line:
                    raise ValueError("Unexpected end of file in PLY header")
                decoded = line.decode("ascii", errors="replace").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break

            ply_format = "ascii"
            vertex_count = 0
            properties: list[tuple] = []  # (name, dtype_str)
            in_vertex_element = False

            for hline in header_lines:
                parts = hline.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    ply_format = parts[1]
                elif parts[0] == "element" and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex_element = True
                elif parts[0] == "element" and parts[1] != "vertex":
                    in_vertex_element = False
                elif parts[0] == "property" and in_vertex_element:
                    # property <type> <name>
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))

            if vertex_count == 0:
                logger.warning("PLY file contains 0 vertices: %s", file_path)
                return self._empty_result("PLY", file_path)

            # ---- Map PLY types to numpy dtypes --------------------------------
            _ply_to_numpy = {
                "float": "f4",
                "float32": "f4",
                "double": "f8",
                "float64": "f8",
                "uchar": "u1",
                "uint8": "u1",
                "char": "i1",
                "int8": "i1",
                "ushort": "u2",
                "uint16": "u2",
                "short": "i2",
                "int16": "i2",
                "uint": "u4",
                "uint32": "u4",
                "int": "i4",
                "int32": "i4",
            }

            np_dtype = np.dtype(
                [(name, _ply_to_numpy.get(ptype, "f4")) for name, ptype in properties]
            )

            # ---- Read vertex data --------------------------------------------
            if ply_format == "ascii":
                raw_data = np.loadtxt(fh, dtype=np_dtype, max_rows=vertex_count)
            elif ply_format in ("binary_little_endian", "binary_big_endian"):
                endian: Literal["<", ">"] = (
                    "<" if ply_format == "binary_little_endian" else ">"
                )
                np_dtype_endian = np_dtype.newbyteorder(endian)
                raw_data = np.frombuffer(
                    fh.read(np_dtype_endian.itemsize * vertex_count),
                    dtype=np_dtype_endian,
                    count=vertex_count,
                )
            else:
                raise ValueError(f"Unsupported PLY format: {ply_format}")

            # ---- Extract arrays ----------------------------------------------
            prop_names = [p[0] for p in properties]

            if not all(c in prop_names for c in ("x", "y", "z")):
                raise ValueError("PLY file is missing x/y/z vertex properties")

            points = np.column_stack(
                (
                    raw_data["x"].astype(np.float64),
                    raw_data["y"].astype(np.float64),
                    raw_data["z"].astype(np.float64),
                )
            )

            classifications: np.ndarray | None = None
            for cname in ("classification", "label", "class", "scalar_classification"):
                if cname in prop_names:
                    classifications = np.asarray(raw_data[cname])
                    break

            intensities: np.ndarray | None = None
            if "intensity" in prop_names:
                intensities = np.asarray(raw_data["intensity"])

            colors: np.ndarray | None = None
            if all(c in prop_names for c in ("red", "green", "blue")):
                colors = np.column_stack(
                    (
                        raw_data["red"].astype(np.uint8),
                        raw_data["green"].astype(np.uint8),
                        raw_data["blue"].astype(np.uint8),
                    )
                )

        metadata: dict[str, Any] = {
            "format": "PLY",
            "point_count": vertex_count,
            "ply_format": ply_format,
            "properties": prop_names,
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
            },
        }

        logger.info(
            "Read %d points from %s (PLY %s, %d properties)",
            vertex_count,
            file_path,
            ply_format,
            len(properties),
        )

        return {
            "points": points,
            "classifications": classifications,
            "intensities": intensities,
            "colors": colors,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # XYZ (whitespace-delimited plain text)
    # ------------------------------------------------------------------
    def _read_xyz(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
        """Read a plain-text XYZ file.

        Each line contains at least three whitespace-separated values
        representing X, Y, Z coordinates.  Additional columns (if
        present) are interpreted as intensity, classification, R, G, B
        in that order, or can be overridden via *kwargs*.

        Keyword Args:
            delimiter: Column delimiter (default ``None`` -- any
                whitespace).
            skip_header: Number of header lines to skip (default 0).
            columns: Explicit list of column names. Recognised names
                are ``x``, ``y``, ``z``, ``intensity``,
                ``classification``, ``red``, ``green``, ``blue``.
        """
        delimiter = kwargs.get("delimiter", None)
        skip_header = kwargs.get("skip_header", 0)
        columns: list[str] | None = kwargs.get("columns", None)
        comments = kwargs.get("comments", "#")

        data = np.loadtxt(
            str(file_path),
            delimiter=delimiter,
            skiprows=skip_header,
            comments=comments,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 3:
            raise ValueError(
                f"XYZ file must have at least 3 columns, found {data.shape[1]}"
            )

        # Assign column names
        if columns is None:
            default_cols = [
                "x",
                "y",
                "z",
                "intensity",
                "classification",
                "red",
                "green",
                "blue",
            ]
            columns = default_cols[: data.shape[1]]

        col_map: dict[str, int] = {name: idx for idx, name in enumerate(columns)}

        points = np.column_stack(
            (
                data[:, col_map["x"]],
                data[:, col_map["y"]],
                data[:, col_map["z"]],
            )
        )

        classifications: np.ndarray | None = None
        if "classification" in col_map:
            classifications = data[:, col_map["classification"]].astype(np.int32)

        intensities: np.ndarray | None = None
        if "intensity" in col_map:
            intensities = data[:, col_map["intensity"]]

        colors: np.ndarray | None = None
        if all(c in col_map for c in ("red", "green", "blue")):
            colors = np.column_stack(
                (
                    data[:, col_map["red"]],
                    data[:, col_map["green"]],
                    data[:, col_map["blue"]],
                )
            ).astype(np.uint8)

        metadata: dict[str, Any] = {
            "format": "XYZ",
            "point_count": int(points.shape[0]),
            "num_columns": int(data.shape[1]),
            "columns": columns,
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
            },
        }

        logger.info(
            "Read %d points from %s (XYZ, %d columns)",
            points.shape[0],
            file_path,
            data.shape[1],
        )

        return {
            "points": points,
            "classifications": classifications,
            "intensities": intensities,
            "colors": colors,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # CSV (comma-separated, with optional header)
    # ------------------------------------------------------------------
    def _read_csv(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
        """Read point cloud data from a CSV file.

        The CSV file should have a header row.  The reader auto-detects
        common column names for coordinates (``x``, ``y``, ``z``,
        ``easting``, ``northing``, ``elevation``, ``longitude``,
        ``latitude``, ``altitude``), intensity, classification, and
        RGB colour.

        Keyword Args:
            x_col: Name of the X/easting/longitude column.
            y_col: Name of the Y/northing/latitude column.
            z_col: Name of the Z/elevation/altitude column.
            delimiter: CSV delimiter (default ``','``).
        """
        delimiter = kwargs.get("delimiter", ",")
        x_col: str | None = kwargs.get("x_col", None)
        y_col: str | None = kwargs.get("y_col", None)
        z_col: str | None = kwargs.get("z_col", None)

        # Read header
        with open(file_path, encoding="utf-8") as fh:
            first_line = fh.readline().strip()

        headers = [h.strip().lower() for h in first_line.split(delimiter)]

        # Auto-detect coordinate columns
        _x_candidates = ["x", "easting", "longitude", "lon", "lng", "e"]
        _y_candidates = ["y", "northing", "latitude", "lat", "n"]
        _z_candidates = ["z", "elevation", "altitude", "alt", "height", "h"]

        def _find_col(explicit: str | None, candidates: list[str]) -> str | None:
            if explicit and explicit.lower() in headers:
                return explicit.lower()
            for c in candidates:
                if c in headers:
                    return c
            return None

        x_col_resolved = _find_col(x_col, _x_candidates)
        y_col_resolved = _find_col(y_col, _y_candidates)
        z_col_resolved = _find_col(z_col, _z_candidates)

        if not all((x_col_resolved, y_col_resolved, z_col_resolved)):
            raise ValueError(
                f"Could not identify X/Y/Z columns in CSV header: {headers}. "
                "Provide explicit x_col, y_col, z_col keyword arguments."
            )

        # Read the full file with numpy
        data = np.genfromtxt(
            str(file_path),
            delimiter=delimiter,
            skip_header=1,
            dtype=np.float64,
            filling_values=np.nan,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        xi = headers.index(x_col_resolved)  # type: ignore[arg-type]
        yi = headers.index(y_col_resolved)  # type: ignore[arg-type]
        zi = headers.index(z_col_resolved)  # type: ignore[arg-type]

        points = np.column_stack((data[:, xi], data[:, yi], data[:, zi]))

        # Optional columns
        classifications: np.ndarray | None = None
        for cname in ("classification", "class", "label"):
            if cname in headers:
                classifications = data[:, headers.index(cname)].astype(np.int32)
                break

        intensities: np.ndarray | None = None
        if "intensity" in headers:
            intensities = data[:, headers.index("intensity")]

        colors: np.ndarray | None = None
        if all(c in headers for c in ("red", "green", "blue")):
            colors = np.column_stack(
                (
                    data[:, headers.index("red")],
                    data[:, headers.index("green")],
                    data[:, headers.index("blue")],
                )
            ).astype(np.uint8)

        metadata: dict[str, Any] = {
            "format": "CSV",
            "point_count": int(points.shape[0]),
            "headers": headers,
            "coordinate_columns": {
                "x": x_col_resolved,
                "y": y_col_resolved,
                "z": z_col_resolved,
            },
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
            },
        }

        logger.info(
            "Read %d points from %s (CSV, columns: %s)",
            points.shape[0],
            file_path,
            headers,
        )

        return {
            "points": points,
            "classifications": classifications,
            "intensities": intensities,
            "colors": colors,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_result(fmt: str, file_path: Path) -> dict[str, Any]:
        """Return an empty result dictionary."""
        return {
            "points": np.empty((0, 3), dtype=np.float64),
            "classifications": None,
            "intensities": None,
            "colors": None,
            "metadata": {
                "format": fmt,
                "point_count": 0,
                "source": str(file_path),
            },
        }


class PointCloudWriter:
    """Writer class for point cloud data.

    Supports LAS, LAZ, PLY, XYZ, and CSV formats.  LAS/LAZ writing
    requires the optional ``laspy`` package.

    Example::

        writer = PointCloudWriter()
        writer.write(
            points=my_points,            # Nx3 numpy array
            file_path="output.las",
            classifications=my_labels,   # optional Nx1 array
            intensities=my_intensities,  # optional Nx1 array
        )
    """

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_POINT_CLOUD_FORMATS.copy()

    def write(
        self,
        points: np.ndarray,
        file_path: str | Path,
        classifications: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write point cloud data to *file_path*.

        Args:
            points: ``(N, 3)`` array of XYZ coordinates.
            file_path: Destination file path.  The format is inferred
                from the extension.
            classifications: Optional ``(N,)`` array of integer class
                labels.
            intensities: Optional ``(N,)`` array of intensity values.
            colors: Optional ``(N, 3)`` array of RGB colour values
                (uint8 or uint16).
            metadata: Optional metadata dict (used for CRS, scale,
                offset in LAS output).
            **kwargs: Format-specific keyword arguments.

        Raises:
            ValueError: If the format is not supported or *points* has
                wrong shape.
            ImportError: If LAS/LAZ is requested but ``laspy`` is not
                installed.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported point cloud format: {file_ext}. "
                f"Supported: {list(self.supported_formats.keys())}"
            )

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"points must be an (N, 3+) array, got shape {points.shape}"
            )

        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_ext in (".las", ".laz"):
                self._write_las(
                    points,
                    file_path,
                    classifications=classifications,
                    intensities=intensities,
                    colors=colors,
                    metadata=metadata,
                    **kwargs,
                )
            elif file_ext == ".ply":
                self._write_ply(
                    points,
                    file_path,
                    classifications=classifications,
                    intensities=intensities,
                    colors=colors,
                    **kwargs,
                )
            elif file_ext == ".xyz":
                self._write_xyz(
                    points,
                    file_path,
                    classifications=classifications,
                    intensities=intensities,
                    **kwargs,
                )
            elif file_ext == ".csv":
                self._write_csv(
                    points,
                    file_path,
                    classifications=classifications,
                    intensities=intensities,
                    colors=colors,
                    **kwargs,
                )

            logger.info(
                "Successfully wrote %d points to %s",
                points.shape[0],
                file_path,
            )

        except (ImportError, ValueError):
            raise
        except Exception as e:
            logger.error("Failed to write point cloud file %s: %s", file_path, e)
            raise

    # ------------------------------------------------------------------
    # LAS / LAZ
    # ------------------------------------------------------------------
    def _write_las(
        self,
        points: np.ndarray,
        file_path: Path,
        *,
        classifications: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write a LAS/LAZ file using *laspy*."""
        if not HAS_LASPY:
            raise ImportError(
                "laspy is required to write LAS/LAZ files. "
                "Install with: pip install laspy[lazrs]"
            )

        metadata = metadata or {}

        # Determine point format: 0 (basic), 2 (with colour), 6 (1.4 basic), 7 (1.4 + colour)
        file_version = kwargs.get("file_version", "1.4")
        if colors is not None:
            point_format_id = kwargs.get(
                "point_format_id", 7 if file_version == "1.4" else 2
            )
        else:
            point_format_id = kwargs.get(
                "point_format_id", 6 if file_version == "1.4" else 0
            )

        point_format = laspy.PointFormat(point_format_id)

        if file_version == "1.4":
            header = laspy.LasHeader(
                point_format=point_format, version=laspy.Version(1, 4)
            )
        else:
            header = laspy.LasHeader(
                point_format=point_format, version=laspy.Version(1, 2)
            )

        # Scale & offset
        scale = metadata.get("scale", [0.001, 0.001, 0.001])
        offset = metadata.get("offset", points.min(axis=0).tolist())
        header.scales = np.asarray(scale, dtype=np.float64)
        header.offsets = np.asarray(offset, dtype=np.float64)

        las_data = laspy.LasData(header)
        las_data.x = points[:, 0]
        las_data.y = points[:, 1]
        las_data.z = points[:, 2]

        if classifications is not None:
            las_data.classification = np.asarray(classifications, dtype=np.uint8)

        if intensities is not None:
            las_data.intensity = np.asarray(intensities, dtype=np.uint16)

        if colors is not None and all(
            dim in point_format.dimension_names for dim in ("red", "green", "blue")
        ):
            colors_arr = np.asarray(colors)
            # Scale uint8 colours to uint16 as required by LAS spec
            if colors_arr.max() <= 255:
                colors_arr = colors_arr.astype(np.uint16) * 257
            las_data.red = colors_arr[:, 0]
            las_data.green = colors_arr[:, 1]
            las_data.blue = colors_arr[:, 2]

        las_data.write(str(file_path))

    # ------------------------------------------------------------------
    # PLY
    # ------------------------------------------------------------------
    def _write_ply(
        self,
        points: np.ndarray,
        file_path: Path,
        *,
        classifications: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        binary: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write an ASCII or binary-little-endian PLY file."""
        n = points.shape[0]
        properties = ["x", "y", "z"]
        dtypes = [("x", "f8"), ("y", "f8"), ("z", "f8")]

        if intensities is not None:
            properties.append("intensity")
            dtypes.append(("intensity", "f4"))
        if classifications is not None:
            properties.append("classification")
            dtypes.append(("classification", "u1"))
        if colors is not None:
            for c in ("red", "green", "blue"):
                properties.append(c)
                dtypes.append((c, "u1"))

        ply_format = "binary_little_endian" if binary else "ascii"

        # Build header
        header_lines = [
            "ply",
            f"format {ply_format} 1.0",
            f"element vertex {n}",
        ]

        _np_to_ply = {
            "f8": "double",
            "f4": "float",
            "u1": "uchar",
            "u2": "ushort",
            "i4": "int",
            "u4": "uint",
        }

        for prop_name, dt in dtypes:
            ply_type = _np_to_ply.get(dt, "float")
            header_lines.append(f"property {ply_type} {prop_name}")

        header_lines.append("end_header")
        header_str = "\n".join(header_lines) + "\n"

        # Build structured array
        vertex_dtype = np.dtype(dtypes)
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]

        if intensities is not None:
            vertices["intensity"] = np.asarray(intensities, dtype=np.float32)
        if classifications is not None:
            vertices["classification"] = np.asarray(classifications, dtype=np.uint8)
        if colors is not None:
            color_array = np.asarray(colors, dtype=np.uint8)
            vertices["red"] = color_array[:, 0]
            vertices["green"] = color_array[:, 1]
            vertices["blue"] = color_array[:, 2]

        with open(file_path, "wb") as fh:
            fh.write(header_str.encode("ascii"))
            if binary:
                vertices.tofile(fh)
            else:
                for row in vertices:
                    line = " ".join(str(row[p]) for p in properties) + "\n"
                    fh.write(line.encode("ascii"))

    # ------------------------------------------------------------------
    # XYZ
    # ------------------------------------------------------------------
    def _write_xyz(
        self,
        points: np.ndarray,
        file_path: Path,
        *,
        classifications: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        delimiter: str = " ",
        precision: int = 6,
        **kwargs: Any,
    ) -> None:
        """Write a plain-text XYZ file."""
        columns = [points[:, 0], points[:, 1], points[:, 2]]

        if intensities is not None:
            columns.append(np.asarray(intensities))
        if classifications is not None:
            columns.append(np.asarray(classifications))

        data = np.column_stack(columns)

        fmt = delimiter.join([f"%.{precision}f"] * 3)
        if intensities is not None:
            fmt += delimiter + f"%.{precision}f"
        if classifications is not None:
            fmt += delimiter + "%d"

        np.savetxt(str(file_path), data, fmt=fmt)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def _write_csv(
        self,
        points: np.ndarray,
        file_path: Path,
        *,
        classifications: np.ndarray | None = None,
        intensities: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        delimiter: str = ",",
        precision: int = 6,
        **kwargs: Any,
    ) -> None:
        """Write point cloud data to a CSV file with a header row."""
        header_parts = ["x", "y", "z"]
        columns = [points[:, 0], points[:, 1], points[:, 2]]
        fmt_parts = [f"%.{precision}f"] * 3

        if intensities is not None:
            header_parts.append("intensity")
            columns.append(np.asarray(intensities))
            fmt_parts.append(f"%.{precision}f")

        if classifications is not None:
            header_parts.append("classification")
            columns.append(np.asarray(classifications, dtype=np.float64))
            fmt_parts.append("%d")

        if colors is not None:
            for i, c in enumerate(("red", "green", "blue")):
                header_parts.append(c)
                columns.append(np.asarray(colors[:, i], dtype=np.float64))
                fmt_parts.append("%d")

        data = np.column_stack(columns)
        header = delimiter.join(header_parts)
        fmt = delimiter.join(fmt_parts)

        np.savetxt(
            str(file_path),
            data,
            fmt=fmt,
            header=header,
            comments="",  # suppress the '#' prefix on the header line
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def read_point_cloud_file(
    file_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read point cloud data from file using the appropriate reader.

    This is a convenience wrapper around :class:`PointCloudReader`.

    Args:
        file_path: Path to a point cloud file.
        **kwargs: Format-specific keyword arguments.

    Returns:
        Dictionary with ``points``, ``classifications``, ``intensities``,
        ``colors``, and ``metadata`` keys.
    """
    reader = PointCloudReader()
    return reader.read(file_path, **kwargs)


def write_point_cloud_file(
    points: np.ndarray,
    file_path: str | Path,
    classifications: np.ndarray | None = None,
    intensities: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Write point cloud data to file using the appropriate writer.

    This is a convenience wrapper around :class:`PointCloudWriter`.

    Args:
        points: ``(N, 3)`` array of XYZ coordinates.
        file_path: Destination file path.
        classifications: Optional ``(N,)`` classification array.
        intensities: Optional ``(N,)`` intensity array.
        colors: Optional ``(N, 3)`` RGB colour array.
        metadata: Optional metadata dict.
        **kwargs: Format-specific keyword arguments.
    """
    writer = PointCloudWriter()
    writer.write(
        points,
        file_path,
        classifications=classifications,
        intensities=intensities,
        colors=colors,
        metadata=metadata,
        **kwargs,
    )


def supported_point_cloud_formats() -> dict[str, str]:
    """Get dictionary of supported point cloud formats.

    Returns:
        Dictionary mapping file extensions (e.g. ``'.las'``) to
        human-readable format names.
    """
    return SUPPORTED_POINT_CLOUD_FORMATS.copy()
