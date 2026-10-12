"""GeoLibre ``.geolibre.json`` project writer.

Renders GEO-INFER results (H3 grid cells, GeoJSON layers, raster/tile sources)
into the GeoLibre project JSON format so they can be opened in the GeoLibre
web, desktop, or Jupyter viewer. The emitted schema mirrors GeoLibre's
documented project format (``docs/project-format.md`` in opengeos/GeoLibre) and
its Python reference builders (``project.py``), format version ``0.1.0``.

The writer is intentionally deterministic and dependency-free: it accepts plain
GeoJSON FeatureCollections and emits plain JSON. All layer ids are assigned
from sequential indices (``layer-0``, ``layer-1``, ...) so equal inputs always
produce byte-identical projects, which keeps the output suitable for the same
deterministic-receipt contract that ``visualization_receipt`` enforces.

See also: :func:`geo_infer_space.core.visualization_receipt.write_visualization_receipt`
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

# Mirrors GeoLibre's DEFAULT_PROJECT_PREFERENCES (packages/core/src/types.ts).
DEFAULT_PROJECT_PREFERENCES: dict[str, Any] = {
    "map": {
        "restrictBounds": False,
        "bounds": [-180, -85, 180, 85],
        "minZoom": 0,
        "maxZoom": 24,
        "maxPitch": 85,
        "renderWorldCopies": True,
    },
    "environmentVariables": [],
}

# Mirrors GeoLibre's DEFAULT_LAYER_STYLE (packages/core/src/types.ts).
DEFAULT_LAYER_STYLE: dict[str, Any] = {
    "minZoom": 0,
    "maxZoom": 24,
    "fillColor": "#3b82f6",
    "strokeColor": "#1e40af",
    "strokeWidth": 2,
    "fillOpacity": 0.6,
    "circleRadius": 6,
    "textColor": "#111827",
    "textHaloColor": "#ffffff",
    "textHaloWidth": 2,
    "textSize": 16,
    "extrusionEnabled": False,
    "extrusionColor": "#3b82f6",
    "extrusionOpacity": 0.8,
    "extrusionHeightProperty": "height",
    "extrusionHeightScale": 1,
    "extrusionBase": 0,
    "extrusionAdvancedStyleEnabled": False,
    "extrusionColorExpression": "",
    "extrusionHeightExpression": "",
    "vectorStyleMode": "single",
    "vectorStyleProperty": "",
    "vectorStyleClassCount": 5,
    "vectorStyleColorRamp": "viridis",
    "vectorStyleClassificationScheme": "equal-interval",
    "vectorStyleStops": [
        {"value": 0, "color": "#dbeafe"},
        {"value": 1, "color": "#2563eb"},
    ],
    "vectorStyleExpression": "",
    "pointRenderer": "single",
    "heatmapRadius": 30,
    "heatmapIntensity": 1,
    "clusterRadius": 50,
    "clusterMaxZoom": 14,
    "rasterBrightnessMin": 0,
    "rasterBrightnessMax": 1,
    "rasterSaturation": 0,
    "rasterContrast": 0,
    "rasterHueRotate": 0,
}

# Format version matched to GeoLibre's project schema.
GEOLIBRE_PROJECT_VERSION: str = "0.1.0"

DEFAULT_BASEMAP_STYLE_URL: str = (
    "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)


def default_map_view(
    center: Sequence[float] | None = None,
    zoom: float | None = None,
) -> dict[str, Any]:
    """Return the app's default camera, optionally overridden.

    Args:
        center: Optional ``[lng, lat]`` map center (must have exactly 2 elements).
        zoom: Optional initial zoom level.

    Returns:
        A ``mapView`` dict (center, zoom, bearing, pitch).

    Raises:
        ValueError: If ``center`` does not have exactly 2 elements.
    """
    map_view: dict[str, Any] = {
        "center": [-100, 40],
        "zoom": 2,
        "bearing": 0,
        "pitch": 0,
    }
    if center is not None:
        if len(center) != 2:
            raise ValueError(
                "center must be a [lng, lat] sequence with exactly 2 elements"
            )
        map_view["center"] = [float(center[0]), float(center[1])]
    if zoom is not None:
        map_view["zoom"] = float(zoom)
    return map_view


def _layer_base(
    name: str, layer_type: str, layer_id: str, **style: Any
) -> dict[str, Any]:
    """Build the shared layer skeleton with a merged style dict.

    The default layer style is deep-copied so nested values are never shared
    across layers.
    """
    merged_style = {**copy.deepcopy(DEFAULT_LAYER_STYLE), **style}
    return {
        "id": layer_id,
        "name": name,
        "type": layer_type,
        "visible": True,
        "opacity": 1,
        "style": merged_style,
        "metadata": {},
    }


def geojson_layer(
    name: str,
    data: Mapping[str, Any],
    *,
    layer_id: str | None = None,
    index: int = 0,
    source_url: str | None = None,
    **style: Any,
) -> dict[str, Any]:
    """Build a GeoJSON layer with an inlined FeatureCollection.

    Args:
        name: Layer display name.
        data: A GeoJSON FeatureCollection dict.
        layer_id: Explicit layer id (defaults to ``layer-{index}``).
        index: Sequential index used for the default layer id.
        source_url: Optional URL the data originated from.
        **style: Style overrides merged into the default layer style.

    Returns:
        A layer dict for the project's ``layers`` array.
    """
    layer = _layer_base(name, "geojson", layer_id or f"layer-{index}", **style)
    source: dict[str, Any] = {"type": "geojson"}
    if source_url:
        source["url"] = source_url
        layer["sourcePath"] = source_url
    layer["source"] = source
    layer["geojson"] = data
    return layer


def tile_layer(
    name: str,
    url: str,
    *,
    layer_id: str | None = None,
    index: int = 0,
    tile_size: int = 256,
    attribution: str | None = None,
    **style: Any,
) -> dict[str, Any]:
    """Build a raster XYZ tile layer (``{z}/{x}/{y}`` template).

    Args:
        name: Layer display name.
        url: The XYZ tile URL template.
        layer_id: Explicit layer id (defaults to ``layer-{index}``).
        index: Sequential index for the default layer id.
        tile_size: Tile size in pixels (typically 256).
        attribution: Optional attribution string.
        **style: Style overrides merged into the default layer style.

    Returns:
        A layer dict for the project's ``layers`` array.
    """
    layer = _layer_base(name, "xyz", layer_id or f"layer-{index}", **style)
    source: dict[str, Any] = {
        "type": "raster",
        "tiles": [url],
        "tileSize": tile_size,
        "url": url,
    }
    if attribution:
        source["attribution"] = attribution
    layer["source"] = source
    layer["metadata"] = {"sourceKind": "xyz-url"}
    return layer


def build_project(
    name: str,
    layers: Sequence[Mapping[str, Any]],
    *,
    map_view: Mapping[str, Any] | None = None,
    center: Sequence[float] | None = None,
    zoom: float | None = None,
    basemap_style_url: str = DEFAULT_BASEMAP_STYLE_URL,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a GeoLibre project dict from layers.

    Args:
        name: Project display name.
        layers: Sequence of layer dicts (use :func:`geojson_layer`,
            :func:`tile_layer`, or the GeoLibre-compatible builders).
        map_view: Optional full ``mapView`` dict; if omitted one is derived
            from ``center``/``zoom``.
        center: Optional ``[lng, lat]`` center used when ``map_view`` is omitted.
        zoom: Optional zoom used when ``map_view`` is omitted.
        basemap_style_url: MapLibre style JSON URL for the basemap.
        metadata: Optional free-form project metadata.

    Returns:
        A project dict matching GeoLibre's ``.geolibre.json`` format v0.1.0.
    """
    if map_view is None:
        map_view = default_map_view(center=center, zoom=zoom)
    project: dict[str, Any] = {
        "version": GEOLIBRE_PROJECT_VERSION,
        "name": name,
        "mapView": copy.deepcopy(dict(map_view)),
        "basemapStyleUrl": basemap_style_url,
        "basemapVisible": True,
        "basemapOpacity": 1,
        "layers": [dict(layer) for layer in layers],
        "styles": {},
        "preferences": copy.deepcopy(DEFAULT_PROJECT_PREFERENCES),
        "metadata": dict(metadata or {}),
    }
    return project


def dumps_project(project: Mapping[str, Any]) -> str:
    """Serialise a GeoLibre project dict to pretty JSON.

    Args:
        project: A project dict from :func:`build_project`.

    Returns:
        A JSON string with a trailing newline, stable for equal inputs.
    """
    return json.dumps(project, indent=2, ensure_ascii=False) + "\n"


def write_project(project: Mapping[str, Any], path: Path | str) -> Path:
    """Write a GeoLibre project dict to a ``.geolibre.json`` file.

    Args:
        project: A project dict from :func:`build_project`.
        path: Output path. The ``.geolibre.json`` suffix is recommended.

    Returns:
        The resolved output :class:`Path`.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_project(project), encoding="utf-8")
    return out


def build_h3_grid_project(
    name: str,
    grid_geojson: Mapping[str, Any],
    *,
    center: Sequence[float] | None = None,
    zoom: float | None = None,
    fill_color: str = "#3b82f6",
    stroke_color: str = "#1e40af",
    fill_opacity: float = 0.4,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a GeoLibre project from an H3 grid GeoJSON FeatureCollection.

    Convenience wrapper for the common GEO-INFER case of visualising an H3
    grid (from :func:`geo_infer_space.polygon_to_cells` + cell boundaries).

    Args:
        name: Project display name.
        grid_geojson: A GeoJSON FeatureCollection of H3 cell polygons.
        center: Optional ``[lng, lat]`` map center.
        zoom: Optional initial zoom.
        fill_color: Hex fill color for the grid.
        stroke_color: Hex stroke color for the grid.
        fill_opacity: Fill opacity in ``[0, 1]``.
        metadata: Optional project metadata.

    Returns:
        A project dict with a single styled GeoJSON grid layer.
    """
    layer = geojson_layer(
        "H3 Grid",
        grid_geojson,
        fillColor=fill_color,
        strokeColor=stroke_color,
        fillOpacity=fill_opacity,
    )
    return build_project(
        name,
        [layer],
        center=center,
        zoom=zoom,
        metadata=metadata,
    )


__all__ = [
    "GEOLIBRE_PROJECT_VERSION",
    "DEFAULT_BASEMAP_STYLE_URL",
    "DEFAULT_LAYER_STYLE",
    "DEFAULT_PROJECT_PREFERENCES",
    "default_map_view",
    "geojson_layer",
    "tile_layer",
    "build_project",
    "dumps_project",
    "write_project",
    "build_h3_grid_project",
]
