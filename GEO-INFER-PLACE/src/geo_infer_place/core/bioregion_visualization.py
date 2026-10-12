"""Bioregion-scale Folium map for Cascadia ecological overview.

Generates a lightweight (~1-3 MB) interactive HTML map with:
- Supplied H3 cells colored by ecoregion
- Volcano markers with threat-level colors and eruption info
- Supplied subduction-zone geometry without inferred hazard probabilities
- Watershed polygons with salmon ESU info
- Tribal nation markers
- Bioregion boundary outline
- Salmon ESU sidebar panel
"""

from __future__ import annotations

import json
from html import escape

from shapely.geometry import shape
import logging
from pathlib import Path
from typing import Any

import folium
import h3
import yaml

logger = logging.getLogger(__name__)

# Threat-level color mapping for volcanoes
_VOLCANO_COLORS = {
    "Very High": "red",
    "High": "orange",
    "Moderate": "yellow",
    "Low": "green",
}

# Ecoregion color palette (rotates through a set of muted greens/blues)
_ECOREGION_COLORS = [
    "#2d6a4f",
    "#40916c",
    "#52b788",
    "#74c69d",
    "#95d5b2",
    "#b7e4c7",
    "#1e6091",
    "#2a9d8f",
    "#457b9d",
    "#6096ba",
]


def _escaped(value: Any) -> Any:
    """Escape source labels before interpolation into HTML popups."""
    if isinstance(value, str):
        return escape(value, quote=True)
    if isinstance(value, list):
        return [_escaped(item) for item in value]
    if isinstance(value, dict):
        return {key: _escaped(item) for key, item in value.items()}
    return value


def _load_json(path: Path) -> dict[str, Any]:
    """Validate WGS84 GeoJSON layers before rendering."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("type") != "FeatureCollection"
        or not isinstance(data.get("features"), list)
    ):
        raise ValueError(f"Expected GeoJSON FeatureCollection: {path.name}")
    crs_name = data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if crs_name not in {
        "EPSG:4326",
        "urn:ogc:def:crs:OGC:1.3:CRS84",
        "urn:ogc:def:crs:EPSG::4326",
    }:
        raise ValueError(f"GeoJSON layer requires WGS84 coordinates: {path.name}")
    allowed = {
        "cascadia_bioregion_boundary.geojson": {"Polygon", "MultiPolygon"},
        "cascadia_major_watersheds.geojson": {"Polygon", "MultiPolygon"},
        "cascadia_subduction_zone.geojson": {"LineString", "MultiLineString"},
        "cascadia_volcanoes.geojson": {"Point"},
    }
    for feature in data["features"]:
        geometry = shape(feature.get("geometry"))
        if (
            geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in allowed.get(path.name, {geometry.geom_type})
        ):
            raise ValueError(f"Invalid layer geometry: {path.name}")
        west, south, east, north = geometry.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError(f"Geometry outside WGS84 bounds: {path.name}")
    return _escaped(data)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path.name}")
    return _escaped(data)


def _get_esa_listed(salmon_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all ESA-listed salmon/steelhead entries."""
    listed = []
    for group in [
        "chinook_salmon",
        "coho_salmon",
        "steelhead",
        "sockeye_salmon",
        "chum_salmon",
        "other_species",
    ]:
        for entry in salmon_data.get(group, []):
            status = entry.get("esa_status", "")
            if status not in ("Not Listed", "Not Listed (Species of Concern)", ""):
                listed.append(
                    {
                        "name": entry["name"],
                        "status": status,
                        "group": group.replace("_", " ").title(),
                    }
                )
    return listed


def _build_salmon_sidebar(esa_species: list[dict[str, Any]]) -> str:
    """Build HTML sidebar listing all ESA-listed salmon species."""
    rows = []
    for sp in esa_species:
        color = "#c0392b" if sp["status"] == "Endangered" else "#e67e22"
        rows.append(
            f'<tr><td style="padding:2px 6px;font-size:11px;">{sp["name"]}</td>'
            f'<td style="padding:2px 6px;color:{color};font-size:11px;font-weight:bold;">'
            f"{sp['status']}</td>"
            f'<td style="padding:2px 6px;color:#666;font-size:10px;">{sp["group"]}</td></tr>'
        )
    rows_html = "\n".join(rows)
    return f"""
<div id="salmon-panel" style="
    position:fixed; top:10px; right:10px; z-index:9999;
    background:white; border:1px solid #ccc; border-radius:6px;
    padding:10px; max-height:400px; overflow-y:auto; width:360px;
    font-family:sans-serif; box-shadow:3px 3px 8px rgba(0,0,0,0.2);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <strong style="font-size:13px;">🐟 ESA-Listed Pacific Salmonids ({len(esa_species)})</strong>
    <button onclick="document.getElementById('salmon-panel').style.display='none'"
            style="border:none;background:none;cursor:pointer;font-size:14px;">✕</button>
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>
      <th style="text-align:left;font-size:11px;border-bottom:1px solid #eee;padding:2px 6px;">Name</th>
      <th style="text-align:left;font-size:11px;border-bottom:1px solid #eee;padding:2px 6px;">Status</th>
      <th style="text-align:left;font-size:11px;border-bottom:1px solid #eee;padding:2px 6px;">Group</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


def _add_integration_layers(
    m: Any,
    h3_data: dict[str, Any],
    integration_results: dict[str, Any],
) -> None:
    """Add optional GEO-INFER integration result layers to the Folium map.

    Adds H3 hexagon layers for seismic hazard, forest health, and ecosystem
    services when the corresponding integration results are available.
    """
    # Seismic hazard layer (CSZ hazard scores per hexagon)
    seismic = integration_results.get("seismic_risk", {})
    hazard_scores = seismic.get("hazard_scores", {}) if seismic.get("available") else {}
    if hazard_scores:
        seismic_layer = folium.FeatureGroup(name="CSZ Seismic Hazard (GEO-INFER-RISK)")
        max_score = max(hazard_scores.values()) or 1.0
        for cell_id, score in hazard_scores.items():
            boundary = h3.cell_to_boundary(cell_id)
            # boundary is list of (lat, lon) — folium Polygon expects [[lat, lon], ...]
            locations = [[lat, lon] for lat, lon in boundary]
            norm = float(score) / max_score
            r = int(255 * norm)
            g = int(100 * (1 - norm))
            b = 0
            color = f"#{r:02x}{g:02x}{b:02x}"
            folium.Polygon(
                locations=locations,
                color=color,
                weight=0.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                tooltip=f"CSZ hazard: {score:.3f}",
            ).add_to(seismic_layer)
        seismic_layer.add_to(m)
        logger.info("Added seismic hazard layer: %d hexagons", len(hazard_scores))

    # Forest health layer
    forest = integration_results.get("forest_health", {})
    forest_results = forest.get("results", {}) if forest.get("available") else {}
    if forest_results and isinstance(forest_results, dict):
        forest_layer = folium.FeatureGroup(name="Forest Health (GEO-INFER-FOREST)")
        skipped = 0
        for cell_id, cell_data in forest_results.items():
            try:
                score = float(
                    cell_data.get("score", cell_data)
                    if isinstance(cell_data, dict)
                    else cell_data
                )
                boundary = h3.cell_to_boundary(cell_id)
                locations = [[lat, lon] for lat, lon in boundary]
                g = int(180 * score)
                color = f"#00{g:02x}00"
                folium.Polygon(
                    locations=locations,
                    color=color,
                    weight=0.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.35,
                    tooltip=f"Forest health: {score:.2f}",
                ).add_to(forest_layer)
            except (TypeError, KeyError, ValueError) as exc:
                skipped += 1
                logger.warning(
                    "Skipping malformed forest health hexagon %r: %s", cell_id, exc
                )
                continue
        forest_layer.add_to(m)
        logger.info(
            "Added forest health layer: %d hexagons (%d skipped)",
            len(forest_results) - skipped,
            skipped,
        )

    # Ecosystem services tooltip enrichment (added to h3_data hexagons)
    econ = integration_results.get("ecosystem_services", {})
    bank_summary = econ.get("bank_summary", {}) if econ.get("available") else {}
    if bank_summary:
        econ_layer = folium.FeatureGroup(name="Ecosystem Services (GEO-INFER-ECON)")
        credit_types = econ.get("credit_types", [])
        # Add a single info marker at bioregion centroid
        folium.Marker(
            location=[46.5, -121.5],
            icon=folium.DivIcon(
                html='<div style="font-size:12px;background:white;border:1px solid #888;'
                'border-radius:4px;padding:3px 6px;font-family:sans-serif;">'
                f"🌿 Ecosystem Credits: {', '.join(credit_types)}</div>",
                icon_size=(220, 28),
            ),
            tooltip=f"Natural capital bank active ({len(credit_types)} credit types)",
        ).add_to(econ_layer)
        econ_layer.add_to(m)


def create_bioregion_map(
    config_dir: Path,
    h3_data: dict[str, Any],
    output_path: Path,
    integration_results: dict[str, Any] | None = None,
    *,
    allow_missing_layers: bool = False,
) -> str:
    """Create a lightweight bioregional Folium map.

    Args:
        config_dir: Directory containing cascadia config files (GeoJSON/YAML).
        h3_data: Optional H3 hexagon data dict (cell_id -> props). May be empty.
        output_path: Where to write the HTML file.
        integration_results: Optional results from GEO-INFER integration suite.
            When present, adds enriched H3 layers (seismic hazard, ecosystem services,
            forest health) to the map.
        allow_missing_layers: When True, absent layer files are skipped instead
            of raising FileNotFoundError for the missing set.

    Returns:
        Absolute path of the generated HTML file as a string.
    """
    config_dir = Path(config_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load config data ---
    volcanoes_path = config_dir / "cascadia_volcanoes.geojson"
    csz_path = config_dir / "cascadia_subduction_zone.geojson"
    watersheds_path = config_dir / "cascadia_major_watersheds.geojson"
    boundary_path = config_dir / "cascadia_bioregion_boundary.geojson"
    ecoregions_path = config_dir / "cascadia_ecoregions.yaml"
    salmon_path = config_dir / "cascadia_salmon_esus.yaml"
    indigenous_path = config_dir / "cascadia_indigenous_territories.yaml"

    layer_paths = [volcanoes_path, csz_path, watersheds_path, boundary_path]
    missing_layers = [path.name for path in layer_paths if not path.is_file()]
    if missing_layers and not allow_missing_layers:
        raise FileNotFoundError(
            "Bioregion layers unavailable: " + ", ".join(missing_layers)
        )
    volcanoes_data = (
        _load_json(volcanoes_path) if volcanoes_path.exists() else {"features": []}
    )
    csz_data = _load_json(csz_path) if csz_path.exists() else {"features": []}
    watersheds_data = (
        _load_json(watersheds_path) if watersheds_path.exists() else {"features": []}
    )
    boundary_data = (
        _load_json(boundary_path) if boundary_path.exists() else {"features": []}
    )
    ecoregions_data = _load_yaml(ecoregions_path) if ecoregions_path.exists() else {}
    salmon_data = _load_yaml(salmon_path) if salmon_path.exists() else {}
    indigenous_data = _load_yaml(indigenous_path) if indigenous_path.exists() else {}

    # --- Build ecoregion color map ---
    ecoregion_list = ecoregions_data.get("ecoregions", [])
    ecoregion_color_map: dict[str, str] = {}
    for i, eco in enumerate(ecoregion_list):
        code = eco.get("code", eco.get("name", f"ECO{i}"))
        ecoregion_color_map[code] = _ECOREGION_COLORS[i % len(_ECOREGION_COLORS)]

    # --- Create base map (center on Cascadia bioregion) ---
    m = folium.Map(
        location=[46.5, -121.5],
        zoom_start=6,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    if h3_data:
        cells_layer = folium.FeatureGroup(name="H3 analysis cells")
        for cell, properties in h3_data.items():
            if not h3.is_valid_cell(cell):
                raise ValueError(f"Invalid H3 analysis cell: {cell}")
            code = (
                str(properties.get("ecoregion_code", "Unclassified"))
                if isinstance(properties, dict)
                else "Unclassified"
            )
            color = ecoregion_color_map.get(code, _ECOREGION_COLORS[0])
            folium.Polygon(
                locations=h3.cell_to_boundary(cell),
                color=color,
                fill=True,
                fill_opacity=0.3,
                weight=0.5,
                tooltip=escape(code),
            ).add_to(cells_layer)
        cells_layer.add_to(m)

    # --- Layer 1: Bioregion boundary ---
    if boundary_data.get("features"):
        boundary_layer = folium.GeoJson(
            boundary_data,
            name="Bioregion Boundary",
            style_function=lambda _: {
                "fillColor": "none",
                "color": "#555",
                "weight": 2,
                "dashArray": "4 4",
                "fillOpacity": 0,
            },
            tooltip="Cascadia Bioregion",
        )
        m.add_child(boundary_layer)

    # --- Layer 2: Watershed polygons ---
    if watersheds_data.get("features"):

        def _watershed_style(feature: dict) -> dict:
            return {
                "fillColor": "#1e6091",
                "color": "#0a4c78",
                "weight": 1.5,
                "fillOpacity": 0.25,
            }

        def _watershed_popup(feature: dict) -> folium.Popup:
            props = feature.get("properties", {})
            name = props.get("name", "Watershed")
            area = props.get("area_sq_mi", "?")
            salmon_count = props.get("salmon_esu_count", "?")
            major_dams = props.get("major_dams", [])
            dams_html = (
                "".join(f"<li>{d}</li>" for d in major_dams)
                if major_dams
                else "<li>None</li>"
            )
            html = f"""<div style="font-family:sans-serif;min-width:180px">
<b>💧 {name}</b><br>
<small>Area: {area} sq mi</small><br>
<small>Salmon ESUs: {salmon_count}</small><br>
<small>Major dams:<ul style="margin:2px 0 0 12px">{dams_html}</ul></small>
</div>"""
            return folium.Popup(html, max_width=220)

        watershed_layer = folium.GeoJson(
            watersheds_data,
            name="Major Watersheds",
            style_function=_watershed_style,
            popup=folium.GeoJsonPopup(
                fields=["name"],
                aliases=["Watershed:"],
                localize=True,
            ),
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Watershed:"]),
        )
        m.add_child(watershed_layer)

    # --- Layer 3: CSZ fault line ---
    if csz_data.get("features"):
        csz_layer = folium.GeoJson(
            csz_data,
            name="Cascadia Subduction Zone",
            style_function=lambda _: {
                "color": "#c0392b",
                "weight": 3,
                "dashArray": "6 4",
                "opacity": 0.9,
            },
            tooltip="Cascadia Subduction Zone",
            popup=folium.Popup(
                "Source-provided subduction-zone geometry; no hazard probability inferred.",
                max_width=230,
            ),
        )
        m.add_child(csz_layer)

    # --- Layer 4: Volcano markers ---
    volcano_layer = folium.FeatureGroup(name="Volcanoes")
    for feature in volcanoes_data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        threat = props.get("threat_level", "Unknown")
        color = _VOLCANO_COLORS.get(threat, "gray")
        name = props.get("name", "Unknown Volcano")
        elevation_m = props.get("elevation_m")
        last_eruption = props.get("last_major_eruption")
        lahars = [
            d
            for d in props.get("lahar_risk_drainages", [])
            if d != "Not provided by this source"
        ]
        rows = [
            "<small><b>Threat level:</b> "
            f'<span style="color:{color};font-weight:bold">{threat}</span></small>'
        ]
        if elevation_m is not None:
            rows.append(f"<small><b>Elevation:</b> {elevation_m} m</small>")
        if last_eruption and last_eruption != "Not provided by this source":
            rows.append(f"<small><b>Last major eruption:</b> {last_eruption}</small>")
        if lahars:
            rows.append(
                "<small><b>Lahar risk drainages:</b>"
                '<ul style="margin:2px 0 0 12px">'
                + "".join(f"<li>{d}</li>" for d in lahars)
                + "</ul></small>"
            )
        popup_html = (
            '<div style="font-family:sans-serif;min-width:180px">'
            f"<b>🌋 {name}</b><br>" + "<br>".join(rows) + "</div>"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"🌋 {name} ({threat})",
        ).add_to(volcano_layer)
    if volcanoes_data.get("features"):
        m.add_child(volcano_layer)

    # --- Layer 5: Indigenous territory markers ---
    indigenous_layer = folium.FeatureGroup(name="Indigenous Territories")
    for state_key in ("washington_state", "oregon_state", "california_tribes"):
        for nation in indigenous_data.get(state_key, []):
            t_lat = nation.get("latitude")
            t_lon = nation.get("longitude")
            t_name = nation.get("name", "Tribal Nation")
            if t_lat is None or t_lon is None:
                continue
            folium.CircleMarker(
                location=[t_lat, t_lon],
                radius=5,
                color="#8B4513",
                fill=True,
                fill_color="#CD853F",
                fill_opacity=0.7,
                weight=1.5,
                popup=folium.Popup(
                    f'<div style="font-family:sans-serif"><b>🏔️ {t_name}</b></div>',
                    max_width=200,
                ),
                tooltip=t_name,
            ).add_to(indigenous_layer)
    m.add_child(indigenous_layer)

    # --- Layer 6: Integration results (seismic hazard, forest health, ecosystem services) ---
    if integration_results:
        _add_integration_layers(m, h3_data, _escaped(integration_results))

    # --- Layer control ---
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Salmon ESU sidebar panel ---
    esa_species = _get_esa_listed(salmon_data)
    sidebar_html = _build_salmon_sidebar(esa_species)

    # --- Legend ---
    legend_html = """
<div style="position:fixed;bottom:30px;left:10px;z-index:9999;background:white;
     border:1px solid #ccc;border-radius:6px;padding:10px;font-family:sans-serif;
     font-size:12px;box-shadow:3px 3px 8px rgba(0,0,0,0.2);">
  <b>🌋 Volcano Threat</b><br>
  <span style="color:red">●</span> Very High &nbsp;
  <span style="color:orange">●</span> High<br>
  <span style="color:#ccaa00">●</span> Moderate &nbsp;
  <span style="color:green">●</span> Low<br>
  <hr style="margin:4px 0">
  <span style="color:#c0392b">——</span> Cascadia Subduction Zone<br>
  <span style="color:#1e6091">▪</span> Major Watersheds<br>
  <span style="color:#8B4513">●</span> Indigenous Territories<br>
  <span style="color:#555">- -</span> Bioregion Boundary
</div>
"""

    # Inject sidebar and legend as HTML elements
    m.get_root().html.add_child(folium.Element(sidebar_html))
    if missing_layers:
        missing_html = (
            "<div id='missing-layers' style='position:fixed;bottom:10px;left:10px;z-index:9999;background:white;padding:10px'>Unavailable layers: "
            + ", ".join(escape(name) for name in missing_layers)
            + "</div>"
        )
        m.get_root().html.add_child(folium.Element(missing_html))
    else:
        m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = """
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9998;
     background:white;border:1px solid #ccc;border-radius:6px;padding:8px 16px;
     font-family:sans-serif;font-size:15px;font-weight:bold;
     box-shadow:3px 3px 8px rgba(0,0,0,0.2);">
  🌲 Cascadia Bioregion — Ecological Overview
</div>
"""
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    m.save(str(output_path))
    layer_manifest = {
        "status": "partial" if missing_layers else "complete",
        "data_root": str(config_dir.resolve()),
        "layers": {
            path.name: {
                "status": "unavailable" if path.name in missing_layers else "loaded"
            }
            for path in layer_paths
        },
    }
    output_path.with_suffix(".layers.json").write_text(
        json.dumps(layer_manifest, indent=2) + "\n", encoding="utf-8"
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Bioregion map saved: %s (%.1f MB)", output_path, size_mb)
    return str(output_path.resolve())
