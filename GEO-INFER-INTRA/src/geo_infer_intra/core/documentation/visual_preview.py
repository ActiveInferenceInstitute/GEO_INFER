"""
Reproducible spatial visual preview generators for GEO-INFER modules.

Generates interactive Folium/Leaflet HTML preview cards, vector SVG cards,
PNG binary cards, and deterministic metadata receipts for all GEO-INFER
domain modules.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import h3
from PIL import Image, ImageDraw, PngImagePlugin

# Canonical specifications and spatial profiles for all 45 GEO-INFER modules
MODULE_PROFILES: dict[str, dict[str, Any]] = {
    "ACT": {
        "name": "Active Inference Engine",
        "description": "Free energy minimization, epistemic exploration, and action selection",
        "category": "Foundations",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#2563eb",
        "secondary_color": "#60a5fa",
        "features": ["Belief Posterior", "Free Energy Landscape", "Policy Prior"],
    },
    "AG": {
        "name": "Agricultural Systems",
        "description": "Precision crop analytics, irrigation management, and yield modeling",
        "category": "Domain",
        "center": [36.7783, -119.4179],
        "zoom": 10,
        "primary_color": "#16a34a",
        "secondary_color": "#4ade80",
        "features": ["NDVI Field Zonation", "Soil Moisture Grids", "Irrigation Flow"],
    },
    "AGENT": {
        "name": "Multi-Agent Systems",
        "description": "Agent lifecycle, emergent coordination, and stigmergic interactions",
        "category": "Optimization",
        "center": [37.7833, -122.4167],
        "zoom": 13,
        "primary_color": "#9333ea",
        "secondary_color": "#c084fc",
        "features": [
            "Agent Swarm Positions",
            "Interaction Trails",
            "Communication Radii",
        ],
    },
    "AI": {
        "name": "Artificial Intelligence Engine",
        "description": "Machine-learning embeddings, spatial neural operators, and surrogate models",
        "category": "Foundations",
        "center": [37.7600, -122.4400],
        "zoom": 12,
        "primary_color": "#4f46e5",
        "secondary_color": "#818cf8",
        "features": [
            "Latent Feature Space",
            "Neural Operator Field",
            "Prediction Confidence",
        ],
    },
    "ANT": {
        "name": "Ant Colony Optimization",
        "description": "Swarm routing, pheromone field decay, and stigmergic optimization",
        "category": "Optimization",
        "center": [37.7500, -122.4200],
        "zoom": 13,
        "primary_color": "#ea580c",
        "secondary_color": "#fb923c",
        "features": ["Pheromone Intensity", "Foraging Paths", "Nest-Source Vectors"],
    },
    "API": {
        "name": "API Management System",
        "description": "Service endpoints, GraphQL schemas, and gateway telemetry",
        "category": "Infrastructure",
        "center": [37.7900, -122.4000],
        "zoom": 14,
        "primary_color": "#0284c7",
        "secondary_color": "#38bdf8",
        "features": ["Gateway Topology", "Latency Isocurves", "Service Ingress"],
    },
    "APP": {
        "name": "Application Framework",
        "description": "Interactive dashboards, reactive spatial UI, and view state",
        "category": "Applications",
        "center": [37.7700, -122.4300],
        "zoom": 13,
        "primary_color": "#0d9488",
        "secondary_color": "#2dd4bf",
        "features": ["Dashboard Viewport", "Active Layers", "Widget Anchors"],
    },
    "ART": {
        "name": "Artificial Intelligence Art",
        "description": "Generative aesthetics, cartographic styling, and spatial rendering",
        "category": "Creative",
        "center": [37.8000, -122.4100],
        "zoom": 13,
        "primary_color": "#db2777",
        "secondary_color": "#f472b6",
        "features": ["Harmonic Field", "Color Ramp Boundary", "Flow Contours"],
    },
    "BAYES": {
        "name": "Bayesian Inference Engine",
        "description": "Spatial Gaussian processes, variational ELBO, and MCMC sampling",
        "category": "Foundations",
        "center": [37.7650, -122.4350],
        "zoom": 12,
        "primary_color": "#7c3aed",
        "secondary_color": "#a78bfa",
        "features": ["GP Posterior Mean", "Variational Uncertainty", "Inducing Points"],
    },
    "BIO": {
        "name": "Biological Systems",
        "description": "Biodiversity mapping, ecological niche modeling, and genomic dispersion",
        "category": "Domain",
        "center": [38.5000, -122.8000],
        "zoom": 11,
        "primary_color": "#059669",
        "secondary_color": "#34d399",
        "features": ["Species Occurrence", "Habitat Suitability", "Corridor Network"],
    },
    "CIV": {
        "name": "Civic Engagement",
        "description": "Participatory GIS, municipal sentiment, and community feedback loops",
        "category": "Domain",
        "center": [37.7750, -122.4180],
        "zoom": 13,
        "primary_color": "#d97706",
        "secondary_color": "#fbbf24",
        "features": ["Neighborhood Feedback", "Voting Stations", "Public Forums"],
    },
    "CLIMATE": {
        "name": "Climate Analysis Module",
        "description": "CMIP6 climate projections, extreme heat indices, and anomaly detection",
        "category": "Domain",
        "center": [37.5000, -122.0000],
        "zoom": 9,
        "primary_color": "#0891b2",
        "secondary_color": "#22d3ee",
        "features": [
            "Temperature Anomaly",
            "Precipitation Shift",
            "Heat Vulnerability",
        ],
    },
    "COG": {
        "name": "Cognitive Modeling",
        "description": "Spatial attention maps, multi-scale saliency, and working memory",
        "category": "Foundations",
        "center": [37.7800, -122.4250],
        "zoom": 13,
        "primary_color": "#6366f1",
        "secondary_color": "#818cf8",
        "features": ["Attention Saliency", "Cognitive Horizon", "Belief State Vector"],
    },
    "COMMS": {
        "name": "Communication Systems",
        "description": "Mesh network topology, telemetry routing, and bandwidth modeling",
        "category": "Infrastructure",
        "center": [37.7700, -122.4100],
        "zoom": 13,
        "primary_color": "#0284c7",
        "secondary_color": "#38bdf8",
        "features": ["Antenna Coverage", "Mesh Relay Nodes", "Signal Attenuation"],
    },
    "DATA": {
        "name": "Data Management Engine",
        "description": "GeoParquet chunking, secure GISP1 envelope validation, and storage",
        "category": "Foundations",
        "center": [37.7850, -122.4050],
        "zoom": 13,
        "primary_color": "#475569",
        "secondary_color": "#94a3b8",
        "features": ["Storage Partitions", "Cache Indexes", "Ingest Pipelines"],
    },
    "ECON": {
        "name": "Economic Analysis",
        "description": "Spatial econometrics, land value modeling, and trade networks",
        "category": "Domain",
        "center": [37.7920, -122.4010],
        "zoom": 14,
        "primary_color": "#ca8a04",
        "secondary_color": "#facc15",
        "features": ["Land Value Isobars", "Commercial Density", "Market Radius"],
    },
    "EDU": {
        "name": "Educational Technology Module",
        "description": "Geospatial curriculum tracks, interactive tutorials, and learner metrics",
        "category": "Domain",
        "center": [37.7700, -122.4500],
        "zoom": 13,
        "primary_color": "#2563eb",
        "secondary_color": "#60a5fa",
        "features": ["Learning Waypoints", "Exercise Boundaries", "Cohort Progress"],
    },
    "EMERGENCY": {
        "name": "Emergency Management Module",
        "description": "Evacuation routing, multi-hazard incident dispatch, and resource allocation",
        "category": "Domain",
        "center": [37.7600, -122.4300],
        "zoom": 12,
        "primary_color": "#dc2626",
        "secondary_color": "#f87171",
        "features": ["Evacuation Corridors", "Incident Perimeter", "Shelter Points"],
    },
    "ENERGY": {
        "name": "Energy Systems Module",
        "description": "Renewable potential, grid load balancing, and transmission routing",
        "category": "Domain",
        "center": [37.7000, -122.1500],
        "zoom": 10,
        "primary_color": "#eab308",
        "secondary_color": "#fde047",
        "features": ["Solar Irradiance", "Grid Substations", "Wind Potential"],
    },
    "EXAMPLES": {
        "name": "Cross-Module Examples",
        "description": "End-to-end integration exemplars, tutorials, and demonstration pipelines",
        "category": "Orchestration",
        "center": [37.7749, -122.4194],
        "zoom": 11,
        "primary_color": "#3b82f6",
        "secondary_color": "#93c5fd",
        "features": ["Workflow Stepper", "Cross-Domain Links", "Output Manifest"],
    },
    "FOREST": {
        "name": "Forest Management Module",
        "description": "Canopy cover analysis, fuel load estimation, and timber biomass",
        "category": "Domain",
        "center": [41.7500, -124.2000],
        "zoom": 10,
        "primary_color": "#15803d",
        "secondary_color": "#22c55e",
        "features": ["Canopy Density", "Fuel Load Index", "Timber Parcels"],
    },
    "GIT": {
        "name": "Git & Orchestration",
        "description": "Provenance DAGs, versioned spatial state, and reproducible workflows",
        "category": "Infrastructure",
        "center": [37.7800, -122.3900],
        "zoom": 14,
        "primary_color": "#334155",
        "secondary_color": "#64748b",
        "features": ["Commit Tree", "Branch Checkpoints", "Diff Polygons"],
    },
    "HEALTH": {
        "name": "Geospatial Health Analytics",
        "description": "Epidemiological diffusion, hospital accessibility, and environmental exposure",
        "category": "Domain",
        "center": [37.7600, -122.4400],
        "zoom": 12,
        "primary_color": "#e11d48",
        "secondary_color": "#fb7185",
        "features": [
            "Infection Clusters",
            "Care Access Isochrones",
            "Exposure Contours",
        ],
    },
    "INSURANCE": {
        "name": "Insurance Operations",
        "description": "Underwriting decisions, premium pricing, claims lifecycle, and portfolio capacity",
        "category": "Applications",
        "center": [37.7750, -122.4250],
        "zoom": 12,
        "primary_color": "#0e7490",
        "secondary_color": "#22d3ee",
        "features": [
            "Underwriting Case Flow",
            "Premium Surface",
            "Claim Reserves",
        ],
    },
    "INTRA": {
        "name": "Knowledge Integration",
        "description": "Cross-module ontology, documentation catalog, and visualization hub",
        "category": "Infrastructure",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#0ea5e9",
        "secondary_color": "#38bdf8",
        "features": ["Module Graph", "Ontology Concepts", "Catalog Registry"],
    },
    "IOT": {
        "name": "Internet of Things Integration",
        "description": "Streaming sensor ingestion, telemetry buffers, and anomaly alerts",
        "category": "Foundations",
        "center": [37.7700, -122.4150],
        "zoom": 13,
        "primary_color": "#06b6d4",
        "secondary_color": "#67e8f9",
        "features": ["Sensor Feeds", "Buffer Geometries", "Threshold Alerts"],
    },
    "LOG": {
        "name": "Logistics",
        "description": "Freight corridors, last-mile optimization, and fleet dispatch",
        "category": "Domain",
        "center": [37.8000, -122.2800],
        "zoom": 11,
        "primary_color": "#f97316",
        "secondary_color": "#fdba74",
        "features": ["Freight Corridors", "Distribution Hubs", "Route Efficiency"],
    },
    "MARINE": {
        "name": "Marine and Coastal Module",
        "description": "Bathymetry modeling, shoreline vulnerability, and ocean currents",
        "category": "Domain",
        "center": [37.8200, -122.5000],
        "zoom": 11,
        "primary_color": "#0284c7",
        "secondary_color": "#0ea5e9",
        "features": ["Bathymetric Contours", "Current Vectors", "Coastal Buffers"],
    },
    "MATH": {
        "name": "Mathematical Foundations",
        "description": "SIMD geometry kernels, spherical trigonometry, and numerical transforms",
        "category": "Foundations",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#8b5cf6",
        "secondary_color": "#c4b5fd",
        "features": ["SIMD Ray-Casting Grid", "Convex Hull", "Point-in-Polygon"],
    },
    "METAGOV": {
        "name": "Meta-Governance & Organizational Governance Module",
        "description": "Polycentric governance, institutional decision protocols, and auditability",
        "category": "Governance",
        "center": [37.7790, -122.4180],
        "zoom": 13,
        "primary_color": "#64748b",
        "secondary_color": "#94a3b8",
        "features": ["Jurisdiction Borders", "Policy Envelopes", "Voting Weights"],
    },
    "NORMS": {
        "name": "Norms & Standards",
        "description": "ISO/OGC geospatial standard compliance, metadata schemas, and validation",
        "category": "Governance",
        "center": [37.7720, -122.4170],
        "zoom": 13,
        "primary_color": "#475569",
        "secondary_color": "#64748b",
        "features": ["Zoning Constraints", "Standard Buffers", "Audit Checkpoints"],
    },
    "OPS": {
        "name": "Operations",
        "description": "Cluster orchestration, memory budgets, and task execution",
        "category": "Infrastructure",
        "center": [37.7850, -122.4000],
        "zoom": 13,
        "primary_color": "#334155",
        "secondary_color": "#475569",
        "features": ["Worker Nodes", "Job Allocation", "Resource Metrics"],
    },
    "ORG": {
        "name": "Organizational Systems",
        "description": "Enterprise structure, role definitions, and workflow delegation",
        "category": "Governance",
        "center": [37.7780, -122.4150],
        "zoom": 13,
        "primary_color": "#059669",
        "secondary_color": "#10b981",
        "features": ["Unit Boundaries", "Delegation Graph", "Access Scopes"],
    },
    "PEP": {
        "name": "People & Communities",
        "description": "Demographic equity, social vulnerability, and human mobility patterns",
        "category": "Domain",
        "center": [37.7600, -122.4200],
        "zoom": 12,
        "primary_color": "#f59e0b",
        "secondary_color": "#fbbf24",
        "features": ["Equity Index Cells", "Demographic Density", "Community Centers"],
    },
    "PLACE": {
        "name": "Place-Based Analysis",
        "description": "Cascadia regional bioregions, volcanoes, and watershed hydrography",
        "category": "Domain",
        "center": [45.5152, -122.6784],
        "zoom": 7,
        "primary_color": "#059669",
        "secondary_color": "#34d399",
        "features": [
            "Cascadia Volcanoes",
            "Megathrust Fault Line",
            "Watershed Hydrography",
        ],
    },
    "REQ": {
        "name": "Requirements Management",
        "description": "System specifications, verification traces, and constraint checking",
        "category": "Governance",
        "center": [37.7700, -122.4200],
        "zoom": 13,
        "primary_color": "#64748b",
        "secondary_color": "#94a3b8",
        "features": ["Requirement Zones", "Verification Status", "Trace Links"],
    },
    "RISK": {
        "name": "Risk Assessment",
        "description": "Directed multi-hazard interaction matrices and compound exceedance modeling",
        "category": "Foundations",
        "center": [37.7500, -122.4400],
        "zoom": 11,
        "primary_color": "#dc2626",
        "secondary_color": "#ef4444",
        "features": [
            "Compound Hazard Zone",
            "Exceedance Probability",
            "Vulnerability Surface",
        ],
    },
    "SEC": {
        "name": "Security",
        "description": "GISP1 cryptographic envelopes, geospatial token validation, and anonymization",
        "category": "Infrastructure",
        "center": [37.7800, -122.4100],
        "zoom": 13,
        "primary_color": "#1e293b",
        "secondary_color": "#334155",
        "features": ["Cryptographic Boundary", "Audit Points", "Anonymized Hexagons"],
    },
    "SIM": {
        "name": "Simulation",
        "description": "Digital twin environments, Monte Carlo rollouts, and physics simulation",
        "category": "Foundations",
        "center": [37.7650, -122.4250],
        "zoom": 12,
        "primary_color": "#7c3aed",
        "secondary_color": "#8b5cf6",
        "features": ["Digital Twin State", "Simulation Mesh", "Trajectory Envelopes"],
    },
    "SPACE": {
        "name": "Spatial Analysis Engine",
        "description": "H3 v4 indexing, spatial joins, CRS reprojection, and GeoLibre projects",
        "category": "Foundations",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#2563eb",
        "secondary_color": "#3b82f6",
        "features": ["H3 Resolution Cells", "CRS Bounds", "Vector Geometry"],
    },
    "SPM": {
        "name": "Spatial Process Modeling",
        "description": "Topological random field theory, Euler characteristic, and resel inference",
        "category": "Foundations",
        "center": [37.7550, -122.4350],
        "zoom": 12,
        "primary_color": "#4338ca",
        "secondary_color": "#6366f1",
        "features": [
            "Euler Characteristic Surface",
            "FWE Peak Threshold",
            "Cluster Resels",
        ],
    },
    "TEST": {
        "name": "Testing Framework",
        "description": "Unified test orchestrator, contract validators, and property suites",
        "category": "Infrastructure",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#15803d",
        "secondary_color": "#16a34a",
        "features": ["Test Coverage Cells", "Assertion Metrics", "Validation Gateways"],
    },
    "TIME": {
        "name": "Temporal Analysis Engine",
        "description": "Spatiotemporal streaming, dynamic windowing, and anomaly prediction",
        "category": "Foundations",
        "center": [37.7700, -122.4200],
        "zoom": 12,
        "primary_color": "#0284c7",
        "secondary_color": "#0ea5e9",
        "features": ["Temporal Trajectories", "Sliding Windows", "Anomaly Peaks"],
    },
    "TRANSPORT": {
        "name": "Transportation Systems Module",
        "description": "Multimodal transit routing, network flow analysis, and congestion relief",
        "category": "Domain",
        "center": [37.7749, -122.4194],
        "zoom": 12,
        "primary_color": "#d97706",
        "secondary_color": "#f59e0b",
        "features": ["Transit Corridors", "Traffic Flow Speeds", "Intermodal Hubs"],
    },
    "WATER": {
        "name": "Water Resource Module",
        "description": "Pollution plume dispersion, watershed flow routing, and groundwater aquifers",
        "category": "Domain",
        "center": [37.8000, -122.4000],
        "zoom": 11,
        "primary_color": "#0284c7",
        "secondary_color": "#38bdf8",
        "features": [
            "Watershed Catchment",
            "Pollution Plume Area",
            "Groundwater Aquifer",
        ],
    },
}


@dataclass(frozen=True)
class SpatialPreviewArtifacts:
    """Container for generated preview files and metadata."""

    module_id: str
    html_path: Path
    svg_path: Path
    png_path: Path
    manifest_path: Path
    input_sha256: str
    html_bytes: int
    svg_bytes: int
    png_bytes: int


RESOLUTION = 7
ILLUSTRATION_LABEL = (
    "Illustrative module overview; geometry computed with H3, no measured values."
)
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"


def _profile(module_id: str) -> dict[str, Any]:
    """Resolve a canonical module profile before using it in output paths."""
    if not isinstance(module_id, str) or module_id.upper() not in MODULE_PROFILES:
        raise ValueError(f"Unknown module ID: {module_id}")

    profile = MODULE_PROFILES[module_id.upper()]
    for field in ("primary_color", "secondary_color"):
        if not isinstance(profile[field], str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", profile[field]
        ):
            raise ValueError("Preview colors must be six-digit hex values")
    return profile


def _dimensions(width: int, height: int) -> None:
    """Bound canvas allocation and reject invalid CSS/SVG dimensions."""
    if any(type(v) is not int or not 1 <= v <= 4096 for v in (width, height)):
        raise ValueError("Canvas dimensions must be integers between 1 and 4096")


def _cells(module_id: str) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return real H3 cell identifiers and closed latitude/longitude boundaries."""
    center = h3.latlng_to_cell(*_profile(module_id)["center"], RESOLUTION)
    ids = [center, *sorted(set(h3.grid_disk(center, 1)) - {center})]
    result = []
    for cell in ids:
        boundary = list(h3.cell_to_boundary(cell))
        result.append((cell, [*boundary, boundary[0]]))
    return result


def _projected_rings(
    module_id: str, width: int, height: int
) -> list[list[tuple[float, float]]]:
    """Fit the same geographic H3 boundaries into a static canvas."""
    rings = [ring for _, ring in _cells(module_id)]
    xs = [lng for ring in rings for lat, lng in ring]
    ys = [lat for ring in rings for lat, lng in ring]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    scale = min(width * 0.42 / (x1 - x0), height * 0.56 / (y1 - y0))
    return [
        [
            (
                width * 0.72 + (lng - (x0 + x1) / 2) * scale,
                height * 0.54 - (lat - (y0 + y1) / 2) * scale,
            )
            for lat, lng in ring
        ]
        for ring in rings
    ]


def render_svg_card(
    module_id: str,
    output_path: Path | None = None,
    *,
    width: int = 700,
    height: int = 380,
) -> str:
    """Render actual H3 geometry with an explicit illustrative-data label."""
    profile = _profile(module_id)
    mod = module_id.upper()
    _dimensions(width, height)
    rings = _projected_rings(mod, width, height)
    polygons = []
    for (cell, _), ring in zip(_cells(mod), rings):
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in ring)
        polygons.append(
            f'<polygon data-h3="{cell}" points="{points}" fill="{profile["secondary_color"]}" fill-opacity="0.45" stroke="{profile["primary_color"]}"><title>{cell}</title></polygon>'
        )
    name = html_lib.escape(profile["name"])
    category = html_lib.escape(profile["category"])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title-{mod} desc-{mod}" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<title id="title-{mod}">GEO-INFER-{mod}: {name}</title>
<desc id="desc-{mod}">{ILLUSTRATION_LABEL}</desc>
<rect width="100%" height="100%" rx="12" fill="#f8fafc"/>
<text x="24" y="36" font-family="sans-serif" font-size="16" fill="#0f172a">GEO-INFER-{mod}</text>
<text x="24" y="66" font-family="sans-serif" font-size="14" fill="#334155">{name}</text>
<text x="24" y="92" font-family="sans-serif" font-size="12" fill="#475569">{category}</text>
{"".join(polygons)}
<circle cx="{width * 0.72}" cy="{height * 0.54}" r="3" fill="{profile["primary_color"]}"/>
<text x="24" y="{height - 38}" font-family="sans-serif" font-size="11" fill="#334155">Illustrative overview. H3 resolution {RESOLUTION}; no measured values.</text>
<text x="24" y="{height - 18}" font-family="sans-serif" font-size="10" fill="#475569">WGS84 · CC BY-NC-SA 4.0</text>
</svg>'''
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg, encoding="utf-8")
    return svg


def render_leaflet_html(
    module_id: str,
    output_path: Path | None = None,
    *,
    width: int = 800,
    height: int = 450,
) -> str:
    """Render deterministic Leaflet HTML with an always-available static preview."""
    profile = _profile(module_id)
    mod = module_id.upper()
    _dimensions(width, height)
    payload = {
        "center": profile["center"],
        "zoom": profile["zoom"],
        "cells": _cells(mod),
        "color": profile["primary_color"],
        "label": ILLUSTRATION_LABEL,
    }
    # Escape HTML delimiters even inside JSON so profile content cannot close a script.
    encoded = (
        json.dumps(payload, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    name = html_lib.escape(profile["name"])
    svg = render_svg_card(mod, width=width, height=height)
    markup = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEO-INFER-{mod}: {name}</title>
<link rel="stylesheet" href="{LEAFLET_CSS}">
<style>body{{margin:1rem;font-family:system-ui,sans-serif;color:#0f172a;background:#f8fafc}}main{{max-width:{width}px;margin:auto}}#map{{height:{height}px;display:none}}svg{{max-width:100%;height:auto}}summary{{cursor:pointer}}p{{line-height:1.5}}</style></head>
<body><main><h1>GEO-INFER-{mod}: {name}</h1><p>{ILLUSTRATION_LABEL}</p>
<div id="map" aria-label="Interactive H3 geometry"></div>
<details id="static-preview" open><summary>Static preview (available offline)</summary>{svg}</details>
<script id="preview-data" type="application/json">{encoded}</script>
<script src="{LEAFLET_JS}"></script>
<script>
if (typeof L !== 'undefined') {{
 const data = JSON.parse(document.getElementById('preview-data').textContent);
 document.getElementById('map').style.display = 'block';
 const map = L.map('map').setView(data.center, data.zoom);
 L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{attribution:'© OpenStreetMap contributors'}}).addTo(map);
 const group = L.featureGroup().addTo(map);
 for (const [cell, ring] of data.cells) {{
  const popup = document.createElement('span'); popup.textContent = cell + ': ' + data.label;
  L.polygon(ring, {{color:data.color, fillOpacity:0.35}}).bindPopup(popup).addTo(group);
 }}
 map.fitBounds(group.getBounds());
}}
</script></main></body></html>'''
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markup, encoding="utf-8")
    return markup


def render_png_card(
    module_id: str,
    output_path: Path | None = None,
    *,
    width: int = 400,
    height: int = 240,
) -> bytes:
    """Draw the geographic preview with Pillow and embed its data provenance."""

    profile = _profile(module_id)
    mod = module_id.upper()
    _dimensions(width, height)
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    for ring in _projected_rings(mod, width, height):
        draw.polygon(
            ring,
            fill=profile["secondary_color"],
            outline=profile["primary_color"],
            width=2,
        )
    draw.text((12, 12), f"GEO-INFER-{mod}", fill="#0f172a")
    draw.text(
        (12, height - 34), "Illustrative overview; no measured values.", fill="#334155"
    )
    draw.text((12, height - 18), f"H3 resolution {RESOLUTION} / WGS84", fill="#475569")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Title", f"GEO-INFER-{mod} Preview")
    metadata.add_text("Provenance", ILLUSTRATION_LABEL)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", pnginfo=metadata)
    data = buffer.getvalue()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return data


def generate_module_preview_suite(
    module_id: str, output_dir: Path | str
) -> SpatialPreviewArtifacts:
    """Write reproducible geometry previews with per-artifact checksums."""
    profile = _profile(module_id)
    mod = module_id.upper()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = f"geo-infer-{mod.lower()}_preview"
    html_file, svg_file, png_file = [
        out / f"{slug}.{suffix}" for suffix in ("html", "svg", "png")
    ]
    html_text = render_leaflet_html(mod, html_file)
    svg_text = render_svg_card(mod, svg_file)
    png_data = render_png_card(mod, png_file)
    provenance = {
        "data_kind": "illustrative",
        "geometry_source": "h3.cell_to_boundary",
        "h3_version": h3.__version__,
        "resolution": RESOLUTION,
        "crs": "EPSG:4326",
        "label": ILLUSTRATION_LABEL,
    }
    digest = hashlib.sha256(
        json.dumps(
            {"profile": profile, "cells": _cells(mod), "provenance": provenance},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "geo-infer-intra-visual-preview/v2",
        "module_id": f"GEO-INFER-{mod}",
        "name": profile["name"],
        "category": profile["category"],
        "input_sha256": digest,
        "provenance": provenance,
        "external_resources": [
            LEAFLET_CSS,
            LEAFLET_JS,
            "https://tile.openstreetmap.org/",
        ],
        "artifacts": [],
        "accessibility": {
            "has_title": True,
            "has_svg_viewbox": True,
            "has_png_metadata": True,
            "has_static_fallback": True,
        },
    }
    for path, mime in (
        (html_file, "text/html"),
        (svg_file, "image/svg+xml"),
        (png_file, "image/png"),
    ):
        data = path.read_bytes()
        manifest["artifacts"].append(
            {
                "name": path.name,
                "type": mime,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    receipt = out / f"{slug}.manifest.json"
    receipt.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SpatialPreviewArtifacts(
        mod,
        html_file,
        svg_file,
        png_file,
        receipt,
        digest,
        len(html_text.encode()),
        len(svg_text.encode()),
        len(png_data),
    )


def generate_all_module_previews(
    output_dir: Path | str,
) -> dict[str, SpatialPreviewArtifacts]:
    """Generate all module bundles in stable registry order."""
    return {
        mod: generate_module_preview_suite(mod, output_dir)
        for mod in sorted(MODULE_PROFILES)
    }


__all__ = [
    "MODULE_PROFILES",
    "SpatialPreviewArtifacts",
    "render_leaflet_html",
    "render_svg_card",
    "render_png_card",
    "generate_module_preview_suite",
    "generate_all_module_previews",
]


# Anchor used when a module page references its interactive spatial preview.
_PREVIEW_SECTION_TITLE = "## 🗺️ Interactive Spatial Preview"


def _render_preview_markdown(module_id: str, slug: str, prefix: str) -> str:
    """Render a Markdown snippet that anchors a module to its preview bundle."""
    friendly_name = MODULE_PROFILES[module_id]["name"]
    return "\n".join(
        [
            _PREVIEW_SECTION_TITLE,
            "",
            f"Pre-rendered spatial snapshot for **GEO-INFER-{module_id}** "
            f"(*{friendly_name}*). Reproducible preview cards are generated by "
            "`geo_infer_intra.core.documentation.visual_preview`.",
            "",
            "| Preview | Widget |",
            "| --- | --- |",
            f"| ![GEO-INFER-{module_id} Leaflet Preview]({prefix}/{slug}_preview.svg) | "
            f"[Interactive map]({prefix}/{slug}_preview.html) · [PNG]({prefix}/{slug}_preview.png) |",
            "",
            "> **Reproducible contract:** each map ships as "
            f"`{slug}_preview.html`, `{slug}_preview.svg`, `{slug}_preview.png`, "
            f"and `{slug}_preview.manifest.json` beneath `previews/`. The receipt "
            "records geometry provenance and artifact SHA-256 hashes. Values are illustrative, not observations.",
            "",
        ]
    )


def _inject_preview_section(module_doc: Path, module_id: str, prefix: str) -> bool:
    """Insert (or refresh) the interactive spatial preview section into a page."""
    slug = f"geo-infer-{module_id.lower()}"
    text = module_doc.read_text(encoding="utf-8")
    section = _render_preview_markdown(module_id, slug, prefix)

    if _PREVIEW_SECTION_TITLE in text:
        # Replace the existing preview section (from title to next heading).
        lines = text.splitlines()
        start = next(
            i for i, line in enumerate(lines) if line.strip() == _PREVIEW_SECTION_TITLE
        )
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if (
                lines[j].startswith("## ")
                and lines[j].strip() != _PREVIEW_SECTION_TITLE
            ):
                end = j
                break
        new_lines = lines[:start] + section.splitlines() + lines[end:]
        module_doc.write_text("\n".join(new_lines), encoding="utf-8")
        return True

    module_doc.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
    return True


def build_previews(modules_dir: Path, output_dir: Path) -> int:
    """Generate preview bundles for all 45 modules and write a preview index."""
    modules_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(os.path.relpath(output_dir, modules_dir)).as_posix()

    emitted = 0
    index_lines = [
        "# Spatial Preview Cards",
        "",
        "Auto-generated single and static preview snapshots for all GEO-INFER "
        "domain modules, produced by "
        "`geo_infer_intra.core.documentation.visual_preview`.",
        "",
        "| Module | Preview |",
        "| --- | --- |",
    ]

    for module_id in sorted(MODULE_PROFILES.keys()):
        slug = f"geo-infer-{module_id.lower()}"
        generate_module_preview_suite(module_id, output_dir)
        emitted += 1

        module_doc = modules_dir / f"geo-infer-{module_id.lower()}.md"
        if module_doc.is_file():
            _inject_preview_section(module_doc, module_id, prefix)

        index_lines.append(
            f"| {module_id} | [Interactive]({prefix}/{slug}_preview.html) · [Static SVG]({prefix}/{slug}_preview.svg) |"
        )

    index_lines.append("")
    (modules_dir / "previews_index.md").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )
    return emitted
