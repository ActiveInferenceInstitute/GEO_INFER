"""
CrescentCityIntelMapper: map the Crescent City civic-intel contract onto Del Norte H3.

Imports the machine-readable crescent-city-intel geospatial contract
(``pages-data/geo-intel.json`` - schema ``crescent-city-geo-intel/v1``) produced
by the ``docxology/crescent-city-intel`` platform, and maps its civic
intelligence domains + hazard-relevant subset onto the municipality H3 grid so
the comprehensive dashboard can weight municipal-code policy by hazard intent
(tsunami, seismic, flood, erosion) across the same spatial canvas as the
forest / coastal / fire / seismic analyzers.

The mapping base is generalized through ``MunicipalGeoIntelMapper``: bounds and
hazard domains are read from the contract's own ``anchor`` / ``hazard`` blocks
rather than hardcoded Del Norte constants, so ANY conforming municipality
contract drives the same mapping code.  ``CrescentCityIntelMapper`` is the
packaged default (Crescent City / Del Norte County).

Data source:
    The contract is read from ``CRESCENT_INTEL_GEO_JSON`` (env override) or a
    packaged seed ``data/crescent-city-geo-intel.json`` bundled beside this
    module. The packaged seed is a reviewed copy of the platform's committed
    ``pages-data/geo-intel.json``; it keeps this mapper dependency-free (no live
    output/ or sibling-repo path needed) and gracefully reports when absent.
"""

import importlib.resources
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import h3

logger = logging.getLogger(__name__)

# Default packaged seed path bundled with the module (resolved from the
# installed package, not a repo-relative path).
_SEED_REL = Path(
    str(
        importlib.resources.files(
            "geo_infer_place.locations.del_norte_county"
        ).joinpath("data/crescent-city-geo-intel.json")
    )
)

# Hazard-class orientation sets used by the geometry-derived coverage scoring.
_SEISMIC_TAGS = frozenset({"tsunami", "seismic"})
_COASTAL_TAGS = frozenset({"erosion", "flood", "flood zone", "inundation"})

# A civic policy "applies" in a cell when its weight clears this threshold.
_COVERAGE_THRESHOLD = 0.30

_EARTH_KM = 6371.0

CoastalEdge = Literal["west", "east", "south", "north"]
_COASTAL_EDGES: tuple[str, ...] = ("west", "east", "south", "north")

# Contract ``anchor.coastalEdge`` values that explicitly mark a landlocked
# municipality, i.e. no bounds edge is a shoreline.
_INLAND_EDGE_TOKENS = frozenset({"none", "inland", "landlocked", "interior"})


def _clamp01(value: float) -> float:
    """Clamp a value into the unit interval [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two (lat, lon) points (haversine)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (
        math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * _EARTH_KM * math.asin(math.sqrt(a))


@dataclass
class MunicipalGeoIntelMapper:
    """Generic civic-intelligence mapper driven by a ``crescent-city-geo-intel/v1``
    contract.

    The base class derives everything spatial from the contract document itself,
    so it transfers to any municipality that ships the same schema. Coast
    orientation is coastline-agnostic: the effective coastal edge is resolved
    from the contract (``anchor.coastalEdge``), overridable via the
    ``coastal_edge`` constructor argument, with fallback handling:

    - ``anchor.bounds`` drives the H3 grid extent (``bounds()``)
    - ``hazard.relevantDomains`` is the hazard-policy subset (``hazard_domains()``)
    - ``anchor.latitude/longitude`` is the municipal seat for proximity scoring
    - ``anchor.coastalEdge`` (``"west"/"east"/"south"/"north"``) orients hazard
      weighting toward whichever bounds edge is the shoreline; an inland marker
      (``"none"/"inland"/"landlocked"/"interior"``) disables coastal weighting.
      When a contract neither declares an edge nor is explicitly landlocked,
      ``_default_coastal_edge`` (``"west"``) preserves the implicit Crescent
      City west coast.

    Each cell in the grid receives a ``hazard_density`` in [0, 1] and a
    ``domain_coverage`` (count of civic hazard domains whose policy weight
    clears ``_COVERAGE_THRESHOLD``), both derived deterministically from the
    contract geometry. ``CrescentCityIntelMapper`` supplies the packaged default
    seed plus a conservative fallback extent for the known site.
    """

    seed_path: Path | None = None
    h3_resolution: int = 8
    # None = auto-detect: resolve from the contract's ``anchor.coastalEdge``,
    # else fall back to ``_default_coastal_edge``. An explicit value overrides
    # any contract declaration.
    coastal_edge: CoastalEdge | None = None
    # Class-level fallback orientation for a contract that declares no coast; a
    # subclass may override (kept "west" for Crescent City / Del Norte).
    _default_coastal_edge: ClassVar[CoastalEdge] = "west"

    # Municipality-specific fallback extent used only when the contract omits
    # ``anchor.bounds`` so a known site keeps working offline.
    _fallback_bounds: dict[str, float] = field(default_factory=dict)

    # Seed resolution hooks (env override name + packaged default path).
    _seed_env: str = "CRESCENT_INTEL_GEO_JSON"
    _packaged_seed: Path = _SEED_REL

    # ---- Public state ----
    contract: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    source_path: Path | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Resolve the seed path (env override -> passed -> packaged) and load."""
        if self.coastal_edge is not None and self.coastal_edge not in _COASTAL_EDGES:
            allowed = ", ".join(_COASTAL_EDGES)
            raise ValueError(
                f"coastal_edge must be one of {allowed} or None "
                f"(auto-detect from contract); got {self.coastal_edge!r}"
            )
        self._seed_override: Path | None = None
        env = os.environ.get(self._seed_env)
        if env and Path(env).exists():
            self._seed_override = Path(env)
        self._load_contract()

    # -- Loading --

    def _default_seed_path(self) -> Path:
        if self._seed_override is not None:
            return self._seed_override
        return self._packaged_seed

    def _load_contract(self) -> None:
        """Load the contract JSON with graceful absence reporting."""
        raw_path = self.seed_path or self._default_seed_path()
        path = Path(raw_path)
        if not path.exists():
            self.loaded = False
            self.error = f"Intel contract not found at {path}"
            logger.warning(self.error)
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self.contract = cast(dict[str, Any], json.load(fh))
            if self.contract.get("schema") != "crescent-city-geo-intel/v1":
                raise ValueError(
                    f"Unexpected intel schema {self.contract.get('schema')!r}"
                )
            self.loaded = True
            self.source_path = path
            logger.info(
                "Crescent City intel contract loaded: %s domains (schema v1)",
                self.contract.get("domainCount"),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.loaded = False
            self.contract = {}
            self.error = f"Failed to load Crescent City intel contract: {exc}"
            logger.warning(self.error)

    # -- Mapper surface --

    def municipality_name(self) -> str:
        """Human municipality label from the contract anchor ('' if unset)."""
        if not self.loaded:
            return ""
        return str(self.contract.get("anchor", {}).get("name", "") or "")

    def anchor(self) -> dict[str, Any]:
        """Return the contract anchor dict (empty when not loaded)."""
        if not self.loaded:
            return {}
        anchor = self.contract.get("anchor", {}) or {}
        return cast(dict[str, Any], anchor)

    def domains(self) -> list[dict[str, Any]]:
        """Return the civic-intel domains (empty list when not loaded)."""
        if not self.loaded:
            return []
        return self.contract.get("domains", []) or []

    def hazard_domains(self) -> list[dict[str, Any]]:
        """Return the hazard-relevant domain subset."""
        if not self.loaded:
            return []
        return self.contract.get("hazard", {}).get("relevantDomains", []) or []

    def domain_ids(self) -> list[str]:
        """Return sorted civic-domain slugs (useful for gap reporting)."""
        return sorted(d.get("id", "") for d in self.domains() if d.get("id"))

    def bounds(self) -> dict[str, float]:
        """Return the grid bounds: contract ``anchor.bounds`` or fallback.

        When neither the contract nor the subclass fallback supplies geometry,
        an empty dict is returned and the mapper reports "no_data".
        """
        if not self.loaded and not self._fallback_bounds:
            return {}
        anchor = self.contract.get("anchor", {}) if self.loaded else {}
        bounds = anchor.get("bounds", {}) or {}
        fb = self._fallback_bounds
        west = float(bounds.get("west", fb.get("west", 0.0)))
        south = float(bounds.get("south", fb.get("south", 0.0)))
        east = float(bounds.get("east", fb.get("east", 0.0)))
        north = float(bounds.get("north", fb.get("north", 0.0)))
        return {"west": west, "south": south, "east": east, "north": north}

    def _grid_geojson(self) -> dict[str, Any]:
        """Return a GeoJSON Polygon (lng,lat ring) covering the grid bounds."""
        b = self.bounds()
        if not b:
            return {"type": "Polygon", "coordinates": [[]]}
        ring = [
            [b["west"], b["south"]],
            [b["east"], b["south"]],
            [b["east"], b["north"]],
            [b["west"], b["north"]],
            [b["west"], b["south"]],  # close ring
        ]
        return {"type": "Polygon", "coordinates": [ring]}

    # -- Spatial hazard scoring (geometry-derived, transferable) --

    def _cell_latlng(self, cell_id: str) -> tuple[float, float]:
        """Return (lat, lon) at the H3 cell center."""
        lat, lng = h3.cell_to_latlng(cell_id)
        return float(lat), float(lng)

    def _resolve_coastal_edge(self) -> CoastalEdge | None:
        """Resolve the effective coastal bounds edge for hazard weighting.

        Resolution priority (coastline-agnostic, no western-shoreline
        assumption):

        1. A caller-supplied ``coastal_edge`` argument wins outright.
        2. ``anchor.coastalEdge`` from the contract: a valid bounds edge
           (``"west"/"east"/"south"/"north"``) orients the coast there; an
           inland marker (``"none"/"inland"/"landlocked"/"interior"``) marks a
           landlocked municipality and returns ``None`` (no coast).
        3. Otherwise fall back to ``_default_coastal_edge`` (``"west"``), which
           preserves Crescent City's implicit west coast when a contract
           declares no orientation — so Del Norte results are unchanged.

        Returns ``None`` only for an explicitly landlocked contract, meaning no
        bounds edge is treated as a shoreline.
        """
        if self.coastal_edge is not None:
            return self.coastal_edge
        if self.loaded:
            anchor = self.contract.get("anchor", {}) or {}
            raw = anchor.get("coastalEdge")
            if raw is not None:
                value = str(raw).strip().lower()
                if value in _INLAND_EDGE_TOKENS:
                    return None
                if value in _COASTAL_EDGES:
                    return cast(CoastalEdge, value)
                logger.warning("Ignoring unknown contract coastalEdge %r", raw)
        return self._default_coastal_edge

    def _coast_proximity(self, lat: float, lng: float, b: dict[str, float]) -> float:
        """Normalized proximity to the effective coastal edge in [0, 1].

        Orients against whichever bounds edge the contract (or the caller)
        declares as the coast, so eastern-coast and north/south shorelines
        weight correctly instead of assuming a western coast. Returns 0.0 for a
        landlocked contract (no coastal edge).
        """
        edge = self._resolve_coastal_edge()
        if edge is None:
            return 0.0
        if edge in {"west", "east"}:
            span = float(b["east"] - b["west"])
            if span <= 0.0:
                return 0.0
            offset = (
                float(b["east"]) - lng if edge == "west" else lng - float(b["west"])
            )
        else:
            span = float(b["north"] - b["south"])
            if span <= 0.0:
                return 0.0
            offset = (
                float(b["north"]) - lat if edge == "south" else lat - float(b["south"])
            )
        return _clamp01(offset / span)

    def _seat_proximity(self, lat: float, lng: float) -> float:
        """Distance-decay closeness to the municipal seat-anchored in [0,1]."""
        anchor = self.anchor()
        if not anchor.get("latitude") or not anchor.get("longitude"):
            return 0.0
        b = self.bounds()
        if not b:
            return 0.0
        a_lat = float(anchor["latitude"])
        a_lng = float(anchor["longitude"])
        lat_span_km = abs(b["north"] - b["south"]) * 111.0
        lon_span_km = (
            abs(b["east"] - b["west"]) * 111.0 * max(math.cos(math.radians(a_lat)), 0.2)
        )
        ref_km = max(lat_span_km, lon_span_km, 1.0)
        dist_km = _distance_km(a_lat, a_lng, lat, lng)
        return _clamp01(1.0 - (dist_km / ref_km))

    def _domain_weight(self, domain: dict[str, Any], lat: float, lng: float) -> float:
        """Policy-coverage weight in [0,1] for a hazard domain at (lat, lng).

        Geometry-derived (coast proximity + municipal-seat proximity) and tuned
        by the domain's hazard tags — transferable to any municipality, not a
        fabricated observation.
        """
        b = self.bounds()
        if not b:
            return 0.0
        seat = self._seat_proximity(lat, lng)
        if not self.anchor():
            return 0.0
        coast = self._coast_proximity(lat, lng, b)
        tags = set(str(t).lower() for t in (domain.get("hazardTags") or []))
        if tags & _SEISMIC_TAGS:
            # Tsunami / seismic: severe near both seat and the coast edge.
            return _clamp01(0.45 * coast + 0.55 * seat)
        if tags & _COASTAL_TAGS:
            # Erosion / flood / inundation: dominated by the coastal edge.
            return _clamp01(0.30 * seat + 0.70 * coast)
        return _clamp01(seat)

    def _all_hazard_tags(self) -> set[str]:
        """Collect the union of hazard tags across all hazard domains."""
        tags: set[str] = set()
        for dom in self.hazard_domains():
            for tag in dom.get("hazardTags", []):
                tags.add(str(tag))
        return tags

    def generate_h3_cells(self) -> dict[str, dict[str, Any]]:
        """Generate an H3-indexed civic-intel surface covering the bounds.

        Returns:
            Mapping of ``h3_cell -> {hazard_density, hazard_tags,
            domain_coverage, coverage_by_domain, domain_count,
            civic_domains, intel_present}`` for every grid cell intersecting
            the bounds polygon. Empty when the contract lacks geometry.
        """
        if not self.loaded:
            return {}
        geojson = self._grid_geojson()
        if not geojson.get("coordinates"):
            return {}
        try:
            cells = set(h3.geo_to_cells(geojson, self.h3_resolution))
        except TypeError:
            # Older h3 (<4.5) used the vertex-list form; fall back gracefully.
            cells = set()
            b = self.bounds()
            if b:
                for lat, lng in [
                    (b["south"], b["west"]),
                    (b["south"], b["east"]),
                    (b["north"], b["east"]),
                    (b["north"], b["west"]),
                ]:
                    cells.add(h3.latlng_to_cell(lat, lng, self.h3_resolution))

        hazard_domains = self.hazard_domains()
        domain_ids = self.domain_ids()
        by_id: dict[str, dict[str, Any]] = {
            str(d.get("id")): d for d in hazard_domains if d.get("id")
        }
        out: dict[str, dict[str, Any]] = {}
        for cell_id in cells:
            lat, lng = self._cell_latlng(cell_id)
            weights: dict[str, float] = {}
            for did, dom in by_id.items():
                weights[did] = round(float(self._domain_weight(dom, lat, lng)), 3)
            applying_ids = [
                did for did, w in weights.items() if w >= _COVERAGE_THRESHOLD
            ]
            density = max(weights.values()) if weights else 0.0
            tags: list[str] = []
            for did in applying_ids:
                tags.extend(str(t) for t in (by_id[did].get("hazardTags") or []))
            out[cell_id] = {
                "hazard_density": round(float(density), 3),
                "hazard_tags": sorted(set(tags)),
                "domain_coverage": len(applying_ids),
                "coverage_by_domain": dict(weights),
                "domain_count": len(domain_ids),
                "intel_present": True,
                "civic_domains": list(domain_ids)[:25],
            }
        return out

    def generate_hazard_surface(self) -> dict[str, Any]:
        """Build a compact hazard-intent summary (domain-aware, not a full grid).

        Each hazard domain is enriched with a ``coverage`` (mean weight across
        the grid) so a dashboard panel ranks the top policy hazard domains.
        Returns ``{"status": "no_hazard_domains", "domains": []}`` when absent.
        """
        hazards = self.hazard_domains()
        if not hazards:
            return {"status": "no_hazard_domains", "domains": []}
        cell_surface = self.generate_h3_cells()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for cell_data in cell_surface.values():
            by_domain = cell_data.get("coverage_by_domain") or {}
            for did, weight in by_domain.items():
                sums[did] = sums.get(did, 0.0) + float(weight)
                counts[did] = counts.get(did, 0) + 1
        domains_out: list[dict[str, Any]] = []
        for d in hazards:
            did = d.get("id")
            if did is None:
                continue
            avg = (sums.get(did, 0.0) / counts[did]) if counts.get(did) else 0.0
            entry: dict[str, Any] = {
                "id": d.get("id"),
                "name": d.get("name"),
                "icon": d.get("icon"),
                "hazardTags": d.get("hazardTags", []),
                "topics": [
                    {"name": t.get("name"), "sections": t.get("sections", [])}
                    for t in d.get("topics", [])
                ],
                "coverage": round(float(avg), 3),
            }
            domains_out.append(entry)
        return {"status": "ok", "domains": domains_out}


@dataclass
class CrescentCityIntelMapper(MunicipalGeoIntelMapper):
    """Crescent City pairing: the packaged municipal contract + Del Norte fallback.

    Default subclass of ``MunicipalGeoIntelMapper``.  Bundles the reviewed
    ``crescent-city-geo-intel.json`` seed and, should the contract or its
    geometry be absent, delivers a conservative Del Norte County extent so the
    dashboard import path stays deterministic offline.
    """

    _fallback_bounds: dict[str, float] = field(
        default_factory=lambda: {
            "west": -124.408,
            "south": 41.458,
            "east": -123.536,
            "north": 42.006,
        }
    )
