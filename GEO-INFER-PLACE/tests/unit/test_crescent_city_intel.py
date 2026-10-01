#!/usr/bin/env python3
"""
Test suite for CrescentCityIntelMapper — the civic-intel bridge from the
crescent-city-intel platform's machine-readable contract (schema
``crescent-city-geo-intel/v1``) onto the Del Norte H3 canvas.

Covers:
- Contract loading (packaged seed, missing-path degradation)
- 12 civic-domain surface + hazard-relevant subset
- H3 cell generation across the Del Norte bounds
- Hazard-intent surface shaping

All tests use the real mapper + the packaged (committed) seed; no mocks.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import h3

_PACKAGED_SEED = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "geo_infer_place"
    / "locations"
    / "del_norte_county"
    / "data"
    / "crescent-city-geo-intel.json"
)


def _seed_json() -> dict[str, Any]:
    """Load the real packaged seed (the test fixture)."""
    with open(_PACKAGED_SEED, encoding="utf-8") as fh:
        return json.load(fh)


def _mapper(seed_path: Path) -> Any:
    from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
        CrescentCityIntelMapper,
    )

    return CrescentCityIntelMapper(seed_path=seed_path)


def _write_contract(
    tmpdir: str,
    *,
    coastal_edge: str | None,
    bounds: dict[str, float],
    hazard_tags: list | None = None,
) -> Path:
    """Write a minimal ``crescent-city-geo-intel/v1`` contract to ``tmpdir``.

    The synthetic municipal contract is real JSON (read back by the mapper via
    the same code path as the packaged seed); ``coastal_edge`` sets the optional
    ``anchor.coastalEdge`` declaration (None omits it entirely).
    """
    tags = list(hazard_tags) if hazard_tags is not None else ["erosion", "flood zone"]
    anchor: dict[str, Any] = {
        "name": "Eastport",
        "municipality": "Eastport",
        "county": "Test County",
        "state": "Test State",
        "latitude": (bounds["south"] + bounds["north"]) / 2.0,
        "longitude": (bounds["west"] + bounds["east"]) / 2.0,
        "bounds": bounds,
    }
    if coastal_edge is not None:
        anchor["coastalEdge"] = coastal_edge
    contract: dict[str, Any] = {
        "schema": "crescent-city-geo-intel/v1",
        "anchor": anchor,
        "domainCount": 1,
        "domains": [
            {
                "id": "hazard-policy",
                "name": "Hazard Policy",
                "icon": "🌊",
                "hazardTags": list(tags),
            }
        ],
        "hazard": {
            "relevantDomains": [
                {
                    "id": "hazard-policy",
                    "name": "Hazard Policy",
                    "icon": "🌊",
                    "hazardTags": list(tags),
                    "topics": [],
                }
            ],
            "relevantDomainCount": 1,
        },
    }
    seed_path = Path(tmpdir) / "eastport-geo-intel.json"
    seed_path.write_text(json.dumps(contract), encoding="utf-8")
    return seed_path


class TestContractLoad(unittest.TestCase):
    """Contract loading from the packaged seed + degradation paths."""

    def test_contract_has_v1_schema_and_12_domains(self) -> None:
        seed = _seed_json()
        assert seed["schema"] == "crescent-city-geo-intel/v1"
        assert seed["domainCount"] == 12

    def test_mapper_loads_packaged_seed(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        assert mapper.loaded is True
        assert len(mapper.domains()) == 12
        assert mapper.source_path == _PACKAGED_SEED

    def test_mapper_graceful_when_seed_missing(self) -> None:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="test_intel_"))
        seed_path = tmp / "missing-geo-intel.json"
        mapper = _mapper(seed_path)
        assert mapper.loaded is False
        assert mapper.error is not None
        assert mapper.domains() == []
        assert mapper.hazard_domains() == []

    def test_mapper_rejects_wrong_schema(self) -> None:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="test_intel_bad_"))
        seed_path = tmp / "bad.json"
        seed_path.write_text(json.dumps({"schema": "wrong/v1", "domains": []}))
        mapper = _mapper(seed_path)
        assert mapper.loaded is False


class TestContactPublicSurface(unittest.TestCase):
    """Civic-domain + hazard surface projection."""

    def test_domain_ids_have_unique_slugs(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        ids = mapper.domain_ids()
        assert len(ids) == 12
        assert len(set(ids)) == 12

    def test_hazard_domains_flagged(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        hazard = mapper.hazard_domains()
        assert len(hazard) >= 1
        # Emergency-management is the flagship tsunami/preparedness domain.
        assert "emergency-management" in [d["id"] for d in hazard]

    def test_bounds_are_valid_del_norte_extent(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        b = mapper.bounds()
        assert b["west"] < b["east"] < -123
        assert b["south"] < b["north"]

    def test_generate_h3_cells_produces_grid(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        cells = mapper.generate_h3_cells()
        assert len(cells) > 0
        first = next(iter(cells.values()))
        assert first["intel_present"] is True
        assert isinstance(first["hazard_tags"], list)

    def test_cells_carry_density_and_coverage(self) -> None:
        """Per-cell hazard density + domain-coverage counts are computed."""
        mapper = _mapper(_PACKAGED_SEED)
        cells = mapper.generate_h3_cells()
        for data in cells.values():
            assert 0.0 <= data["hazard_density"] <= 1.0
            assert isinstance(data["domain_coverage"], int)
            assert isinstance(data["coverage_by_domain"], dict)
            assert data["domain_count"] == 12
        densities = {data["hazard_density"] for data in cells.values()}
        assert len(densities) > 1, "expected spatially varying hazard density"

    def test_coast_proximity_decreases_from_west_to_east(self) -> None:
        """The western coast scores one and the inland edge scores zero."""
        mapper = _mapper(_PACKAGED_SEED)
        bounds = mapper.bounds()
        midpoint_lat = (bounds["south"] + bounds["north"]) / 2.0
        midpoint_lng = (bounds["west"] + bounds["east"]) / 2.0

        assert mapper._coast_proximity(midpoint_lat, bounds["west"], bounds) == 1.0
        self.assertAlmostEqual(
            mapper._coast_proximity(midpoint_lat, midpoint_lng, bounds), 0.5
        )
        assert mapper._coast_proximity(midpoint_lat, bounds["east"], bounds) == 0.0

    def test_coast_proximity_supports_every_bounds_edge(self) -> None:
        """Generic municipalities can orient the coast on any bounds edge."""
        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        west = MunicipalGeoIntelMapper(seed_path=_PACKAGED_SEED)
        bounds = west.bounds()
        midpoint_lat = (bounds["south"] + bounds["north"]) / 2.0
        midpoint_lng = (bounds["west"] + bounds["east"]) / 2.0

        east = MunicipalGeoIntelMapper(seed_path=_PACKAGED_SEED, coastal_edge="east")
        assert east._coast_proximity(midpoint_lat, bounds["east"], bounds) == 1.0
        assert east._coast_proximity(midpoint_lat, bounds["west"], bounds) == 0.0

        south = MunicipalGeoIntelMapper(seed_path=_PACKAGED_SEED, coastal_edge="south")
        assert south._coast_proximity(bounds["south"], midpoint_lng, bounds) == 1.0
        assert south._coast_proximity(bounds["north"], midpoint_lng, bounds) == 0.0

        north = MunicipalGeoIntelMapper(seed_path=_PACKAGED_SEED, coastal_edge="north")
        assert north._coast_proximity(bounds["north"], midpoint_lng, bounds) == 1.0
        assert north._coast_proximity(bounds["south"], midpoint_lng, bounds) == 0.0

    def test_invalid_coastal_edge_fails_closed(self) -> None:
        """Invalid orientation is rejected instead of silently misweighting cells."""
        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        with self.assertRaisesRegex(ValueError, "coastal_edge"):
            MunicipalGeoIntelMapper(
                seed_path=_PACKAGED_SEED,
                coastal_edge="center",  # type: ignore[arg-type]
            )

    def test_generic_mapper_transferable_from_contract(self) -> None:
        """MunicipalGeoIntelMapper drives the same mapping from the contract alone."""
        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        mapper = MunicipalGeoIntelMapper(seed_path=_PACKAGED_SEED, h3_resolution=8)
        assert mapper.loaded is True
        cells = mapper.generate_h3_cells()
        assert len(cells) > 0
        # Same geometry as the Crescent City default (same contract extent).
        default = _mapper(_PACKAGED_SEED)
        assert default.bounds() == mapper.bounds()

    def test_hazard_surface_ranks_domains(self) -> None:
        """Each hazard domain carries a mean grid-coverage score."""
        mapper = _mapper(_PACKAGED_SEED)
        surface = mapper.generate_hazard_surface()
        for domain in surface["domains"]:
            assert "coverage" in domain
            assert 0.0 <= domain["coverage"] <= 1.0

    def test_hazard_surface_shapes_domains(self) -> None:
        mapper = _mapper(_PACKAGED_SEED)
        surface = mapper.generate_hazard_surface()
        assert surface["status"] == "ok"
        assert len(surface["domains"]) >= 1
        for domain in surface["domains"]:
            assert "hazardTags" in domain
            assert "topics" in domain


class TestCoastlineAgnosticOrientation(unittest.TestCase):
    """The transferable mapper orients to the contract's coastline — no implicit
    western-shoreline assumption (the fleet residual)."""

    # Synthetic eastern-coast municipality geometry: ocean on the EAST edge,
    # landward flank on the west (lng/lat span ~ Del Norte scale).
    _EAST_BOUNDS: dict[str, float] = {
        "west": -80.30,
        "south": 25.70,
        "east": -80.10,
        "north": 25.90,
    }

    @staticmethod
    def _mid_lng(bounds: dict[str, float]) -> float:
        return (bounds["west"] + bounds["east"]) / 2.0

    def test_del_norte_default_still_orients_west_without_declaration(self) -> None:
        """A contract that omits coastalEdge keeps the implicit west coast, so
        Del Norte behavior is unchanged."""
        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        mapper = _mapper(_PACKAGED_SEED)
        assert mapper._resolve_coastal_edge() == "west"
        # Auto-detection must match an explicit west orientation cell-for-cell.
        explicit = MunicipalGeoIntelMapper(
            seed_path=_PACKAGED_SEED, coastal_edge="west"
        )
        assert mapper.generate_h3_cells() == explicit.generate_h3_cells()

    def test_contract_declaring_east_self_orients_eastward(self) -> None:
        """Eastern-coast municipal contract drives the coastal edge to the east
        with no caller-supplied orientation."""
        import tempfile

        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        with tempfile.TemporaryDirectory(prefix="intel_east_") as td:
            seed_path = _write_contract(
                td, coastal_edge="east", bounds=self._EAST_BOUNDS
            )
            mapper = MunicipalGeoIntelMapper(seed_path=seed_path)
            assert mapper.loaded is True
            assert mapper._resolve_coastal_edge() == "east"

            b = mapper.bounds()
            mid_lat = (b["south"] + b["north"]) / 2.0
            # Eastern shoreline scores one; the inland western flank scores zero.
            assert mapper._coast_proximity(mid_lat, b["east"], b) == 1.0
            self.assertAlmostEqual(mapper._coast_proximity(mid_lat, b["west"], b), 0.0)

            # End-to-end: coastal-weighted cells are denser east of the midpoint.
            cells = mapper.generate_h3_cells()
            assert len(cells) > 0
            mid_lng = self._mid_lng(b)
            east_density = max(
                (
                    d["hazard_density"]
                    for c, d in cells.items()
                    if h3.cell_to_latlng(c)[1] > mid_lng
                ),
                default=0.0,
            )
            west_density = max(
                (
                    d["hazard_density"]
                    for c, d in cells.items()
                    if h3.cell_to_latlng(c)[1] <= mid_lng
                ),
                default=0.0,
            )
            assert east_density > west_density

    def test_inland_contract_disables_coastal_weighting(self) -> None:
        """A landlocked contract (coastalEdge='none') treats no bounds edge as a
        coast, so coastal proximity is neutral everywhere."""
        import tempfile

        from geo_infer_place.locations.del_norte_county.crescent_city_intel import (
            MunicipalGeoIntelMapper,
        )

        with tempfile.TemporaryDirectory(prefix="intel_inland_") as td:
            seed_path = _write_contract(
                td, coastal_edge="none", bounds=self._EAST_BOUNDS
            )
            mapper = MunicipalGeoIntelMapper(seed_path=seed_path)
            assert mapper.loaded is True
            assert mapper._resolve_coastal_edge() is None

            b = mapper.bounds()
            mid_lat = (b["south"] + b["north"]) / 2.0
            mid_lng = self._mid_lng(b)
            for lat, lng in [
                (mid_lat, b["west"]),
                (mid_lat, b["east"]),
                (b["north"], mid_lng),
                (b["south"], mid_lng),
            ]:
                assert mapper._coast_proximity(lat, lng, b) == 0.0

            # Coast-driven hazard weight drops to the seat-only term.
            dom = {"id": "hazard-policy", "hazardTags": ["erosion", "flood zone"]}
            seat = mapper._seat_proximity(mid_lat, mid_lng)
            self.assertAlmostEqual(
                mapper._domain_weight(dom, mid_lat, mid_lng), 0.30 * seat
            )


class TestModuleEnrichment(unittest.TestCase):
    """RISK / BAYES / ACT civic-intel results wired onto the dashboard surface.

    ``DelNorteComprehensiveDashboard.enrich_civic_intel_with_module_results``
    feeds the SAME ``crescent-city-geo-intel/v1`` contract into each sibling
    module's civic_intel helper and reconciles the computed risk weights,
    categorical priors and ACT policy preference onto a ``moduleWeights`` row
    set that the map renders. These tests use the real packaged seed and the
    real module helpers (no mocks) and pin the deterministic values.
    """

    def _dashboard(self) -> Any:
        from geo_infer_place.locations.del_norte_county.comprehensive_dashboard import (
            DelNorteComprehensiveDashboard,
        )

        return DelNorteComprehensiveDashboard()

    def _enrich(self) -> dict[str, Any]:
        dashboard = self._dashboard()
        return dashboard.enrich_civic_intel_with_module_results(
            {"status": "ok"}, contract=_seed_json()
        )

    def test_risk_bayes_act_modules_all_compute(self) -> None:
        enriched = self._enrich()
        sources = enriched["moduleResults"]["sources"]
        assert sources == {"risk": "ok", "bayes": "ok", "act": "ok"}

        risk = enriched["riskWeights"]
        assert isinstance(risk, dict)
        # Tsunami and seismic are the flagship Del Norte hazard-policy domains
        # and must carry section-evidence weights in the unit interval.
        assert "tsunami" in risk
        assert "seismic" in risk
        for weight in risk.values():
            assert 0.0 <= float(weight) <= 1.0
        self.assertAlmostEqual(float(risk["tsunami"]), 1.0, places=3)

        bayes = enriched["bayesPriors"]
        assert bayes["status"] == "ok"
        probs = bayes["priorByDomain"]
        assert len(probs) >= 1
        self.assertAlmostEqual(sum(float(p) for p in probs.values()), 1.0, places=6)
        assert "emergency-management" in probs

        act = enriched["actPolicy"]
        assert act["status"] == "ok"
        assert act["deterministic"] is True
        assert act["dominantHazard"] == "tsunami"
        # The deterministic argmin policy selector favours the all-clear state
        # (highest preference) over every hazard-avoiding candidate.
        assert act["selectedAction"]["action"] == "all-clear"

    def test_module_rows_reconcile_risk_prior_and_policy(self) -> None:
        enriched = self._enrich()
        rows = enriched["moduleWeights"]
        assert isinstance(rows, list) and len(rows) >= 1
        # Rows are sorted by descending RISK weight.
        risks = [float(row["riskWeight"]) for row in rows]
        assert risks == sorted(risks, reverse=True)
        for row in rows:
            assert "tag" in row
            assert 0.0 <= float(row["riskWeight"]) <= 1.0
            if row["priorWeight"] is not None:
                assert 0.0 <= float(row["priorWeight"]) <= 1.0
            if row["preference"] is not None:
                assert 0.0 <= float(row["preference"]) <= 1.0
            assert row["action"]
        # The tsunami row is flagged as the dominant hazard to mitigate.
        tsunami_row = next(r for r in rows if r["tag"] == "tsunami")
        assert "tsunami" in tsunami_row["action"]

    def test_module_results_are_deterministic(self) -> None:
        first = self._enrich()
        second = self._enrich()
        assert first["moduleWeights"] == second["moduleWeights"]
        assert first["riskWeights"] == second["riskWeights"]
        assert first["actPolicy"] == second["actPolicy"]

    def test_enrich_degrades_gracefully_on_missing_module(self) -> None:
        """A missing sibling module records ``unavailable`` and the rest compute."""
        name = "geo_infer_bayes"
        sub = "geo_infer_bayes.civic_intel"
        original_pkg = sys.modules.get(name)
        original_sub = sys.modules.get(sub)
        try:
            # Simulate an absent BAYES package via the real import machinery:
            # a ``None`` sys.modules entry makes ``importlib.import_module``
            # raise ImportError, exercising the same defensive path as an
            # uninstalled dependency (not a stubbed result).
            sys.modules[name] = None  # type: ignore[index]
            sys.modules[sub] = None  # type: ignore[index]
            enriched = self._enrich()
        finally:
            for key, value in ((name, original_pkg), (sub, original_sub)):
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

        assert enriched["moduleResults"]["status"] == "ok"
        assert enriched["moduleResults"]["sources"]["bayes"] == "unavailable"
        assert enriched["bayesPriors"]["status"] == "unavailable"
        # RISK and ACT are independent and still compute, so the map keeps
        # full civic-intel coverage without the missing module.
        assert "tsunami" in enriched["riskWeights"]
        assert enriched["actPolicy"]["status"] == "ok"

    def test_module_popup_block_escapes_hostile_tags(self) -> None:
        """The pre-built module popup block HTML-escapes contract-derived tags.

        The contract is an external data surface; a hostile tag must not inject
        markup into the generated dashboard.
        """
        dashboard = self._dashboard()
        hostile_rows = [
            {
                "tag": "<script>alert('x')</script>flood",
                "riskWeight": 1.0,
                "priorWeight": 0.5,
                "preference": 0.25,
                "action": "<img src=x onerror=alert(1)>avoid",
            }
        ]
        block = dashboard._module_rows_popup_html(hostile_rows)
        # The popup block renders the tag (external contract data) plus numeric
        # weights; a hostile tag must not inject markup.
        assert "<script>" not in block
        assert "&lt;script&gt;" in block

        # Contract-derived popup fields are escaped too.
        popup = dashboard._civic_intel_popup(
            "cell&1",
            0.5,
            ["<b>emergency"],
            {"<b>emergency": {"name": "<h1>Emergency", "section_count": 2}},
            ["<i>tsunami</i>"],
            module_rows_block=block,
        )
        assert "<h1>" not in popup
        assert "<i>" not in popup
        assert "&lt;h1&gt;" in popup
        assert "&lt;script&gt;" in popup

    def test_module_popup_block_is_invariant_and_empty_when_no_rows(self) -> None:
        """The popup block is deterministic; no rows yields an empty block."""
        dashboard = self._dashboard()
        enriched = self._enrich()
        rows = enriched["moduleWeights"]
        block = dashboard._module_rows_popup_html(rows)
        assert "Module results" in block
        assert block == dashboard._module_rows_popup_html(enriched["moduleWeights"])
        # The block is reused verbatim per cell, so it is identical across the
        # whole layer (single build, not per-cell recompute).
        assert dashboard._module_rows_popup_html(enriched["moduleWeights"]) == block
        assert dashboard._module_rows_popup_html([]) == ""

    def test_module_panel_escapes_tag_and_action(self) -> None:
        """The rendered module panel escapes contract-derived tag + action."""
        import folium  # type: ignore[import-not-found]

        dashboard = self._dashboard()
        result = {
            "moduleWeights": [
                {
                    "tag": "<b>tsunami</b>",
                    "riskWeight": 1.0,
                    "priorWeight": 0.2,
                    "preference": 0.1,
                    "action": "mitigate <script>alert(1)</script>",
                }
            ],
            "moduleResults": {"sources": {"risk": "ok", "bayes": "ok", "act": "ok"}},
        }
        m = folium.Map(location=[41.7, -124.2], zoom_start=10)
        dashboard._add_module_hazard_weights_panel(m, result)
        rendered = m.get_root().render()
        assert "<b>tsunami</b>" not in rendered
        assert "&lt;b&gt;tsunami&lt;/b&gt;" in rendered
        assert "<script>alert(1)</script>" not in rendered


if __name__ == "__main__":
    unittest.main()
