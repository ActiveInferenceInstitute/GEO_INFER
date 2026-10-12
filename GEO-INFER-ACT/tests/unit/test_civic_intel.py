"""Unit tests for Crescent City civic-intel ingestion and hazard policy-prior plating.

Covers parsing the ``crescent-city-geo-intel/v1`` contract (surfacing the
hazard-relevant domain subset) and the deterministic hazard prior that weights
policy selection away from notified municipal hazards.
"""

import json
from typing import Any

import numpy as np
import pytest

from geo_infer_act import (
    CivicIntelBounds,
    CrescentCityIntel,
    HazardDomain,
    hazard_policy_prior,
    parse_crescent_city_intel,
)
from geo_infer_act.core.civic_intel import (
    CRESCENT_CITY_OBSERVATIONS_SCHEMA,
    load_crescent_city_geo_observations,
)
from geo_infer_act.core.policy_selection import PolicySelector


def _hazard_contract_fixture() -> dict[str, Any]:
    """A small inline crescent-city-geo-intel/v1 contract with a hazard subset."""
    return {
        "schema": "crescent-city-geo-intel/v1",
        "anchor": {
            "name": "Crescent City",
            "guid": "CR4919",
            "municipality": "Crescent City, CA",
            "county": "Del Norte County",
            "state": "California",
            "latitude": 41.76,
            "longitude": -124.2,
            "bounds": {
                "west": -124.408,
                "south": 41.458,
                "east": -123.536,
                "north": 42.006,
            },
        },
        "generatedAt": "2026-08-23T21:38:42.863Z",
        "domainCount": 12,
        "domains": [
            {
                "id": "emergency-management",
                "name": "Emergency Management",
                "tags": ["tsunami", "seismic", "evacuation"],
                "sections": [],
            },
            {
                "id": "business-development",
                "name": "Business Development",
                "tags": ["crabbing", "harbor"],
                "sections": [],
            },
        ],
        "hazard": {
            "relevantDomains": [
                {
                    "id": "emergency-management",
                    "name": "Emergency Management",
                    "icon": "🌊",
                    "hazardTags": ["seismic", "tsunami"],
                    "topics": [
                        {
                            "name": "Tsunami Preparedness & Evacuation",
                            "tags": ["tsunami", "seismic"],
                            "sections": [
                                {
                                    "sectionNumber": "§ 8.04",
                                    "relevance": "Health and Safety - emergency management authority",
                                }
                            ],
                        }
                    ],
                }
            ],
            "relevantDomainCount": 1,
        },
    }


class TestParseCrescentCityIntel:
    """Tests for lowering the geo-intel contract into an ACT record."""

    def test_parse_surfaces_hazard_subset(self) -> None:
        """The tsunami/seismic hazard domain is surfaced by the parser."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())

        assert parsed["schema"] == "crescent-city-geo-intel/v1"
        assert parsed["city"] == "Crescent City"
        assert parsed["bounds"] == {
            "west": -124.408,
            "south": 41.458,
            "east": -123.536,
            "north": 42.006,
        }
        hazard_domains = parsed["hazardDomains"]
        assert len(hazard_domains) == 1
        assert hazard_domains[0]["id"] == "emergency-management"
        assert "tsunami" in hazard_domains[0]["hazardTags"]
        assert "seismic" in hazard_domains[0]["hazardTags"]

    def test_parse_surfaces_topic_sections(self) -> None:
        """The municipal code section nested under the hazard topic is retained."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())
        topic = parsed["hazardDomains"][0]["topics"][0]
        assert topic["name"] == "Tsunami Preparedness & Evacuation"
        assert topic["sections"][0]["sectionNumber"] == "§ 8.04"

    def test_parse_is_deterministic_across_seeds(self) -> None:
        """Parsing output is identical regardless of the accepted seed."""
        base = parse_crescent_city_intel(seed=0, source=_hazard_contract_fixture())
        reseeded = parse_crescent_city_intel(seed=42, source=_hazard_contract_fixture())
        assert base["city"] == reseeded["city"]
        assert base["hazardDomains"] == reseeded["hazardDomains"]
        assert base["bounds"] == reseeded["bounds"]

    def test_parse_graceful_when_absent(self) -> None:
        """A missing contract degrades to an empty record rather than raising."""
        parsed = parse_crescent_city_intel(source="/nonexistent/geo-intel.json")
        assert parsed["city"] == ""
        assert parsed["hazardDomains"] == []
        assert parsed["bounds"] == {}

    def test_default_source_is_the_bundled_bayes_copy(self) -> None:
        """With no source, parsing reads the canonical bundled BAYES contract."""
        parsed = parse_crescent_city_intel()

        assert parsed["schema"] == "crescent-city-geo-intel/v1"
        assert parsed["city"] == "Crescent City"
        assert parsed["hazardDomains"]

    def test_parse_accepts_documented_json_string(self) -> None:
        """An injected JSON object string uses the same parser as a mapping."""
        contract = _hazard_contract_fixture()

        from_mapping = parse_crescent_city_intel(source=contract)
        from_json = parse_crescent_city_intel(source=json.dumps(contract))

        assert from_json == from_mapping

    def test_malformed_json_and_hazard_shape_fail_closed(self) -> None:
        """Malformed injected content raises ValueError, never raw parser errors."""
        with pytest.raises(ValueError, match="invalid Crescent City intel JSON"):
            parse_crescent_city_intel(source="{not-json")

        contract = _hazard_contract_fixture()
        contract["hazard"] = None
        with pytest.raises(ValueError, match="hazard must be an object"):
            parse_crescent_city_intel(source=contract)

    def test_malformed_bounds_fail_closed(self) -> None:
        """Non-numeric and inverted bounds raise contract-facing ValueError."""
        non_numeric = _hazard_contract_fixture()
        anchor = non_numeric["anchor"]
        assert isinstance(anchor, dict)
        bounds = anchor["bounds"]
        assert isinstance(bounds, dict)
        bounds["west"] = "not-a-coordinate"
        with pytest.raises(
            ValueError, match="anchor.bounds.west must be a finite number"
        ):
            parse_crescent_city_intel(source=non_numeric)

        inverted = _hazard_contract_fixture()
        inverted_anchor = inverted["anchor"]
        assert isinstance(inverted_anchor, dict)
        inverted_bounds = inverted_anchor["bounds"]
        assert isinstance(inverted_bounds, dict)
        inverted_bounds["west"] = -123.0
        inverted_bounds["east"] = -124.0
        with pytest.raises(ValueError, match="west < east"):
            parse_crescent_city_intel(source=inverted)

    def test_fallback_scan_recognizes_qualified_hazard_tags(self) -> None:
        """Whole-term matching retains qualified tags when no subset is supplied."""
        contract = _hazard_contract_fixture()
        contract["hazard"] = {}
        contract["domains"] = [
            {
                "id": "flood-policy",
                "name": "Flood Policy",
                "tags": ["flood zone", "harbor"],
            }
        ]

        parsed = parse_crescent_city_intel(source=contract)

        assert parsed["hazardDomains"][0]["id"] == "flood-policy"
        assert parsed["hazardDomains"][0]["hazardTags"] == ["flood zone"]

    def test_parse_rejects_schema_mismatch(self) -> None:
        """A wrong or missing schema fails closed like BAYES/RISK, never empty."""
        with pytest.raises(ValueError, match="unexpected Crescent City intel schema"):
            parse_crescent_city_intel(source={"schema": "other/v1", "anchor": {}})

        with pytest.raises(ValueError, match="unexpected Crescent City intel schema"):
            parse_crescent_city_intel(source={"anchor": {}})

    def test_typed_view_constructs(self) -> None:
        """The typed dataclass view is directly constructible."""
        domain = HazardDomain(
            id="emergency-management",
            name="Emergency Management",
            hazardTags=["tsunami", "seismic"],
        )
        bounds = CivicIntelBounds(
            west=-124.408, south=41.458, east=-123.536, north=42.006
        )
        record = CrescentCityIntel(
            city="Crescent City",
            hazardDomains=[domain],
            bounds=bounds,
            schema="crescent-city-geo-intel/v1",
        )
        assert record.as_dict()["city"] == "Crescent City"
        assert record.as_dict()["bounds"]["north"] == 42.006


class TestHazardPolicyPrior:
    """Tests for mapping hazard domains into PolicySelector preferences."""

    def test_prior_lowers_hazard_preferences(self) -> None:
        """Tsunami and seismic states are less preferred than the baseline."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())
        prior = hazard_policy_prior(parsed)

        assert prior["deterministic"] is True
        assert prior["hazardTags"] == ["seismic", "tsunami"]
        assert prior["dominantHazard"] == "tsunami"
        preferences = prior["preferences"]
        # Axis: [baseline(all-clear), seismic, tsunami]
        assert preferences[0] == pytest.approx(np.max(preferences), rel=1e-6)
        assert preferences[2] < preferences[1] < preferences[0]
        assert preferences[0] >= 0.5
        assert float(np.sum(preferences)) == pytest.approx(1.0, abs=1e-9)

    def test_prior_accepts_raw_contract_dict(self) -> None:
        """The prior consumes a raw contract even without a prior parse."""
        prior = hazard_policy_prior(_hazard_contract_fixture())
        assert prior["hazardTags"] == ["seismic", "tsunami"]
        assert prior["deterministic"] is True

    def test_prior_empty_when_no_hazard_signal(self) -> None:
        """A contract without hazard domains yields a baseline-only prior."""
        flat = {
            "schema": "crescent-city-geo-intel/v1",
            "anchor": {"name": "X"},
            "hazard": {},
        }
        prior = hazard_policy_prior(parse_crescent_city_intel(source=flat))
        assert prior["hazardTags"] == []
        assert prior["dominantHazard"] is None
        assert len(prior["preferences"]) == 1
        assert float(prior["preferences"][0]) == pytest.approx(1.0)

    def test_prior_deterministic_repeat(self) -> None:
        """Seed-default reproduction is stable across calls."""
        contract = _hazard_contract_fixture()
        first = hazard_policy_prior(parse_crescent_city_intel(source=contract))
        second = hazard_policy_prior(parse_crescent_city_intel(source=contract))
        np.testing.assert_array_equal(first["preferences"], second["preferences"])

    def test_seeded_hedge_is_reproducible(self) -> None:
        """A seeded hedge is stable while a deterministic default stays fixed."""
        contract = _hazard_contract_fixture()
        parsed = parse_crescent_city_intel(source=contract)
        base = hazard_policy_prior(parsed, seed=None)
        hedged_a = hazard_policy_prior(parsed, seed=7, hedge_share=0.2)
        hedged_b = hazard_policy_prior(parsed, seed=7, hedge_share=0.2)
        np.testing.assert_array_equal(hedged_a["preferences"], hedged_b["preferences"])
        assert hedged_a["deterministic"] is False
        assert not np.allclose(base["preferences"], hedged_a["preferences"])

    @pytest.mark.parametrize(
        "hedge_share",
        [-0.1, 1.1, float("nan"), float("inf"), float("-inf")],
    )
    def test_invalid_hedge_share_fails_closed(self, hedge_share: float) -> None:
        """Invalid hedge fractions cannot masquerade as stochastic priors."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())

        with pytest.raises(ValueError, match=r"hedge_share.*\[0, 1\]"):
            hazard_policy_prior(parsed, hedge_share=hedge_share)

    def test_qualified_hazard_tags_use_base_weights_without_substrings(self) -> None:
        """Qualified v1 tags resolve by whole hazard terms, not substrings."""
        parsed: dict[str, Any] = {
            "hazardDomains": [
                {
                    "id": "qualified-hazards",
                    "hazardTags": [
                        "flood zone",
                        "tsunami zone",
                        "tsunami drill",
                        "backfire",
                        "tsunamic",
                    ],
                    "topics": [],
                }
            ]
        }

        prior = hazard_policy_prior(parsed)

        assert prior["weights"]["flood zone"] == pytest.approx(0.75)
        assert prior["weights"]["tsunami zone"] == pytest.approx(0.90)
        assert prior["weights"]["tsunami drill"] == pytest.approx(0.90)
        assert prior["weights"]["backfire"] == pytest.approx(0.50)
        assert prior["weights"]["tsunamic"] == pytest.approx(0.50)


class TestPolicyCoupling:
    """Verify the hazard prior drives a deterministic lowest-EFE selection."""

    def test_deterministic_selection_avoids_notified_hazards(self) -> None:
        """The policy predicting the all-clear state wins under the hazard prior."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())
        prior = hazard_policy_prior(parsed)

        policies: list[dict[str, Any]] = [
            {
                "id": "maintain_normal_ops",
                "predicted_beliefs": [1.0, 0.0, 0.0],
            },
            {
                "id": "shore_up_structures",
                "predicted_beliefs": [0.0, 1.0, 0.0],
            },
            {
                "id": "approach_waterfront",
                "predicted_beliefs": [0.0, 0.0, 1.0],
            },
        ]
        beliefs = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        selector = PolicySelector(selection_mode="deterministic", random_seed=11)

        result = selector.select_policy(beliefs, policies, preferences=prior)
        evaluation = selector.evaluate_policy_set(beliefs, policies, preferences=prior)

        assert result["selected_index"] == 0
        assert result["policy"]["id"] == "maintain_normal_ops"
        assert evaluation["best_policy_idx"] == 0
        # The safe policy has the lowest expected free energy.
        energies = evaluation["expected_free_energies"]
        assert energies[0] < energies[1]
        assert energies[0] < energies[2]

    def test_deterministic_rule_matches_manual_efe(self) -> None:
        """Selection equals argmin of the computed expected free energies."""
        parsed = parse_crescent_city_intel(source=_hazard_contract_fixture())
        prior = hazard_policy_prior(parsed)
        policies = [
            {"id": "safe", "predicted_beliefs": [1.0, 0.0, 0.0]},
            {"id": "hazard", "predicted_beliefs": [0.0, 0.0, 1.0]},
        ]
        beliefs = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        selector = PolicySelector(selection_mode="deterministic")
        result = selector.select_policy(beliefs, policies, preferences=prior)
        assert result["selected_index"] == int(np.argmin(result["all_free_energies"]))
        assert result["policy"]["id"] == "safe"


# --- crescent-city-geo-observations/v1 delegation (2026-09-08 producer pass) ---


class TestObservationsDelegation:
    """Live hazard observations resolve through the BAYES ingestion core."""

    def test_loader_is_the_bayes_core_object(self) -> None:
        """ACT delegates to the one canonical observations loader."""
        from geo_infer_bayes.geo_observations import (
            load_crescent_city_geo_observations as bayes_loader,
        )

        assert load_crescent_city_geo_observations is bayes_loader

    def test_schema_constant_matches_producer_envelope(self) -> None:
        assert CRESCENT_CITY_OBSERVATIONS_SCHEMA == "crescent-city-geo-observations/v1"

    def test_bundled_snapshot_loads_with_live_monitors(self) -> None:
        envelope = load_crescent_city_geo_observations()
        assert envelope is not None
        assert envelope["schema"] == CRESCENT_CITY_OBSERVATIONS_SCHEMA
        assert len(envelope["monitors"]) == 15

    def test_unavailable_envelope_passes_validation(self) -> None:
        envelope = load_crescent_city_geo_observations(
            {
                "schema": CRESCENT_CITY_OBSERVATIONS_SCHEMA,
                "anchor": {
                    "name": "Crescent City",
                    "guid": "CR4919",
                    "municipality": "Crescent City, CA",
                    "county": "Del Norte County",
                    "state": "California",
                    "latitude": 41.76,
                    "longitude": -124.2,
                },
                "generatedAt": "2026-09-08T00:00:00.000Z",
                "composite": None,
                "monitors": [],
                "hazardSummary": [],
                "freshness": {
                    "contractSchema": "crescent-city-geo-intel/v1",
                    "contractGeneratedAt": None,
                },
            }
        )
        assert envelope is not None
        assert envelope["composite"] is None


def test_bayes_validator_helpers_are_public_contract() -> None:
    """ACT consumes the crescent-city contract via public BAYES names only.

    Regression pin for the underscore-import coupling: the require/parse
    helpers ACT's non-degraded path needs are part of the cross-module
    contract and must stay importable by their public names from
    ``geo_infer_bayes.civic_intel``.
    """
    import geo_infer_bayes.civic_intel as bayes_civic_intel

    from geo_infer_act.core.civic_intel import (
        parse_contract_bounds,
        require_list,
        require_mapping,
    )

    for name in ("parse_contract_bounds", "require_list", "require_mapping"):
        assert name in bayes_civic_intel.__all__, name
        assert getattr(bayes_civic_intel, name) is not None
    assert parse_contract_bounds is bayes_civic_intel.parse_contract_bounds
    assert require_list is bayes_civic_intel.require_list
    assert require_mapping is bayes_civic_intel.require_mapping

    bounds = parse_contract_bounds(
        {"west": -124.4, "south": 41.4, "east": -123.5, "north": 42.0}
    )
    assert bounds["west"] == -124.4
    assert require_list([1, 2], "field") == [1, 2]
    assert require_mapping({"a": 1}, "field") == {"a": 1}
