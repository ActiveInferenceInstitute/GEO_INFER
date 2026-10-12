"""Contract tests for Crescent City civic hazard-intelligence ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_infer_risk.civic_intel import (
    CRESCENT_CITY_OBSERVATIONS_SCHEMA,
    load_crescent_city_geo_observations,
)
from geo_infer_risk import (
    MultiHazardInteractionMatrix,
    crescent_city_hazard_weights,
    load_crescent_city_hazard,
    parse_crescent_city_hazard,
)


def _contract_fixture() -> dict[str, object]:
    """Return a small but structurally complete v1 contract fixture."""

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
        "hazard": {
            "relevantDomains": [
                {
                    "id": "emergency-management",
                    "name": "Emergency Management",
                    "hazardTags": ["seismic", "tsunami"],
                    "topics": [
                        {
                            "name": "Tsunami Preparedness & Evacuation",
                            "tags": ["tsunami", "seismic"],
                            "sections": [
                                {
                                    "sectionNumber": "§ 8.04",
                                    "relevance": "Emergency management authority",
                                },
                                {
                                    "sectionNumber": "§ 15.04",
                                    "relevance": "Seismic and flood requirements",
                                },
                            ],
                        }
                    ],
                },
                {
                    "id": "environmental-protection",
                    "name": "Environmental Protection",
                    "hazardTags": ["flood"],
                    "topics": [
                        {
                            "name": "Flood-zone construction",
                            "tags": ["flood"],
                            "sections": [
                                {
                                    "sectionNumber": "§ 15.32",
                                    "relevance": "Floodplain development standards",
                                }
                            ],
                        }
                    ],
                },
            ],
            "relevantDomainCount": 2,
        },
    }


def test_inline_contract_surfaces_city_hazards_and_code_sections() -> None:
    """The public loader projects Crescent City's hazard policy contract."""

    result = load_crescent_city_hazard(_contract_fixture())

    assert result["city"] is not None
    assert result["city"]["name"] == "Crescent City"
    assert result["city"]["latitude"] == pytest.approx(41.76)
    assert result["city"]["bounds"] == result["bounds"]
    assert result["bounds"] == {
        "west": -124.408,
        "south": 41.458,
        "east": -123.536,
        "north": 42.006,
    }
    emergency = result["hazardDomains"][0]
    assert emergency["id"] == "emergency-management"
    assert emergency["hazardTags"] == ["seismic", "tsunami"]
    assert emergency["sections"] == [
        {
            "sectionNumber": "§ 8.04",
            "relevance": "Emergency management authority",
        },
        {
            "sectionNumber": "§ 15.04",
            "relevance": "Seismic and flood requirements",
        },
    ]


def test_bundled_gold_surfaces_reviewed_hazard_policy() -> None:
    """The default loader reads the reviewed offline Crescent City gold."""

    result = load_crescent_city_hazard()

    assert result["city"] is not None
    assert result["city"]["name"] == "Crescent City"
    assert result["city"]["county"] == "Del Norte County"
    assert {domain["id"] for domain in result["hazardDomains"]} >= {
        "emergency-management",
        "environmental-protection",
        "event-planning",
        "climate-environment",
    }
    # Refreshed 2026-09-08: emergency-management now also carries the
    # "earthquake" tag; its topic sections hold 6 distinct entries.
    assert result["hazardDomains"][0]["hazardTags"] == [
        "earthquake",
        "seismic",
        "tsunami",
    ]
    assert len(result["hazardDomains"][0]["sections"]) == 6


def test_local_json_path_and_missing_seed_are_deterministic(tmp_path: Path) -> None:
    """An explicit local seed is deterministic and a missing path stays empty."""

    seed_path = tmp_path / "geo-intel.json"
    seed_path.write_text(json.dumps(_contract_fixture()), encoding="utf-8")

    assert load_crescent_city_hazard(seed_path) == parse_crescent_city_hazard(
        _contract_fixture()
    )
    empty = {"city": None, "hazardDomains": [], "bounds": None}
    assert load_crescent_city_hazard(tmp_path / "missing.json") == empty


def test_source_is_the_only_input_parameter() -> None:
    """``source`` is the single input; the removed ``seed`` alias is rejected."""

    fixture = _contract_fixture()

    assert load_crescent_city_hazard(fixture) == parse_crescent_city_hazard(fixture)
    assert load_crescent_city_hazard(source=fixture) == parse_crescent_city_hazard(
        fixture
    )
    with pytest.raises(TypeError, match="source must be"):
        load_crescent_city_hazard(object())
    with pytest.raises(TypeError, match="unexpected keyword argument 'seed'"):
        load_crescent_city_hazard(seed=fixture)  # type: ignore[call-arg]


def test_policy_weights_align_with_multi_hazard_matrix_names() -> None:
    """Municipal section evidence maps to matrix hazards without causal rewiring."""

    intel = load_crescent_city_hazard(_contract_fixture())
    matrix = MultiHazardInteractionMatrix(
        ["earthquake", "tsunami", "flood", "wildfire"]
    )

    assert crescent_city_hazard_weights(
        intel,
        matrix.hazard_types,
        default_weight=0.25,
    ) == {
        "earthquake": 1.0,
        "tsunami": 1.0,
        "flood": 0.5,
        "wildfire": 0.25,
    }
    assert matrix.get_interaction("earthquake", "tsunami") == 0.0


def test_policy_weights_pool_composite_tags_without_substring_inference() -> None:
    """Qualified v1 tags map to hazards while unrelated word stems do not."""

    contract = _contract_fixture()
    hazard = contract["hazard"]
    assert isinstance(hazard, dict)
    domains = hazard["relevantDomains"]
    assert isinstance(domains, list)
    emergency = domains[0]
    environmental = domains[1]
    assert isinstance(emergency, dict)
    assert isinstance(environmental, dict)
    emergency["hazardTags"] = ["seismic", "tsunami zone", "tsunami drill"]
    emergency_topics = emergency["topics"]
    assert isinstance(emergency_topics, list)
    assert isinstance(emergency_topics[0], dict)
    emergency_topics[0]["tags"] = ["seismic", "tsunami zone", "tsunami drill"]
    composite_tags = [
        "flood zone",
        "wildfire smoke",
        "climate adaptation",
        "sea-level rise",
        "stormwater",
    ]
    environmental["hazardTags"] = composite_tags
    environmental_topics = environmental["topics"]
    assert isinstance(environmental_topics, list)
    assert isinstance(environmental_topics[0], dict)
    environmental_topics[0]["tags"] = composite_tags

    intel = load_crescent_city_hazard(contract)

    assert crescent_city_hazard_weights(
        intel,
        [
            "earthquake",
            "tsunami",
            "flood",
            "wildfire",
            "climate",
            "sea level",
            "storm",
        ],
        default_weight=0.25,
    ) == {
        "earthquake": 1.0,
        "tsunami": 1.0,
        "flood": 0.5,
        "wildfire": 0.5,
        "climate": 0.5,
        "sea level": 0.5,
        "storm": 0.25,
    }


def test_parser_rejects_contract_version_drift() -> None:
    """A non-v1 payload fails closed instead of being silently reinterpreted."""

    contract = _contract_fixture()
    contract["schema"] = "crescent-city-geo-intel/v2"

    with pytest.raises(ValueError, match="crescent-city-geo-intel/v1"):
        parse_crescent_city_hazard(contract)


def test_parser_rejects_inconsistent_hazard_topics() -> None:
    """Topic tags cannot escape their domain hazardTags declaration."""

    contract = _contract_fixture()
    hazard = contract["hazard"]
    assert isinstance(hazard, dict)
    domains = hazard["relevantDomains"]
    assert isinstance(domains, list)
    first_domain = domains[0]
    assert isinstance(first_domain, dict)
    topics = first_domain["topics"]
    assert isinstance(topics, list)
    first_topic = topics[0]
    assert isinstance(first_topic, dict)
    first_topic["tags"] = ["wildfire"]

    with pytest.raises(ValueError, match="must be listed"):
        parse_crescent_city_hazard(contract)


def test_existing_malformed_json_fails_closed(tmp_path: Path) -> None:
    """An existing invalid JSON file is not treated as an absent seed."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Crescent City geo-intel JSON"):
        load_crescent_city_hazard(invalid)


# --- crescent-city-geo-observations/v1 delegation ---


def test_observations_loader_is_the_bayes_core_object() -> None:
    """RISK delegates to the one canonical observations loader."""
    from geo_infer_bayes.geo_observations import (
        load_crescent_city_geo_observations as bayes_loader,
    )

    assert load_crescent_city_geo_observations is bayes_loader


def test_observations_schema_constant_matches_producer_envelope() -> None:
    assert CRESCENT_CITY_OBSERVATIONS_SCHEMA == "crescent-city-geo-observations/v1"


def test_bundled_observations_snapshot_loads_with_live_monitors() -> None:
    envelope = load_crescent_city_geo_observations()
    assert envelope is not None
    assert envelope["schema"] == CRESCENT_CITY_OBSERVATIONS_SCHEMA
    assert len(envelope["monitors"]) == 15
    assert envelope["anchor"]["county"] == "Del Norte County"


def test_unavailable_observations_envelope_passes_validation() -> None:
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
