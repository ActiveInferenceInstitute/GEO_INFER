"""Cross-module consistency for the Crescent City civic-intel contract.

RISK (``geo_infer_risk.civic_intel``), BAYES (``geo_infer_bayes.civic_intel``)
and ACT (``geo_infer_act.core.civic_intel``) each independently ingest the
``crescent-city-geo-intel/v1`` contract and its hazard-relevant domain subset.
There is no single consumer that proves they interpret the *same* contract
consistently, so a future contract change could silently diverge between
modules.  These tests feed ONE shared fixture through all three public
ingestors and pin the hazard surface (domain ids and word-boundary qualified
tags) to be identical everywhere.

The fixture uses a compact word-boundary 4-domain hazard surface:

- ``emergency-management``: seismic / tsunami
- ``environmental-protection``: erosion / flood
- ``event-planning``: tsunami-drill
- ``climate-environment``: sea-level / flood

All assertions use the real civic-intel modules - no mocks, no stubbing.
"""

from __future__ import annotations

from collections.abc import Iterable


from geo_infer_risk import (
    crescent_city_hazard_weights,
    load_crescent_city_hazard,
)
from geo_infer_act import (
    hazard_policy_prior,
    parse_crescent_city_intel,
)
from geo_infer_bayes import load_crescent_city_intel


from geo_infer_act.core import civic_intel as act_civic
from geo_infer_bayes import civic_intel as bayes_civic
from geo_infer_risk import civic_intel as risk_civic

_SCHEMA = "crescent-city-geo-intel/v1"

_ANCHOR = {
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
}

_HAZARD_DOMAINS = [
    {
        "id": "emergency-management",
        "name": "Emergency Management",
        "icon": "🌊",
        "hazardTags": ["seismic", "tsunami"],
        "topics": [
            {
                "name": "Tsunami Preparedness & Evacuation",
                "tags": ["seismic", "tsunami"],
                "sections": [
                    {
                        "sectionNumber": "8.04",
                        "relevance": "Emergency management authority",
                    },
                    {
                        "sectionNumber": "15.04",
                        "relevance": "Seismic and flood requirements",
                    },
                ],
            }
        ],
    },
    {
        "id": "environmental-protection",
        "name": "Environmental Protection",
        "icon": "🌊",
        "hazardTags": ["erosion", "flood"],
        "topics": [
            {
                "name": "Coastal erosion and floodplain policy",
                "tags": ["erosion", "flood"],
                "sections": [
                    {
                        "sectionNumber": "15.32",
                        "relevance": "Floodplain development standards",
                    }
                ],
            }
        ],
    },
    {
        "id": "event-planning",
        "name": "Event Planning",
        "icon": "📋",
        "hazardTags": ["tsunami-drill"],
        "topics": [
            {
                "name": "Tsunami drill coordination",
                "tags": ["tsunami-drill"],
                "sections": [
                    {
                        "sectionNumber": "18.02",
                        "relevance": "Drill coordination permit",
                    }
                ],
            }
        ],
    },
    {
        "id": "climate-environment",
        "name": "Climate Environment",
        "icon": "🌍",
        "hazardTags": ["sea-level", "flood"],
        "topics": [
            {
                "name": "Sea-level rise and storm flood",
                "tags": ["sea-level", "flood"],
                "sections": [
                    {
                        "sectionNumber": "12.07",
                        "relevance": "Sea-level rise adaptation overlay",
                    }
                ],
            }
        ],
    },
]


def _civic_domains() -> list[dict[str, object]]:
    """Return full civic-domain entries for the same hazard domains.

    BAYES requires every hazard-relevant domain id to appear in the full
    ``domains`` list. The other consumers ignore the full list, so sharing
    one field keeps the fixture valid for all three ingestors.
    """
    civic: list[dict[str, object]] = []
    for domain in _HAZARD_DOMAINS:
        tags = list(domain["hazardTags"])
        civic.append(
            {
                "id": domain["id"],
                "name": domain["name"],
                "icon": domain["icon"],
                "description": f"Crescent City {domain['name'].lower()} policy.",
                "updatedAt": "2026-08-23",
                "topicCount": 1,
                "tags": tags,
                "sections": [],
            }
        )
    return civic


def _shared_contract() -> dict[str, object]:
    """Return ONE contract fixture fed to RISK, BAYES and ACT alike."""
    hazard_domains = list(_HAZARD_DOMAINS)
    return {
        "schema": _SCHEMA,
        "anchor": dict(_ANCHOR),
        "generatedAt": "2026-08-23T00:00:00.000Z",
        "domainCount": len(hazard_domains),
        "domains": _civic_domains(),
        "hazard": {
            "relevantDomains": hazard_domains,
            "relevantDomainCount": len(hazard_domains),
        },
    }


def _normalize_tag(tag: str) -> str:
    """Collapse a producer tag to one canonical (lowercase) token.

    The real contract can carry qualified tags like ``sea-level`` or ``flood
    zone``. RISK and ACT both fold whitespace and case during parsing, so the
    cross-module comparison operates on the normalized token space.
    """
    return " ".join(tag.lower().replace("_", " ").replace("-", " ").split())


def _hazard_tag_set(hazard_domains: Iterable[dict[str, object]]) -> set[str]:
    """Collect the union of domain-level hazard tags from a parsed surface."""
    tags: set[str] = set()
    for domain in hazard_domains:
        for tag in domain.get("hazardTags", []):
            tags.add(_normalize_tag(str(tag)))
    return tags


def test_shared_contract_surfaces_identical_hazard_domains() -> None:
    """All three ingestors surface the same four hazard-relevant domain ids."""
    contract = _shared_contract()

    risk = load_crescent_city_hazard(contract)
    bayes = load_crescent_city_intel(source=contract)
    act = parse_crescent_city_intel(source=contract)

    risk_domains = {domain["id"] for domain in risk["hazardDomains"]}
    bayes_domains = {domain["id"] for domain in bayes["hazardDomains"]}
    act_domains = {domain["id"] for domain in act["hazardDomains"]}

    assert risk_domains == {
        "emergency-management",
        "environmental-protection",
        "event-planning",
        "climate-environment",
    }
    assert risk_domains == bayes_domains == act_domains


def test_risk_and_bayes_and_act_agree_on_hazard_tags() -> None:
    """The word-boundary hazard tag union is identical across consumers."""
    contract = _shared_contract()

    risk = load_crescent_city_hazard(contract)
    bayes = load_crescent_city_intel(source=contract)
    act = parse_crescent_city_intel(source=contract)

    risk_tags = _hazard_tag_set(risk["hazardDomains"])
    bayes_tags = _hazard_tag_set(bayes["hazardDomains"])
    act_tags = _hazard_tag_set(act["hazardDomains"])

    expected = {"seismic", "tsunami", "erosion", "flood", "tsunami drill", "sea level"}
    assert risk_tags == expected
    assert risk_tags == bayes_tags == act_tags


def test_flood_and_sea_level_surface_in_every_consumer() -> None:
    """The two across-domain hazards cross every parse boundary."""
    contract = _shared_contract()

    for parsed in (
        load_crescent_city_hazard(contract)["hazardDomains"],
        load_crescent_city_intel(source=contract)["hazardDomains"],
        parse_crescent_city_intel(source=contract)["hazardDomains"],
    ):
        tags = _hazard_tag_set(parsed)
        assert "flood" in tags
        assert "sea level" in tags


def test_bayes_per_domain_tags_match_risk_weighted_hazards() -> None:
    """BAYES prior tag evidence aligns one-for-one with RISK hazard weights."""
    contract = _shared_contract()
    risk_intel = load_crescent_city_hazard(contract)
    bayes_intel = load_crescent_city_intel(source=contract)

    bayes_tags = {
        tag for domain in bayes_intel["hazardDomains"] for tag in domain["hazardTags"]
    }
    expected_raw = {
        "seismic",
        "tsunami",
        "erosion",
        "flood",
        "tsunami-drill",
        "sea-level",
    }
    assert bayes_tags == expected_raw

    # Every BAYES hazard tag must be a RISK-weighted hazard with non-zero section
    # evidence. RISK folds qualified tags onto whole-phrase boundaries, so the
    # raw producer token and the returned hazard key stay in lockstep.
    risk_weights = crescent_city_hazard_weights(
        risk_intel,
        sorted(bayes_tags),
        default_weight=0.0,
    )
    assert set(risk_weights) == bayes_tags
    assert all(weight > 0.0 for weight in risk_weights.values())

    # The two flood-bearing domains and the sea-level overlay are all weighted.
    assert risk_weights["flood"] > 0.0
    assert risk_weights["sea-level"] > 0.0


def test_act_policy_prior_surfaces_the_same_hazard_surface() -> None:
    """ACT's preference prior reproduces the same normalized tag set."""
    contract = _shared_contract()
    parsed = parse_crescent_city_intel(source=contract)
    prior = hazard_policy_prior(parsed)

    assert prior["deterministic"] is True
    # ACT's prior emits qualified tags literally (hyphen preserved), while RISK
    # and BAYES fold them onto phrase boundaries. Compare on the canonical
    # normalized token so the semantic hazard surface is asserted to match.
    assert {_normalize_tag(tag) for tag in prior["hazardTags"]} == {
        "seismic",
        "tsunami",
        "erosion",
        "flood",
        "tsunami drill",
        "sea level",
    }
    assert prior["dominantHazard"] == "tsunami"
    assert len(prior["hazardTags"]) + 1 == len(prior["preferences"])


def test_shared_core_resolves_to_canonical_bayes_objects() -> None:
    """ACT and RISK re-export the canonical BAYES ingestion core by identity."""
    canonical = bayes_civic.load_crescent_city_contract

    assert act_civic.load_crescent_city_contract is canonical
    assert risk_civic.load_crescent_city_contract is canonical
    assert act_civic.SUPPORTED_SCHEMA is bayes_civic.CRESCENT_CITY_INTEL_SCHEMA
    assert (
        risk_civic.CRESCENT_CITY_GEO_INTEL_SCHEMA
        is bayes_civic.CRESCENT_CITY_INTEL_SCHEMA
    )
    assert (
        str(act_civic.SUPPORTED_SCHEMA)
        == str(risk_civic.CRESCENT_CITY_GEO_INTEL_SCHEMA)
        == "crescent-city-geo-intel/v1"
    )


def test_bundled_contract_ships_exactly_one_canonical_copy() -> None:
    """The reviewed crescent-city-geo-intel.json lives only in BAYES."""
    import importlib.resources
    import json

    bundled = importlib.resources.files("geo_infer_bayes").joinpath(
        "crescent-city-geo-intel.json"
    )
    assert bundled.is_file()
    assert json.loads(bundled.read_text(encoding="utf-8"))["schema"] == _SCHEMA
    assert not (
        importlib.resources.files("geo_infer_risk") / "crescent-city-geo-intel.json"
    ).is_file()
