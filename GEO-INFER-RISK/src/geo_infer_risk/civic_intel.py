"""Crescent City civic-intelligence ingestion for hazard risk analysis.

The parser in this module consumes the frozen ``crescent-city-geo-intel/v1``
contract emitted by the sibling ``crescent-city-intel`` project.  The
canonical reviewed copy of that gold contract ships with GEO-INFER-BAYES and
is resolved here through a guarded shared import (this module keeps working
with an injected mapping or explicit local JSON path when the sibling is
absent).  It performs no discovery, network access, or live-data fallback.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, TypedDict

# Shared civic-intel ingestion core: the canonical contract loader and schema
# constant live in GEO-INFER-BAYES, so every consumer resolves the SAME
# objects and reads the ONE reviewed crescent-city-geo-intel.json copy.
# The import is guarded: when the sibling is absent this module still imports
# and its module-specific hazard-weight surface keeps working (cross-module
# identity tests pin object identity when the sibling IS importable).
try:
    from geo_infer_bayes.civic_intel import (
        CRESCENT_CITY_INTEL_SCHEMA,
        load_crescent_city_contract,
    )
    from geo_infer_bayes.geo_observations import (
        CRESCENT_CITY_OBSERVATIONS_SCHEMA,
        load_crescent_city_geo_observations,
    )
except ImportError:  # pragma: no cover - sibling-absent degradation path
    CRESCENT_CITY_INTEL_SCHEMA = "crescent-city-geo-intel/v1"
    CRESCENT_CITY_OBSERVATIONS_SCHEMA = "crescent-city-geo-observations/v1"

    def load_crescent_city_contract():
        raise ImportError(
            "load_crescent_city_contract requires geo-infer-bayes; the canonical"
            " crescent-city-geo-intel.json copy ships with geo-infer-bayes"
        )

    def load_crescent_city_geo_observations(source=None):
        raise ImportError(
            "load_crescent_city_geo_observations requires geo-infer-bayes; the"
            " canonical crescent-city-geo-observations.json copy ships with"
            " geo-infer-bayes"
        )


CRESCENT_CITY_GEO_INTEL_SCHEMA = CRESCENT_CITY_INTEL_SCHEMA


class CrescentCityBounds(TypedDict):
    """WGS84 bounding box for the Crescent City civic-intel anchor."""

    west: float
    south: float
    east: float
    north: float


class CrescentCityAnchor(TypedDict):
    """Municipal identity and WGS84 geometry from the contract anchor."""

    name: str
    guid: str
    municipality: str
    county: str
    state: str
    latitude: float
    longitude: float
    bounds: CrescentCityBounds


class MunicipalCodeSection(TypedDict):
    """One municipal-code reference attached to a hazard topic."""

    sectionNumber: str
    relevance: str


class CivicHazardDomain(TypedDict):
    """Hazard-relevant civic domain projected for risk consumers."""

    id: str
    name: str
    hazardTags: list[str]
    sections: list[MunicipalCodeSection]


class CrescentCityHazardIntel(TypedDict):
    """Normalized Crescent City hazard-policy surface."""

    city: CrescentCityAnchor | None
    hazardDomains: list[CivicHazardDomain]
    bounds: CrescentCityBounds | None


CrescentCitySeed: TypeAlias = Mapping[str, object] | str | os.PathLike[str] | None


def _empty_hazard_intel() -> CrescentCityHazardIntel:
    """Return a new empty result for an absent local seed."""

    return {"city": None, "hazardDomains": [], "bounds": None}


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    """Validate and return one contract object."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _as_list(value: object, field: str) -> list[object]:
    """Validate and return one contract array."""

    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _required_string(container: Mapping[str, object], key: str, field: str) -> str:
    """Read a required non-empty string from a contract object."""

    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}.{key} must be a non-empty string")
    return value.strip()


def _required_number(container: Mapping[str, object], key: str, field: str) -> float:
    """Read a required finite JSON number from a contract object."""

    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field}.{key} must be a finite number")
    return number


def _required_strings(
    container: Mapping[str, object], key: str, field: str
) -> list[str]:
    """Read an ordered, de-duplicated array of non-empty strings."""

    values = _as_list(container.get(key), f"{field}.{key}")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}.{key}[{index}] must be a non-empty string")
        normalized = value.strip()
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _parse_bounds(anchor: Mapping[str, object]) -> CrescentCityBounds:
    """Validate the WGS84 bounds embedded in the city anchor."""

    raw_bounds = _as_mapping(anchor.get("bounds"), "anchor.bounds")
    bounds = CrescentCityBounds(
        west=_required_number(raw_bounds, "west", "anchor.bounds"),
        south=_required_number(raw_bounds, "south", "anchor.bounds"),
        east=_required_number(raw_bounds, "east", "anchor.bounds"),
        north=_required_number(raw_bounds, "north", "anchor.bounds"),
    )
    if not -180.0 <= bounds["west"] < bounds["east"] <= 180.0:
        raise ValueError("anchor.bounds must have ordered WGS84 longitudes")
    if not -90.0 <= bounds["south"] < bounds["north"] <= 90.0:
        raise ValueError("anchor.bounds must have ordered WGS84 latitudes")
    return bounds


def _parse_anchor(
    contract: Mapping[str, object],
) -> tuple[CrescentCityAnchor, CrescentCityBounds]:
    """Validate and project the Crescent City anchor."""

    raw_anchor = _as_mapping(contract.get("anchor"), "anchor")
    bounds = _parse_bounds(raw_anchor)
    anchor = CrescentCityAnchor(
        name=_required_string(raw_anchor, "name", "anchor"),
        guid=_required_string(raw_anchor, "guid", "anchor"),
        municipality=_required_string(raw_anchor, "municipality", "anchor"),
        county=_required_string(raw_anchor, "county", "anchor"),
        state=_required_string(raw_anchor, "state", "anchor"),
        latitude=_required_number(raw_anchor, "latitude", "anchor"),
        longitude=_required_number(raw_anchor, "longitude", "anchor"),
        bounds=bounds,
    )
    if anchor["name"] != "Crescent City":
        raise ValueError("anchor.name must be 'Crescent City' for the v1 contract")
    if not -90.0 <= anchor["latitude"] <= 90.0:
        raise ValueError("anchor.latitude must be a WGS84 latitude")
    if not -180.0 <= anchor["longitude"] <= 180.0:
        raise ValueError("anchor.longitude must be a WGS84 longitude")
    if not bounds["south"] <= anchor["latitude"] <= bounds["north"]:
        raise ValueError("anchor.latitude must fall within anchor.bounds")
    if not bounds["west"] <= anchor["longitude"] <= bounds["east"]:
        raise ValueError("anchor.longitude must fall within anchor.bounds")
    return anchor, bounds


def _parse_sections(
    domain: Mapping[str, object],
    domain_index: int,
    hazard_tags: Sequence[str],
) -> list[MunicipalCodeSection]:
    """Flatten and de-duplicate municipal-code refs from hazard topics."""

    field = f"hazard.relevantDomains[{domain_index}]"
    topics = _as_list(domain.get("topics"), f"{field}.topics")
    if not topics:
        raise ValueError(f"{field}.topics must not be empty")
    sections: list[MunicipalCodeSection] = []
    seen: set[tuple[str, str]] = set()
    domain_tags = set(hazard_tags)
    for topic_index, raw_topic in enumerate(topics):
        topic_field = f"{field}.topics[{topic_index}]"
        topic = _as_mapping(raw_topic, topic_field)
        _required_string(topic, "name", topic_field)
        topic_tags = _required_strings(topic, "tags", topic_field)
        if not topic_tags:
            raise ValueError(f"{topic_field}.tags must not be empty")
        if not set(topic_tags).issubset(domain_tags):
            raise ValueError(f"{topic_field}.tags must be listed in {field}.hazardTags")
        raw_sections = _as_list(topic.get("sections"), f"{topic_field}.sections")
        for section_index, raw_section in enumerate(raw_sections):
            section_field = f"{topic_field}.sections[{section_index}]"
            section = _as_mapping(raw_section, section_field)
            section_number = _required_string(section, "sectionNumber", section_field)
            relevance = _required_string(section, "relevance", section_field)
            identity = (section_number, relevance)
            if identity in seen:
                continue
            seen.add(identity)
            sections.append(
                MunicipalCodeSection(
                    sectionNumber=section_number,
                    relevance=relevance,
                )
            )
    if not sections:
        raise ValueError(f"{field} must reference at least one municipal-code section")
    return sections


def _parse_hazard_domains(
    contract: Mapping[str, object],
) -> list[CivicHazardDomain]:
    """Validate and project the contract's hazard-relevant domain subset."""

    hazard = _as_mapping(contract.get("hazard"), "hazard")
    raw_domains = _as_list(hazard.get("relevantDomains"), "hazard.relevantDomains")
    declared_count = hazard.get("relevantDomainCount")
    if declared_count is not None:
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != len(raw_domains)
        ):
            raise ValueError(
                "hazard.relevantDomainCount must match hazard.relevantDomains"
            )

    domains: list[CivicHazardDomain] = []
    seen_ids: set[str] = set()
    for index, raw_domain in enumerate(raw_domains):
        field = f"hazard.relevantDomains[{index}]"
        domain = _as_mapping(raw_domain, field)
        domain_id = _required_string(domain, "id", field)
        if domain_id in seen_ids:
            raise ValueError("hazard.relevantDomains ids must be unique")
        seen_ids.add(domain_id)
        hazard_tags = _required_strings(domain, "hazardTags", field)
        if not hazard_tags:
            raise ValueError(f"{field}.hazardTags must not be empty")
        domains.append(
            CivicHazardDomain(
                id=domain_id,
                name=_required_string(domain, "name", field),
                hazardTags=hazard_tags,
                sections=_parse_sections(domain, index, hazard_tags),
            )
        )
    return domains


def parse_crescent_city_hazard(
    contract: Mapping[str, object],
) -> CrescentCityHazardIntel:
    """Purely parse one ``crescent-city-geo-intel/v1`` mapping.

    Parameters
    ----------
    contract:
        Already-decoded JSON mapping.  Malformed or version-mismatched inputs
        fail closed with ``ValueError``.

    Returns
    -------
    CrescentCityHazardIntel
        City point, Del Norte bounds, and flattened municipal-code references
        for each hazard-relevant civic domain.
    """

    schema = contract.get("schema")
    if schema != CRESCENT_CITY_GEO_INTEL_SCHEMA:
        raise ValueError(
            f"schema must be {CRESCENT_CITY_GEO_INTEL_SCHEMA!r}; received {schema!r}"
        )
    city, bounds = _parse_anchor(contract)
    return {
        "city": city,
        "hazardDomains": _parse_hazard_domains(contract),
        "bounds": bounds,
    }


def load_crescent_city_hazard(
    source: CrescentCitySeed = None,
) -> CrescentCityHazardIntel:
    """Load Crescent City hazard policy from an injected mapping or JSON path.

    ``source`` accepts an injected mapping, an explicit local JSON path, or
    ``None`` to load the reviewed package seed.  A missing package seed or
    explicit path returns an empty result.  Existing but malformed files fail closed.  The loader
    never searches sibling projects, downloads data, or falls back to a live
    service.
    """

    if isinstance(source, Mapping):
        return parse_crescent_city_hazard(source)
    if source is None:
        # The reviewed package seed is the canonical bundled contract copy
        # owned by GEO-INFER-BAYES; RISK no longer ships its own duplicate.
        contract = load_crescent_city_contract()
        if contract is None:
            return _empty_hazard_intel()
        return parse_crescent_city_hazard(contract)
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError("source must be a mapping, local JSON path, or None")

    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_hazard_intel()
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid Crescent City geo-intel JSON at {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"unable to read Crescent City geo-intel JSON at {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Crescent City geo-intel JSON root must be an object")
    return parse_crescent_city_hazard(payload)


def _normalized_hazard_name(value: str) -> str:
    """Normalize a hazard name for policy-evidence matching."""

    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def _hazard_lookup_names(value: str) -> tuple[str, ...]:
    """Return contract-tag aliases for one RISK hazard name."""

    normalized = _normalized_hazard_name(value)
    if normalized == "earthquake":
        return ("earthquake", "seismic")
    if normalized == "seismic":
        return ("seismic", "earthquake")
    return (normalized,)


def _hazard_tag_matches(tag: str, hazard: str) -> bool:
    """Match a RISK hazard to an explicit civic tag on phrase boundaries."""

    padded_tag = f" {_normalized_hazard_name(tag)} "
    return any(f" {alias} " in padded_tag for alias in _hazard_lookup_names(hazard))


def crescent_city_hazard_weights(
    hazard_intel: CrescentCityHazardIntel,
    hazard_types: Sequence[str] | None = None,
    *,
    default_weight: float = 0.0,
) -> dict[str, float]:
    """Map municipal policy evidence to normalized RISK hazard weights.

    Each contract hazard tag receives the number of unique referenced municipal
    code sections, divided by the largest such count across the city surface.
    Thus the strongest documented tag has weight ``1.0`` and other documented
    tags are proportional.  When matrix hazards are requested, sections from
    explicit qualified tags such as ``flood zone`` and ``tsunami drill`` are
    pooled on whole-phrase boundaries before normalization.  Tags with no
    section evidence receive ``default_weight`` (``0.0`` by default).

    Pass ``MultiHazardInteractionMatrix.hazard_types`` as ``hazard_types`` to
    align keys and ordering with an existing matrix.  ``earthquake`` is matched
    to the contract's ``seismic`` tag.  Qualified tags never use substring
    inference (for example, ``storm`` does not match ``stormwater``).  These are
    policy-evidence weights, not directed causal interaction strengths, so this
    helper does not mutate the matrix.
    """

    if isinstance(default_weight, bool):
        raise ValueError("default_weight must be finite and between 0 and 1")
    numeric_default = float(default_weight)
    if not math.isfinite(numeric_default) or not 0.0 <= numeric_default <= 1.0:
        raise ValueError("default_weight must be finite and between 0 and 1")

    sections_by_tag: dict[str, set[str]] = {}
    for domain in hazard_intel["hazardDomains"]:
        section_numbers = {section["sectionNumber"] for section in domain["sections"]}
        for raw_tag in domain["hazardTags"]:
            tag = _normalized_hazard_name(raw_tag)
            sections_by_tag.setdefault(tag, set()).update(section_numbers)

    maximum_count = max(
        (len(section_numbers) for section_numbers in sections_by_tag.values()),
        default=0,
    )
    if maximum_count:
        evidence_weights = {
            tag: len(section_numbers) / maximum_count
            for tag, section_numbers in sections_by_tag.items()
        }
    else:
        evidence_weights = {tag: numeric_default for tag in sections_by_tag}

    if hazard_types is None:
        return {tag: evidence_weights[tag] for tag in sorted(evidence_weights)}
    if isinstance(hazard_types, (str, bytes)):
        raise TypeError("hazard_types must be a sequence of hazard names")

    requested = list(hazard_types)
    if any(not isinstance(hazard, str) or not hazard.strip() for hazard in requested):
        raise ValueError("hazard_types must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("hazard_types must be unique")

    matched_sections: dict[str, set[str]] = {}
    for hazard in requested:
        matched_sections[hazard] = set().union(
            *(
                section_numbers
                for tag, section_numbers in sections_by_tag.items()
                if _hazard_tag_matches(tag, hazard)
            )
        )
    normalization_count = max(
        maximum_count,
        max(
            (len(section_numbers) for section_numbers in matched_sections.values()),
            default=0,
        ),
    )
    return {
        hazard: (
            len(section_numbers) / normalization_count
            if section_numbers and normalization_count
            else numeric_default
        )
        for hazard, section_numbers in matched_sections.items()
    }


__all__ = [
    "CRESCENT_CITY_GEO_INTEL_SCHEMA",
    "CivicHazardDomain",
    "CrescentCityAnchor",
    "CrescentCityBounds",
    "CrescentCityHazardIntel",
    "MunicipalCodeSection",
    "crescent_city_hazard_weights",
    "load_crescent_city_hazard",
    "parse_crescent_city_hazard",
]
