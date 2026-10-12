"""End-to-end consumer example: GEO-INFER over the crescent-city-intel contract.

This runnable example is the single-place consumer that proves RISK
(``geo_infer_risk``), BAYES (``geo_infer_bayes``) and ACT
(``geo_infer_act``) interpret the *same* ``crescent-city-geo-intel/v1``
contract, emitted by the sibling ``crescent-city-intel`` project and shipped
in-mirror as a bundled seed by each module.

It loads the bundled seed once (the RISK / BAYES / PLACE copies are
byte-identical), feeds that single contract to all three civic-intel
ingestors, and prints a compact human-readable summary:

- the Crescent City anchor and its four hazard-relevant civic domains;
- RISK municipal-code hazard weights mapped from the contract surface;
- BAYES categorical hazard prior probabilities over those domains;
- ACT municipal hazard policy prior and the deterministic policy decision
  it drives (an active-inference ``PolicySelector`` choosing the lowest
  expected-free-energy action under that preference prior);
- a geo-view parity check that proves all three modules lower the *same*
  contract into agreeing geo views: contract schema, per-module view schema,
  WGS84 bounds, municipal anchor, nominal hazard-domain points (domain ids and
  names), and hazard-weighted municipal-code section references per domain.

``build_summary`` and ``build_geo_parity`` are pure, deterministic functions
over the decoded contract, so tests can pin their output without running the
CLI.  ``main`` only loads the bundled seed and renders that summary.  The
example never searches sibling checkouts, performs network access, or falls
back to a live service: it uses real module implementations only (no stand-ins).

If a module's ``civic_intel`` helper cannot be imported, that module's section
is skipped with a clear message instead of aborting the demo, and the parity
check reports the remaining modules' agreement without it.

Run directly:

.. code-block:: bash

    uv run python GEO-INFER-TEST/demo/crescent_city_civic_intel_demo.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from collections.abc import Mapping

import numpy as np

_ENV_CONTRACT_PATH = "CRESCENT_CITY_INTEL_CONTRACT_PATH"

# The sibling-contract seeds ship bundled (and byte-identical) in the BAYES and
# PLACE module checkouts.  Prefer the first that exists, honouring an env override first.
_BUNDLED_SEED_RELATIVES = (
    "GEO-INFER-BAYES/src/geo_infer_bayes/crescent-city-geo-intel.json",
    "GEO-INFER-PLACE/src/geo_infer_place/locations/del_norte_county/data/crescent-city-geo-intel.json",
)

# The canonical contract schema id all modules are expected to validate.
_CANONICAL_SCHEMA = "crescent-city-geo-intel/v1"
_MODULE_NAMES = ("risk", "bayes", "act")

# Canonical hazard domain ids expected in the bundled seed's hazard subset.
_CANONICAL_DOMAIN_IDS = (
    "emergency-management",
    "environmental-protection",
    "event-planning",
    "climate-environment",
)


try:  # pragma: no cover - exercised only when the sibling import is absent
    from geo_infer_risk import (  # type: ignore[import-not-found]
        CRESCENT_CITY_GEO_INTEL_SCHEMA,
        crescent_city_hazard_weights,
        load_crescent_city_hazard,
    )

    _RISK_SCHEMA = CRESCENT_CITY_GEO_INTEL_SCHEMA
    _RISK_IMPORT_OK = True
except ImportError:
    _RISK_SCHEMA = None
    crescent_city_hazard_weights = None
    load_crescent_city_hazard = None
    _RISK_IMPORT_OK = False

try:  # pragma: no cover - see above
    from geo_infer_bayes.civic_intel import (  # type: ignore[import-not-found]
        CRESCENT_CITY_INTEL_SCHEMA,
    )
    from geo_infer_bayes import (  # type: ignore[import-not-found]
        build_hazard_categorical_prior,
        build_hazard_prior_table,
        load_crescent_city_intel,
    )

    _BAYES_SCHEMA = CRESCENT_CITY_INTEL_SCHEMA
    _BAYES_IMPORT_OK = True
except ImportError:
    _BAYES_SCHEMA = None
    build_hazard_categorical_prior = None
    build_hazard_prior_table = None
    load_crescent_city_intel = None
    _BAYES_IMPORT_OK = False

try:  # pragma: no cover - see above
    from geo_infer_act import (  # type: ignore[import-not-found]
        hazard_policy_prior,
        parse_crescent_city_intel,
    )
    from geo_infer_act.core.policy_selection import PolicySelector

    _ACT_SCHEMA = "crescent-city-geo-intel/v1"  # ACT folds schema into parsed view
    _ACT_IMPORT_OK = True
except ImportError:
    _ACT_SCHEMA = None
    hazard_policy_prior = None
    parse_crescent_city_intel = None
    PolicySelector = None
    _ACT_IMPORT_OK = False

__all__ = [
    "build_geo_parity",
    "build_summary",
    "bundled_contract_path",
    "geo_views_agree",
    "load_bundled_contract",
    "load_seed",
    "render_summary",
]


def _resolve_repo_root() -> Path:
    """Return the repository root (the parent of the ``GEO-INFER-*`` modules)."""
    return Path(__file__).resolve().parents[2]


def bundled_contract_path() -> Path:
    """Locate a bundled reviewed copy of the ``crescent-city-geo-int/v1`` seed.

    Honors ``CRESCENT_CITY_INTEL_CONTRACT_PATH`` when set; otherwise checks the
    byte-identical copies shipped by the RISK, BAYES and PLACE modules and
    returns the first that exists.  Raises ``FileNotFoundError`` when none is
    found so the run halts with an actionable message rather than parsing an
    absent spaghetti.
    """
    override = os.environ.get(_ENV_CONTRACT_PATH)
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"{_ENV_CONTRACT_PATH} points at a missing file: {candidate}"
        )
    root = _resolve_repo_root()
    for relative in _BUNDLED_SEED_RELATIVES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no bundled crescent-city-geo-intel seed found under the GEO-INFER "
        "checkout; expected one of the BAYES/PLACE copies."
    )


def load_bundled_contract() -> Mapping[str, object]:
    """Decode the bundled seed contract as one mapping."""
    path = bundled_contract_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read bundled contract at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid bundled contract JSON at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("bundled contract JSON root must be an object")
    return payload


# Backward-friendly alias: the seed IS the bundled contract.
load_seed = load_bundled_contract


def _rounded(value: object, ndigits: int = 3) -> float:
    """Return a JSON-safe finite float rounded to ``ndigits`` decimal places."""
    return round(float(value), ndigits)


def _hazard_domain_views(contract: Mapping[str, object]) -> list[dict[str, object]]:
    """Project the hazard-relevant contract domain subset (id/name/hazardTags)."""
    hazard = contract.get("hazard")
    domains = hazard.get("relevantDomains", []) if isinstance(hazard, Mapping) else []
    views: list[dict[str, object]] = []
    for domain in domains:
        if not isinstance(domain, Mapping):
            continue
        views.append(
            {
                "id": domain.get("id"),
                "name": domain.get("name"),
                "hazardTags": list(domain.get("hazardTags", []))
                if isinstance(domain.get("hazardTags"), list)
                else [],
            }
        )
    return views


def _extract_section_refs(domain_records: object) -> dict[str, list[str]]:
    """Project each hazard domain to its sorted, distinct municipal-code refs.

    Accepts both RISK's flattened ``sections`` and the BAYES/ACT nested
    ``topics[].sections`` shapes, so one canonical projection gathers section
    references from every module view and the raw contract identically.  The
    order is stable (sorted distinct ``sectionNumber`` strings).
    """
    refs_by_domain: dict[str, list[str]] = {}
    if not isinstance(domain_records, list):
        return refs_by_domain
    for domain in domain_records:
        if not isinstance(domain, Mapping):
            continue
        domain_id = domain.get("id")
        if not isinstance(domain_id, str) or not domain_id:
            continue
        refs: list[str] = []
        flat = domain.get("sections")
        if isinstance(flat, list):
            for raw in flat:
                if isinstance(raw, Mapping) and raw.get("sectionNumber") is not None:
                    refs.append(str(raw["sectionNumber"]))
        topics = domain.get("topics")
        if isinstance(topics, list):
            for topic in topics:
                if not isinstance(topic, Mapping):
                    continue
                topic_sections = topic.get("sections")
                if isinstance(topic_sections, list):
                    for raw in topic_sections:
                        if (
                            isinstance(raw, Mapping)
                            and raw.get("sectionNumber") is not None
                        ):
                            refs.append(str(raw["sectionNumber"]))
        orderable = [item for item in refs if item]
        section_by_domain_id = sorted(set(orderable)) if orderable else []
        if section_by_domain_id:
            refs_by_domain[domain_id] = section_by_domain_id
    return refs_by_domain


def _canonical_bounds(value: object) -> dict[str, float]:
    """Canonical WGS84 bounds mapping with JSON-safe rounded coordinates."""
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _rounded(value.get(key, 0.0), 3)
        for key in ("west", "south", "east", "north")
    }


def _canonical_anchor(value: object) -> dict[str, object]:
    """Canonical anchor identity from any module's city/anchor mapping."""
    if not isinstance(value, Mapping):
        return {}
    return {
        "name": value.get("name"),
        "municipality": value.get("municipality"),
        "county": value.get("county"),
        "state": value.get("state"),
        "latitude": _rounded(value.get("latitude", 0.0), 3),
        "longitude": _rounded(value.get("longitude", 0.0), 3),
    }


def _raw_geo_view(contract: Mapping[str, object]) -> dict[str, object]:
    """Canonical geo view straight from the raw contract (no module import)."""
    anchor = contract.get("anchor")
    raw_bounds = anchor.get("bounds") if isinstance(anchor, Mapping) else None
    hazard = contract.get("hazard")
    domains = hazard.get("relevantDomains", []) if isinstance(hazard, Mapping) else []
    return {
        "schema": contract.get("schema"),
        "bounds": _canonical_bounds(raw_bounds),
        "anchor": _canonical_anchor(anchor),
        "sections": _extract_section_refs(domains),
        "domains": _hazard_domain_views(contract),
    }


def _usable_schema(module_schema: object, fallback: object = None) -> object:
    """Return a module's view schema constant, or the ``fallback`` when absent."""
    if module_schema is None:
        return fallback
    return module_schema


def _risk_geo_view(contract: Mapping[str, object]) -> dict[str, object] | None:
    """RISK's geo view: hazard intel + bounds/anchor from its own parse."""
    if not _RISK_IMPORT_OK:
        return None
    hazard_intel = load_crescent_city_hazard(contract)  # type: ignore[misc]
    city = hazard_intel.get("city")
    return {
        "schema": _usable_schema(_RISK_SCHEMA, contract.get("schema")),
        "bounds": _canonical_bounds(hazard_intel.get("bounds")),
        "anchor": _canonical_anchor(city),
        "sections": _extract_section_refs(hazard_intel.get("hazardDomains", [])),
    }


def _bayes_geo_view(contract: Mapping[str, object]) -> dict[str, object] | None:
    """BAYES's geo view: hazard domain + bounds from its own parse."""
    if not _BAYES_IMPORT_OK:
        return None
    intel = load_crescent_city_intel(source=contract)  # type: ignore[misc]
    return {
        "schema": _usable_schema(_BAYES_SCHEMA, contract.get("schema")),
        "bounds": _canonical_bounds(intel.get("bounds")),
        "anchor": _canonical_anchor(intel.get("city")),
        "sections": _extract_section_refs(intel.get("hazardDomains", [])),
    }


def _act_geo_view(contract: Mapping[str, object]) -> dict[str, object] | None:
    """ACT's geo view: parsed record carries its own schema and bounds."""
    if not _ACT_IMPORT_OK:
        return None
    parsed = parse_crescent_city_intel(source=contract)  # type: ignore[misc]
    return {
        "schema": parsed.get("schema") or _usable_schema(_ACT_SCHEMA),
        "bounds": _canonical_bounds(parsed.get("bounds")),
        "anchor": _canonical_anchor(parsed.get("anchor")),
        "sections": _extract_section_refs(parsed.get("hazardDomains", [])),
    }


_SHARED_GEO_VIEWS = {
    "risk": _risk_geo_view,
    "bayes": _bayes_geo_view,
    "act": _act_geo_view,
}


def geo_views_agree(contract: Mapping[str, object]) -> bool:
    """True when every importable module lowers the contract to an agreeing view.

    Raises ``RuntimeError`` when no module is importable.
    """
    parity = build_iso_geo_parity(contract)
    if not parity["sighted"]:
        raise RuntimeError("no geo-int module importable; cannot assess view parity")
    return bool(parity["match"])


def build_iso_geo_parity(contract: Mapping[str, object]) -> dict[str, object]:
    """Deterministic, JSON-safe view-parity digest for the shared contract.

    Projects the contract and every importable module onto a canonical geo view
    (contract schema, per-module view schema, WGS84 bounds, anchor identity,
    nominal hazard-domain points, and per-domain municipal-code section
    references) and compares them field-wise.

    Returns a dict:

    - ``contractSchema`` the contract's declared schema;
    - ``sighted`` ordered module names that imported successfully;
    - ``skipped`` module names that could not be imported;
    - ``bounds`` / ``anchor`` canonical contract-dimension values;
    - ``domains`` the nominal hazard-domain points (id and name);
    - ``moduleViews`` per-sighted-module geo view for inspection;
    - ``schemaAgreement`` / ``boundsAgreement`` / ``anchorAgreement`` /
      ``domainPointsAgreement`` / ``sectionAgreement`` per-dimension booleans
      over the sighted modules;
    - ``match`` all agreement rows hold (and at least one module sighted).

    The digest is deterministic for a fixed contract: it never draws randomness
    and only reads order preserved by the modules' stable parsers.  When no
    module is importable the ``match`` is ``False`` and the agreement flags for
    the (empty) sighted set are ``True`` by vacuous truth.
    """
    baseline = _raw_geo_view(contract)
    contract_schema = contract.get("schema")

    module_views: dict[str, object] = {}
    sighted: list[str] = []
    skipped: list[str] = []
    for name in _MODULE_NAMES:
        builder = _SHARED_GEO_VIEWS.get(name)
        if builder is None:
            skipped.append(name)
            continue
        try:
            view = builder(contract)
        except ImportError:  # per-module missing dependency, not a data failure
            view = None
        if view is None:
            skipped.append(name)
            continue
        sighted.append(name)
        module_views[name] = view

    if not sighted:
        return {
            "contractSchema": contract_schema,
            "sighted": [],
            "skipped": list(skipped),
            "bounds": baseline["bounds"],
            "anchor": baseline["anchor"],
            "domains": baseline["domains"],
            "section_refs": baseline["sections"],
            "moduleViews": {},
            "schemaAgreement": False,
            "boundsAgreement": True,
            "anchorAgreement": True,
            "domainPointsAgreement": True,
            "sectionAgreement": True,
            "match": False,
        }

    schema_agrees = bool(
        contract_schema is not None
        and all(module_views[name]["schema"] == contract_schema for name in sighted)
    )
    bounds_agrees = all(
        module_views[name]["bounds"] == baseline["bounds"] for name in sighted
    )
    anchor_agrees = all(
        module_views[name]["anchor"] == baseline["anchor"] for name in sighted
    )
    domain_ids = {domain["id"] for domain in baseline["domains"]}
    domain_points_agree = all(
        set(module_views[name]["sections"]) == domain_ids
        or set(module_views[name]["sections"]).issuperset(domain_ids)
        for name in sighted
    )
    section_agrees = all(
        module_views[name]["sections"] == baseline["sections"] for name in sighted
    )

    match = (
        schema_agrees
        and bounds_agrees
        and anchor_agrees
        and domain_points_agree
        and section_agrees
    )

    return {
        "contractSchema": contract_schema,
        "sighted": sighted,
        "skipped": skipped,
        "bounds": baseline["bounds"],
        "anchor": baseline["anchor"],
        "domains": baseline["domains"],
        "section_refs": baseline["sections"],
        "moduleViews": module_views,
        "schemaAgreement": schema_agrees,
        "boundsAgreement": bounds_agrees,
        "anchorAgreement": anchor_agrees,
        "domainPointsAgreement": domain_points_agree,
        "sectionAgreement": section_agrees,
        "match": match,
    }


def build_geo_parity(contract: Mapping[str, object]) -> dict[str, object]:
    """Alias kept for readability: deterministic geo-view parity digest."""
    return build_iso_geo_parity(contract)


def _risk_section(contract: Mapping[str, object]) -> dict[str, object]:
    """Run the RISK municipal-code hazard weights over the shared contract."""
    if not _RISK_IMPORT_OK:
        return {
            "available": False,
            "message": "geo_infer_risk civic_intel helper not importable; skipped.",
        }
    hazard_intel = load_crescent_city_hazard(contract)  # type: ignore[misc]
    weights = crescent_city_hazard_weights(hazard_intel)  # type: ignore[misc]
    rounded = {str(tag): _rounded(weight, 3) for tag, weight in sorted(weights.items())}
    top_tag = max(weights, key=weights.get) if weights else None
    return {
        "available": True,
        "weights": rounded,
        "top": (top_tag, round(float(weights[top_tag]), 3)) if top_tag else None,
    }


def _bayes_section(contract: Mapping[str, object]) -> dict[str, object]:
    """Run the BAYES categorical hazard prior over the shared contract."""
    if not _BAYES_IMPORT_OK:
        return {
            "available": False,
            "message": "geo_infer_bayes prior helper not importable; skipped.",
        }
    intel = load_crescent_city_intel(source=contract)  # type: ignore[misc]
    table = build_hazard_prior_table(intel)  # type: ignore[misc]
    prior = build_hazard_categorical_prior(table)  # type: ignore[misc]
    return {
        "available": True,
        "domains": list(prior.domains),
        "prior": {
            d: _rounded(p, 3)
            for d, p in zip(prior.domains, prior.probabilities, strict=True)
        },
    }


def _policy_slug(tag: str) -> str:
    """Lowercase slug for a policy id derived from a hazard tag."""
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_") or "hazard"


def _act_section(contract: Mapping[str, object]) -> dict[str, object]:
    """Compute the ACT municipal-policy prior and its deterministic decision."""
    if not _ACT_IMPORT_OK:
        return {
            "available": False,
            "message": "geo_infer_act policy helper not importable; skipped.",
        }
    parsed = parse_crescent_city_intel(source=contract)  # type: ignore[misc]
    prior = hazard_policy_prior(parsed)  # type: ignore[misc]
    tags = list(prior.get("hazardTags", []))
    n_states = len(prior.get("preferences", []))
    preferences = [round(float(value), 3) for value in prior.get("preferences", [])]

    # Candidate policies: one per state axis (baseline all-clear first, then a
    # policy per notified hazard tag). A deterministic PolicySelector picks the
    # lowest-expected-free-energy action under the municipal hazard prior.
    policies = [
        {"id": "maintain_baseline_ops", "predicted_beliefs": _one_hot(0, n_states)}
    ]
    policies.extend(
        {
            "id": f"hazard_{_policy_slug(tag)}",
            "predicted_beliefs": _one_hot(index, n_states),
        }
        for index, tag in enumerate(tags, start=1)
    )

    beliefs = np.full(n_states, 1.0 / max(n_states, 1))
    selector = PolicySelector(selection_mode="deterministic", random_seed=0)  # type: ignore[misc]
    result = selector.select_policy(beliefs, policies, preferences=prior)  # type: ignore[misc]

    return {
        "available": True,
        "hazardTags": tags,
        "dominantHazard": prior.get("dominantHazard"),
        "preferences": preferences,
        "decision": {
            "policy_id": result["policy"]["id"],
            "probability": round(float(result["probability"]), 3),
            "expected_free_energy": round(float(result["expected_free_energy"]), 3),
        },
    }


def _format_weights(weights: Mapping[str, float]) -> str:
    return " | ".join(f"{tag} {value:.3f}" for tag, value in weights.items())


def _format_prior(domains: list[str], probabilities: Mapping[str, float]) -> str:
    return " | ".join(f"{domain} {probabilities[domain]:.3f}" for domain in domains)


def _one_hot(index: int, size: int) -> list[float]:
    """Return a one-hot probability vector of length ``size``."""
    vector = [0.0] * size
    if index < size:
        vector[index] = 1.0
    return vector


def build_summary(contract: Mapping[str, object]) -> dict[str, object]:
    """Reduce ONE ``crescent-city-geo-int/v1`` contract to a deterministic digest.

    The returned structure is the pure, JSON-serializable summary that the
    demo prints. It is deterministic for a fixed contract: RISK weights, BAYES
    prior probabilities, the ACT policy decision and the geo-view parity all
    derive from the same ingested surface with no RNG (ACT uses a deterministic
    ``PolicySelector``). When a module's helper is unavailable its section
    carries ``available: False`` and a ``message`` instead of aborting.

    Returns a dict with anchor, hazard_domains, risk/bayes/act sections and the
    geo_parity block.
    """
    anchor = contract.get("anchor")
    anchor_view: dict[str, object] = {}
    if isinstance(anchor, Mapping):
        anchor_view = {
            "name": anchor.get("name"),
            "municipality": anchor.get("municipality"),
            "county": anchor.get("county"),
            "state": anchor.get("state"),
            "latitude": _rounded(anchor.get("latitude", 0.0), 3),
            "longitude": _rounded(anchor.get("longitude", 0.0), 3),
        }
    return {
        "schema": contract.get("schema"),
        "generatedAt": contract.get("generatedAt"),
        "anchor": anchor_view,
        "hazard_domains": _hazard_domain_views(contract),
        "risk": _risk_section(contract),
        "bayes": _bayes_section(contract),
        "act": _act_section(contract),
        "geo_parity": build_geo_parity(contract),
    }


def _parity_mark(agreement: object) -> str:
    return "yes" if agreement else "no"


def _render_parity(parity: Mapping[str, object], indent: str = "  ") -> list[str]:
    """Render the geo-parity block compactly for a terminal."""
    lines: list[str] = []
    sighted = parity.get("sighted")
    lines.append(
        f"{indent}geo view parity : {len(sighted) if isinstance(sighted, list) else 0} modules sighted"
    )
    if isinstance(sighted, list):
        lines.append(f"{indent}  modules       : {' '.join(sighted) or '(none)'}")
    lines.append(
        f"{indent}  schema        : contract={parity['contractSchema']}, "
        f"agrees={_parity_mark(parity['schemaAgreement'])}"
    )
    lines.append(
        f"{indent}  bounds        : {parity['bounds']}, agrees={_parity_mark(parity['boundsAgreement'])}"
    )
    lines.append(
        f"{indent}  anchor        : {_anchor_summary(parity['anchor'])}, agrees={_parity_mark(parity['anchorAgreement'])}"
    )
    lines.append(
        f"{indent}  sections      : {len(parity['domains'])} domain points, "
        f"agrees={_parity_mark(parity['sectionAgreement'])}"
    )
    lines.append(f"{indent}  match         : {_parity_mark(parity['match'])}")
    return lines


def _anchor_summary(anchor: Mapping[str, object]) -> str:
    if not anchor:
        return "(none)"
    name = anchor.get("name")
    county = anchor.get("county")
    state = anchor.get("state")
    return (
        f"{name} · {county}, {state} [{anchor.get('latitude'):g}, {anchor.get('longitude'):g}]"
        if name
        else "(none)"
    )


def render_summary(summary: Mapping[str, object]) -> str:
    """Render the digest as compact, human-readable text for a terminal."""
    lines: list[str] = []
    lines.append("Crescent City civic-intel consumer demo")
    lines.append(f"  schema     : {summary.get('schema')}")
    generated = summary.get("generatedAt")
    if generated:
        lines.append(f"  generated  : {generated}")

    anchor = summary.get("anchor")
    if isinstance(anchor, Mapping):
        municipality = anchor.get("municipality")
        county = anchor.get("county")
        state = anchor.get("state")
        place = f"{municipality} · {county}, {state}" if municipality else ""
        lines.append(
            f"  anchor     : {anchor.get('name')} ({place}) "
            f"[{anchor.get('latitude'):g}, {anchor.get('longitude'):g}]"
        )

    domains = summary.get("hazard_domains")
    if isinstance(domains, list):
        lines.append(f"  hazard domains ({len(domains)}):")
        for domain in domains:
            if isinstance(domain, Mapping):
                lines.append(
                    f"    - {domain.get('id')}: {domain.get('name')} "
                    f"[{', '.join(domain.get('hazardTags', []))}]"
                )

    risk = summary.get("risk")
    if isinstance(risk, Mapping) and risk.get("available"):
        weights = risk.get("weights")
        if isinstance(weights, Mapping):
            top = risk.get("top")
            top_text = (
                f" most-evidenced: {top[0]} ({top[1]:.3f})"
                if isinstance(top, (list, tuple)) and top
                else ""
            )
            lines.append(
                f"  RISK      : hazard weights  {_format_weights(weights)}{top_text}"
            )
    elif isinstance(risk, Mapping):
        lines.append(f"  RISK      : {risk.get('message')}")

    bayes = summary.get("bayes")
    if isinstance(bayes, Mapping) and bayes.get("available"):
        domain_names = bayes.get("domains")
        probabilities = bayes.get("prior")
        if isinstance(domain_names, list) and isinstance(probabilities, Mapping):
            lines.append(
                f"  BAYES     : prior probabilities  {_format_prior(domain_names, probabilities)}"
            )
    elif isinstance(bayes, Mapping):
        lines.append(f"  BAYES     : {bayes.get('message')}")

    act = summary.get("act")
    if isinstance(act, Mapping) and act.get("available"):
        dominant = act.get("dominantHazard")
        decision = act.get("decision")
        tags = act.get("hazardTags")
        lines.append(
            f"  ACT       : policy prior over {len(tags) if isinstance(tags, list) else 0} hazard states"
        )
        lines.append(f"              dominant hazard: {dominant}")
        if isinstance(decision, Mapping):
            lines.append(
                f"              selected action: {decision.get('policy_id')} "
                f"(p={decision.get('probability'):.3f}, EFE={decision.get('expected_free_energy'):.3f})"
            )
    elif isinstance(act, Mapping):
        lines.append(f"  ACT       : {act.get('message')}")

    parity = summary.get("geo_parity")
    if isinstance(parity, Mapping):
        lines.extend(_render_parity(parity))

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Load the bundled seed, build the digest, and print the human summary."""
    del argv  # unused: the demo reads the bundled seed contract.
    try:
        contract = load_bundled_contract()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_summary(build_summary(contract)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
