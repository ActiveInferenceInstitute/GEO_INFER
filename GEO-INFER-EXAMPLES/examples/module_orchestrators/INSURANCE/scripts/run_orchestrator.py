#!/usr/bin/env python3
"""GEO-INFER-INSURANCE module orchestrator.

Runs one documented end-to-end INSURANCE operation on synthetic data: score
the risk of an insured property, underwrite a property application through
the exported convenience API — ``assess_risk`` and ``underwrite_policy``
combined with ``calculate_premium`` — then file and process a claim via
``process_claim``. All work goes through the real ``geo_infer_insurance``
public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_insurance import (
        assess_risk,
        calculate_premium,
        process_claim,
        underwrite_policy,
    )

    application = {
        "property": {
            "address": "1420 Cedar Lane, Coastal County",
            "property_value": 750_000.0,
            "year_built": 1998,
            "construction_type": "masonry",
            "occupancy": "primary_residence",
        },
        "applicant": {
            "name": "Jordan Rivera",
            "contact_info": {
                "email": "jordan.rivera@example.com",
                "phone": "555-0100",
            },
            "credit_score": 760,
        },
        "coverage_requests": [{"coverage_type": "fire", "limit": 600_000.0}],
    }

    risk = assess_risk(
        {"entity_type": "property", "value": 750_000.0, "location": "coastal"}
    )
    case = underwrite_policy(application)
    premium_calc = calculate_premium(
        application, case.risk_assessment or {}, case.rule_evaluation or {}
    )
    premium = (
        premium_calc.to_dict() if hasattr(premium_calc, "to_dict") else premium_calc
    )
    claim = process_claim(
        {
            "policy_id": f"POL-{case.case_id}",
            "date_of_loss": "2026-08-01",
            "claimed_amount": 12_000.0,
            "description": "Wind damage to roof",
        }
    )

    decision = case.decision
    return {
        "operation": "risk_scored_underwriting_with_claim_intake",
        "risk_assessment": {
            "risk_score": round(float(risk["risk_score"]), 6),
            "risk_level": str(risk["risk_level"]),
            "assessment_method": str(risk["assessment_method"]),
            "confidence": round(float(risk["confidence"]), 6),
        },
        "underwriting": {
            "case_id": case.case_id,
            "status": case.status,
            "approved": bool(decision.approved),
            "reason": decision.reason,
            "risk_score": round(float(decision.risk_score), 6),
            "confidence": round(float(decision.confidence), 6),
        },
        "premium": premium,
        "claims": {
            "claim_id": str(claim.claim_id),
            "status": str(claim.status),
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("INSURANCE", _operation))
