"""Portfolio tracking and lightweight optimization for underwriting workflows.

The underwriting engine needs a small, dependency-light portfolio boundary so
approved policies and processed claims can be summarized without requiring a
database or a separate analytics service.  Records are kept in memory by
default; callers can provide their own persistence around the public methods.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, cast
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return dataclass-like or mapping values as a read-only mapping."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        return cast(Mapping[str, Any], value.to_dict())
    if hasattr(value, "__dict__"):
        return cast(Mapping[str, Any], vars(value))
    raise TypeError("portfolio records must be mappings or dataclass-like objects")


class PortfolioManager:
    """Track policy exposure, premium, and claim metrics by portfolio."""

    def __init__(self, config: Any | None = None):
        self.config = config or {}
        self.policies: dict[str, dict[str, Any]] = {}
        self.claims: list[dict[str, Any]] = []
        self._policy_portfolios: dict[str, str] = {}

    def add_policy(self, policy: Any, portfolio_id: str = "default") -> str:
        """Add or replace a policy in a named portfolio.

        ``policy`` may be a policy model, a dataclass, or a mapping.  The
        normalized record keeps only scalar portfolio fields and preserves the
        original record under ``record`` for downstream reporting.
        """
        record = dict(_as_mapping(policy))
        policy_id = str(record.get("policy_id") or record.get("id") or "")
        if not policy_id:
            raise ValueError("policy must include a policy_id or id")

        portfolio_id = str(portfolio_id or "default")
        normalized = {
            "policy_id": policy_id,
            "portfolio_id": portfolio_id,
            "status": getattr(
                record.get("status"), "value", record.get("status", "unknown")
            ),
            "premium": float(
                record.get("total_premium", record.get("premium", 0.0)) or 0.0
            ),
            "exposure": float(
                record.get(
                    "total_exposure",
                    record.get("limit", record.get("insured_value", 0.0)),
                )
                or 0.0
            ),
            "risk_score": float(record.get("risk_score", 0.0) or 0.0),
            "region": str(record.get("region", record.get("territory", "unknown"))),
            "record": record,
            "updated_at": datetime.now().isoformat(),
        }
        self.policies[policy_id] = normalized
        self._policy_portfolios[policy_id] = portfolio_id
        return policy_id

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy and return whether it existed."""
        existed = policy_id in self.policies
        self.policies.pop(policy_id, None)
        self._policy_portfolios.pop(policy_id, None)
        return existed

    def update_portfolio_metrics(self, claim: Any) -> dict[str, Any]:
        """Record a claim and update its associated policy loss metrics."""
        record = dict(_as_mapping(claim))
        claim_id = str(record.get("claim_id") or record.get("id") or len(self.claims))
        amount = float(
            record.get(
                "settlement_amount", record.get("total_paid", record.get("amount", 0.0))
            )
            or 0.0
        )
        normalized = {
            "claim_id": claim_id,
            "policy_id": str(record.get("policy_id", "")),
            "amount": max(0.0, amount),
            "status": getattr(
                record.get("status"), "value", record.get("status", "unknown")
            ),
            "timestamp": datetime.now().isoformat(),
            "record": record,
        }
        self.claims.append(normalized)
        return normalized

    def get_portfolio_summary(self, portfolio_id: str | None = None) -> dict[str, Any]:
        """Return deterministic aggregate portfolio metrics."""
        records = [
            policy
            for policy in self.policies.values()
            if portfolio_id is None or policy["portfolio_id"] == str(portfolio_id)
        ]
        policy_ids = {policy["policy_id"] for policy in records}
        claims = [claim for claim in self.claims if claim["policy_id"] in policy_ids]
        total_premium = sum(policy["premium"] for policy in records)
        total_exposure = sum(policy["exposure"] for policy in records)
        total_claims = sum(claim["amount"] for claim in claims)

        status_counts: dict[str, int] = defaultdict(int)
        region_exposure: dict[str, float] = defaultdict(float)
        for policy in records:
            status_counts[policy["status"]] += 1
            region_exposure[policy["region"]] += policy["exposure"]

        return {
            "portfolio_id": portfolio_id,
            "total_policies": len(records),
            "active_policies": sum(
                policy["status"] in {"active", "bound", "quoted"} for policy in records
            ),
            "total_premium": total_premium,
            "total_exposure": total_exposure,
            "total_claims": total_claims,
            "loss_ratio": total_claims / total_premium if total_premium > 0 else 0.0,
            "average_risk_score": (
                sum(policy["risk_score"] for policy in records) / len(records)
                if records
                else 0.0
            ),
            "policies_by_status": dict(status_counts),
            "exposure_by_region": dict(region_exposure),
            "generated_at": datetime.now().isoformat(),
        }


class PortfolioOptimizer:
    """Provide deterministic capacity and concentration recommendations."""

    def __init__(self, config: Any | None = None):
        self.config = config or {}

    def optimize(
        self, policies: Iterable[Any], max_exposure: float | None = None
    ) -> dict[str, Any]:
        """Rank policies by risk-adjusted premium and flag capacity breaches."""
        records = [dict(_as_mapping(policy)) for policy in policies]
        normalized: list[dict[str, Any]] = [
            {
                "policy_id": str(record.get("policy_id", record.get("id", index))),
                "risk_score": float(record.get("risk_score", 0.0) or 0.0),
                "premium": float(
                    record.get("total_premium", record.get("premium", 0.0)) or 0.0
                ),
                "exposure": float(
                    record.get("total_exposure", record.get("limit", 0.0)) or 0.0
                ),
            }
            for index, record in enumerate(records)
        ]
        ranked = sorted(
            normalized,
            key=lambda record: (
                record["premium"] / max(record["exposure"], 1.0),
                -record["risk_score"],
            ),
            reverse=True,
        )
        total_exposure = sum(record["exposure"] for record in ranked)
        return {
            "recommended_order": ranked,
            "total_exposure": total_exposure,
            "capacity_limit": max_exposure,
            "capacity_exceeded": max_exposure is not None
            and total_exposure > max_exposure,
        }
