"""Talent Acquisition Reporting functions."""

import logging
from typing import Any

from ..models.talent_models import (
    Candidate,
    JobRequisition,
    CandidateStatus,
    JobRequisitionStatus,
)
from ..talent.transformer import (
    convert_candidates_to_dataframe,
    convert_requisitions_to_dataframe,
)

logger = logging.getLogger(__name__)


def generate_candidate_pipeline_report(
    candidates: list[Candidate], requisitions: list[JobRequisition] | None = None
) -> dict[str, Any]:
    """
    Generates a report on the current candidate pipeline status.
    """
    if not candidates:
        return {"message": "No candidate data for pipeline report."}

    cand_df = convert_candidates_to_dataframe(candidates)
    if cand_df.empty:
        return {"message": "Candidate data is empty after conversion."}

    report = {
        "total_candidates": len(cand_df),
        "candidates_by_status": cand_df["status"].value_counts().to_dict(),
    }

    # If requisitions are provided, add pipeline by job requisition
    if requisitions:
        req_df = convert_requisitions_to_dataframe(requisitions)
        if not req_df.empty:
            # Active requisitions with their candidate counts by status.
            pipeline_by_req = {}
            active_reqs = req_df[req_df["status"] == JobRequisitionStatus.OPEN]
            for _, req in active_reqs.iterrows():
                req_id = req["requisition_id"]
                req_cands = cand_df[cand_df["job_requisition_id"] == req_id]
                pipeline_by_req[req_id] = {
                    "job_title": req["job_title"],
                    "total_candidates_for_req": len(req_cands),
                    "status_counts": (
                        req_cands["status"].value_counts().to_dict()
                        if not req_cands.empty
                        else {}
                    ),
                }
            report["pipeline_by_active_requisition"] = pipeline_by_req

    logger.info("Generated candidate pipeline report.")
    return report


def calculate_time_to_hire(hired_candidates: list[Candidate]) -> dict[str, Any]:
    """
    Calculates average, min, max time to hire for candidates who reached 'HIRED' status.
    Assumes 'applied_at' and 'offer.accepted_at' or a 'hired_at' field exists and is populated.
    (Simplified: uses applied_at and assumes updated_at for HIRED status is effectively hired_at)
    """
    if not hired_candidates:
        return {
            "message": "No hired candidate data for time-to-hire calculation.",
            "avg_time_to_hire_days": None,
        }

    durations = []
    for cand in hired_candidates:
        if cand.status == CandidateStatus.HIRED:
            # updated_at is a datetime per the model; used as the hired_at proxy
            duration = cand.updated_at - cand.applied_at
            durations.append(duration.days)

    if not durations:
        return {
            "message": "Not enough valid data to calculate time to hire.",
            "avg_time_to_hire_days": None,
        }

    report = {
        "number_of_hires_in_calc": len(durations),
        "avg_time_to_hire_days": (
            round(sum(durations) / len(durations), 2) if durations else None
        ),
        "min_time_to_hire_days": min(durations) if durations else None,
        "max_time_to_hire_days": max(durations) if durations else None,
    }
    logger.info("Calculated time to hire report.")
    return report


def get_quarterly_metrics(
    quarter: str, year: int, candidates: list[Candidate] | None = None
) -> dict[str, Any]:
    """Calculate talent metrics from candidate records."""
    if not candidates:
        return {
            "quarter": quarter,
            "year": year,
            "message": "No candidate data available for metrics calculation",
            "candidates_sourced": 0,
            "hired_candidates": 0,
            "offer_acceptance_rate_percent": None,
            "avg_time_to_hire_days": None,
        }

    hired = [
        candidate
        for candidate in candidates
        if candidate.status == CandidateStatus.HIRED
    ]
    offers = [candidate for candidate in candidates if candidate.offer is not None]
    accepted_offers = [
        candidate
        for candidate in offers
        if candidate.offer is not None
        and (
            candidate.offer.status == "accepted"
            or candidate.offer.accepted_at is not None
        )
    ]
    time_to_hire = calculate_time_to_hire(hired)
    return {
        "quarter": quarter,
        "year": year,
        "candidates_sourced": len(candidates),
        "hired_candidates": len(hired),
        "offer_acceptance_rate_percent": (
            len(accepted_offers) / len(offers) * 100 if offers else None
        ),
        "avg_time_to_hire_days": time_to_hire.get("avg_time_to_hire_days"),
    }


# More talent reports:
# - Offer acceptance rate
# - Source effectiveness (which sources yield most hires)
# - Interviewer load and feedback turnaround time
