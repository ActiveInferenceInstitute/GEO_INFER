"""Talent Acquisition API Endpoints."""

from datetime import datetime
from typing import Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
import logging
from ..models.talent_models import Candidate, CandidateStatus, JobRequisition
from ..core.data_store import pep_data_manager as store
from ..talent.importer import CSVTalentImporter
from ..talent.transformer import clean_candidate_data, enrich_candidate_data
from ..reporting.talent_reports import (
    generate_candidate_pipeline_report,
    calculate_time_to_hire,
)
from ..visualizations.talent_visuals import plot_candidate_pipeline_by_status
from ..utils import save_upload_file_tmp

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/talent",
    tags=["Talent"],
)


@router.post("/upload/candidates/csv", response_model=dict[str, Any])
async def upload_candidates_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV file with candidate data."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only CSV files."
        )
    temp_file_path = await save_upload_file_tmp(file)
    try:
        importer = CSVTalentImporter(candidate_file_path=str(temp_file_path))
        imported_candidates = importer.import_candidates()
        processed_candidates = clean_candidate_data(imported_candidates)
        processed_candidates = enrich_candidate_data(
            processed_candidates, store.requisitions
        )

        store.candidates.extend(processed_candidates)
        return {
            "message": f"Imported {len(store.candidates)} candidates from {file.filename}"
        }
    except (FileNotFoundError, ValueError) as exc:
        # Domain failure modes of the CSV import path (CSVTalentImporter.connect
        # and CandidateStatus parsing): client faults -> 400 with the domain
        # message. Anything else escapes to the shared error middleware.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


@router.get("/candidates", response_model=list[Candidate])
async def get_all_candidates(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[Candidate]:
    return store.candidates[offset : offset + limit]


@router.get("/reports/candidate-pipeline", response_model=dict[str, Any])
async def get_talent_candidate_pipeline_report() -> dict[str, Any]:
    if not store.candidates:
        raise HTTPException(
            status_code=404, detail="No candidate data. Upload data first."
        )
    return generate_candidate_pipeline_report(store.candidates, store.requisitions)


@router.get("/reports/time-to-hire", response_model=dict[str, Any])
async def get_talent_time_to_hire_report() -> dict[str, Any]:
    hired_candidates = [
        cand for cand in store.candidates if cand.status == CandidateStatus.HIRED
    ]
    if not hired_candidates:
        raise HTTPException(
            status_code=404, detail="No hired candidates found for TTH report."
        )
    return calculate_time_to_hire(hired_candidates)


@router.get("/visualizations/candidate-pipeline-status", response_model=dict[str, str])
async def get_candidate_pipeline_status_plot() -> dict[str, str]:
    if not store.candidates:
        raise HTTPException(
            status_code=404, detail="No candidate data for visualization."
        )
    plot_path = plot_candidate_pipeline_by_status(store.candidates)
    if plot_path:
        return {"message": "Plot generated", "plot_file_path": plot_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate plot.")


@router.get("/requisitions", response_model=list[JobRequisition])
async def get_all_requisitions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[JobRequisition]:
    """Retrieve job requisitions from the in-memory store."""
    return store.requisitions[offset : offset + limit]


@router.post("/requisitions", response_model=JobRequisition, status_code=201)
async def create_requisition(requisition: JobRequisition) -> JobRequisition:
    """Create a job requisition record in the in-memory store."""
    if any(
        existing.requisition_id == requisition.requisition_id
        for existing in store.requisitions
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job requisition with id '{requisition.requisition_id}' "
                "already exists."
            ),
        )
    store.requisitions.append(requisition)
    return requisition


@router.get("/requisitions/{requisition_id}", response_model=JobRequisition)
async def get_requisition(requisition_id: str) -> JobRequisition:
    """Retrieve a single job requisition by id."""
    for existing in store.requisitions:
        if existing.requisition_id == requisition_id:
            return existing
    raise HTTPException(
        status_code=404,
        detail=f"Job requisition with id '{requisition_id}' not found.",
    )


@router.put("/requisitions/{requisition_id}", response_model=JobRequisition)
async def update_requisition(
    requisition_id: str, updated: JobRequisition
) -> JobRequisition:
    """Replace a requisition record; preserves created_at and refreshes updated_at."""
    for index, existing in enumerate(store.requisitions):
        if existing.requisition_id == requisition_id:
            merged = updated.model_copy(
                update={
                    "requisition_id": requisition_id,
                    "created_at": existing.created_at,
                    "updated_at": datetime.now(),
                }
            )
            store.requisitions[index] = merged
            return merged
    raise HTTPException(
        status_code=404,
        detail=f"Job requisition with id '{requisition_id}' not found.",
    )


@router.delete("/requisitions/{requisition_id}", response_model=dict[str, Any])
async def delete_requisition(requisition_id: str) -> dict[str, Any]:
    """Delete a job requisition record by id."""
    for index, existing in enumerate(store.requisitions):
        if existing.requisition_id == requisition_id:
            store.requisitions.pop(index)
            return {"deleted": requisition_id}
    raise HTTPException(
        status_code=404,
        detail=f"Job requisition with id '{requisition_id}' not found.",
    )


# Not implemented (no backing behavior yet — do not rely on these):
# - Endpoints for specific candidate by ID, updating candidate status,
#   interviews, and offers
