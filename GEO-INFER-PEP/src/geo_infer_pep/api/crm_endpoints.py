"""CRM API Endpoints."""

from datetime import datetime
from typing import Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
import logging

from ..models.crm_models import Customer
from ..core.data_store import pep_data_manager as store
from ..crm.importer import CSVCRMImporter
from ..crm.transformer import clean_customer_data, enrich_customer_data
from ..reporting.crm_reports import (
    generate_customer_segmentation_report,
    generate_lead_conversion_report,
)
from ..visualizations.crm_visuals import plot_customer_distribution_by_status
from ..utils import save_upload_file_tmp

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/crm",
    tags=["CRM"],
)


@router.post("/upload/csv", response_model=dict[str, Any])
async def upload_crm_csv(
    file: UploadFile = File(...),
    clean_data: bool = Query(True, description="Perform data cleaning after import"),
    enrich_data: bool = Query(
        True, description="Perform data enrichment after cleaning"
    ),
) -> dict[str, Any]:
    """
    Upload a CSV file with CRM data. Data will be imported, (optionally) cleaned and enriched,
    and then appended to the shared in-memory store (non-destructive).
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only CSV files are accepted."
        )

    temp_file_path = await save_upload_file_tmp(file)

    try:
        importer = CSVCRMImporter(file_path=str(temp_file_path))
        imported_customers = importer.import_customers()

        processed_customers = imported_customers
        if clean_data:
            processed_customers = clean_customer_data(processed_customers)
        if enrich_data:
            processed_customers = enrich_customer_data(processed_customers)

        # Appended to the shared in-memory store; in a real app you'd save
        # to a persistent database.
        store.customers.extend(processed_customers)

        return {
            "message": f"Successfully imported and processed {len(processed_customers)} customers from {file.filename}",
            "imported_count": len(imported_customers),
            "processed_count": len(processed_customers),
            "total_customers_in_store": len(store.customers),
            "cleaning_applied": clean_data,
            "enrichment_applied": enrich_data,
        }
    except ConnectionError as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to connect to data source: {e}"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Temporary CSV file not found after upload. This should not happen.",
        )
    # Non-domain failures escape to the shared error middleware, which returns
    # a generic 500 without leaking internal exception details.
    finally:
        # Clean up the temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()


@router.get("/customers", response_model=list[Customer])
async def get_all_customers(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> list[Customer]:
    """Retrieve all customers from the in-memory store."""
    return store.customers[offset : offset + limit]


@router.get("/customers/count", response_model=dict[str, int])
async def get_customers_count() -> dict[str, int]:
    """Get the total number of customers in the in-memory store."""
    return {"total_customers": len(store.customers)}


@router.get("/reports/segmentation", response_model=dict[str, Any])
async def get_crm_segmentation_report() -> dict[str, Any]:
    """
    Generate and return a customer segmentation report.
    """
    if not store.customers:
        raise HTTPException(
            status_code=404,
            detail="No customer data available to generate report. Please upload data first.",
        )
    report = generate_customer_segmentation_report(store.customers)
    return report


@router.get("/reports/lead-conversion", response_model=dict[str, Any])
async def get_crm_lead_conversion_report() -> dict[str, Any]:
    """
    Generate and return a lead conversion report.
    """
    if not store.customers:
        raise HTTPException(
            status_code=404,
            detail="No customer data available to generate report. Please upload data first.",
        )
    report = generate_lead_conversion_report(store.customers)
    return report


@router.get("/visualizations/status-distribution", response_model=dict[str, str])
async def get_status_distribution_plot() -> dict[str, str]:
    """
    Generate a customer status distribution plot and return its path.
    (In a real app, you might return the image directly or a URL).
    """
    if not store.customers:
        raise HTTPException(
            status_code=404,
            detail="No customer data available to generate visualization. Please upload data first.",
        )

    # Create a temporary directory for this request's plot if needed, or use a shared one
    # For simplicity, using the default from crm_visuals
    plot_path = plot_customer_distribution_by_status(store.customers)
    if plot_path:
        return {"message": "Plot generated successfully", "plot_file_path": plot_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate plot.")


@router.post("/customers", response_model=Customer, status_code=201)
async def create_customer(customer: Customer) -> Customer:
    """Create a customer record in the in-memory store."""
    if any(
        existing.customer_id == customer.customer_id for existing in store.customers
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Customer with id '{customer.customer_id}' already exists.",
        )
    store.customers.append(customer)
    return customer


@router.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: str) -> Customer:
    """Retrieve a single customer by id."""
    for existing in store.customers:
        if existing.customer_id == customer_id:
            return existing
    raise HTTPException(
        status_code=404, detail=f"Customer with id '{customer_id}' not found."
    )


@router.put("/customers/{customer_id}", response_model=Customer)
async def update_customer(customer_id: str, updated: Customer) -> Customer:
    """Replace a customer record; preserves created_at and refreshes updated_at."""
    for index, existing in enumerate(store.customers):
        if existing.customer_id == customer_id:
            merged = updated.model_copy(
                update={
                    "customer_id": customer_id,
                    "created_at": existing.created_at,
                    "updated_at": datetime.now(),
                }
            )
            store.customers[index] = merged
            return merged
    raise HTTPException(
        status_code=404, detail=f"Customer with id '{customer_id}' not found."
    )


@router.delete("/customers/{customer_id}", response_model=dict[str, Any])
async def delete_customer(customer_id: str) -> dict[str, Any]:
    """Delete a customer record by id."""
    for index, existing in enumerate(store.customers):
        if existing.customer_id == customer_id:
            store.customers.pop(index)
            return {"deleted": customer_id}
    raise HTTPException(
        status_code=404, detail=f"Customer with id '{customer_id}' not found."
    )


# Not implemented (no backing behavior yet — do not rely on these):
#   Additional report and visualisation endpoints beyond the ones above.

# To run this (conceptual, assuming main.py wires this router):
# uvicorn main:app --reload
