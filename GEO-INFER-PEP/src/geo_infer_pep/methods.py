"""
Central methods for orchestrating GEO-INFER-PEP functionalities.

This module contains high-level functions that combine various
operations from the submodules (crm, hr, talent, reporting, etc.)
to perform complex tasks.

All data flows through the process-wide shared in-memory store
(``core.data_store.pep_data_manager``), which the FastAPI layer also uses.
Library code logs via :mod:`logging` and never prints directly.
"""

from typing import Any
from datetime import datetime
import logging

# Import actual module implementations and data models
from .models.hr_models import Employee, EmploymentStatus
from .models.talent_models import Candidate, CandidateStatus
from .models.crm_models import Customer
from .core.data_store import pep_data_manager
from .reporting import (
    get_hr_quarterly_metrics,
    get_crm_quarterly_metrics,
    get_talent_quarterly_metrics,
    create_quarterly_overview,
)

# Import utility functions for data processing
from .hr.transformer import clean_employee_data, enrich_employee_data
from .crm.transformer import clean_customer_data, enrich_customer_data
from .talent.transformer import clean_candidate_data, enrich_candidate_data

# Import importers for data processing
from .hr.importer import CSVHRImporter
from .crm.importer import CSVCRMImporter
from .talent.importer import CSVTalentImporter

# Import reporting functions
from .reporting.hr_reports import generate_headcount_report, generate_diversity_report
from .reporting.crm_reports import (
    generate_customer_segmentation_report,
    generate_lead_conversion_report,
)
from .reporting.talent_reports import (
    generate_candidate_pipeline_report,
    calculate_time_to_hire,
)

# The shared store backs every accessor below. The names ``_employees_db``,
# ``_candidates_db`` and ``_customers_db`` are the same list objects held by
# ``pep_data_manager``, so methods-layer and API-layer mutations stay in sync.
_employees_db = pep_data_manager.employees
_candidates_db = pep_data_manager.candidates
_customers_db = pep_data_manager.customers

logger = logging.getLogger(__name__)

__all__ = [
    "process_employee_onboarding_workflow",
    "generate_quarterly_people_report",
    "import_hr_data_from_csv",
    "import_crm_data_from_csv",
    "import_talent_data_from_csv",
    "generate_comprehensive_hr_dashboard",
    "generate_comprehensive_crm_dashboard",
    "generate_comprehensive_talent_dashboard",
    "get_all_employees",
    "get_all_candidates",
    "get_all_customers",
    "clear_all_data",
]


def _normalize_quarter_label(quarter: str) -> str:
    """Normalize quarter inputs like '1' and 'Q1' to a single label."""
    normalized = str(quarter).strip().upper()
    if normalized.startswith("Q"):
        normalized = normalized[1:]
    return f"Q{normalized}"


def _get_employee_by_id(employee_id: str) -> Employee | None:
    """Helper function to find employee by ID."""
    return next((emp for emp in _employees_db if emp.employee_id == employee_id), None)


def _get_candidate_by_id(candidate_id: str) -> Candidate | None:
    """Helper function to find candidate by ID."""
    return next(
        (cand for cand in _candidates_db if cand.candidate_id == candidate_id), None
    )


def _get_customer_by_id(customer_id: str) -> Customer | None:
    """Helper function to find customer by ID."""
    return next(
        (cust for cust in _customers_db if cust.customer_id == customer_id), None
    )


def process_employee_onboarding_workflow(employee_data: dict) -> bool:
    """
    Orchestrates the full employee onboarding process using real data models and processing.

    Args:
        employee_data: Dictionary containing candidate_id and other employee information

    Returns:
        bool: True if onboarding completed successfully, False otherwise

    Raises:
        ValueError: If required data is missing
        TypeError: If benefits_client or learning_client is provided but not callable

    Returns False (and does not raise) when the candidate_id does not match
    an existing candidate or the candidate is not in OFFER_ACCEPTED state.
    """
    candidate_id = employee_data.get("candidate_id")
    if not candidate_id:
        raise ValueError("candidate_id is required for onboarding workflow")

    logger.info("Starting onboarding workflow for candidate %s", candidate_id)

    # Find candidate in database
    candidate = _get_candidate_by_id(candidate_id)
    if not candidate:
        logger.error("Onboarding Aborted: Candidate %s not found.", candidate_id)
        return False

    if candidate.status != CandidateStatus.OFFER_ACCEPTED:
        logger.error(
            "Onboarding Aborted: Candidate %s status is %s, expected %s",
            candidate_id,
            candidate.status,
            CandidateStatus.OFFER_ACCEPTED,
        )
        return False

    # Validate integration clients before touching the shared store: a
    # malformed configuration must abort cleanly, so these TypeErrors
    # intentionally escape the try block below.
    benefits_client = employee_data.get("benefits_client")
    if benefits_client is not None and not callable(benefits_client):
        raise TypeError("benefits_client must be callable")
    learning_client = employee_data.get("learning_client")
    if learning_client is not None and not callable(learning_client):
        raise TypeError("learning_client must be callable")

    try:
        # Create employee record using candidate data
        employee_id = f"emp_{candidate_id}_{candidate.first_name.lower()}_{candidate.last_name.lower()}"

        job_title = employee_data.get("job_title") or "New Hire"
        department = employee_data.get("department") or "General"
        location = employee_data.get("location") or "Remote"

        # Create Employee object with candidate data
        employee = Employee(
            employee_id=employee_id,
            first_name=candidate.first_name,
            last_name=candidate.last_name,
            email=candidate.email,
            phone_number=candidate.phone_number or "",
            hire_date=(
                candidate.offer.accepted_at
                if candidate.offer and candidate.offer.accepted_at
                else None
            ),
            employment_status=EmploymentStatus.ACTIVE,
            job_title=job_title,
            department=department,
            location=location,
        )

        # Process through HR pipeline (clean and enrich data)

        cleaned_employees = clean_employee_data([employee])
        enriched_employees = enrich_employee_data(cleaned_employees)

        # Update employee with enriched data
        updated_employee = enriched_employees[0]
        employee_id = updated_employee.employee_id

        # Invoke downstream integrations BEFORE persisting: any failure here
        # must leave the shared store untouched (clean abort, no partial
        # employee record).
        if benefits_client is not None:
            benefits_client(employee)
        else:
            logger.info(
                "Benefits enrollment not performed; no benefits integration configured for %s",
                employee_id,
            )

        if learning_client is not None:
            learning_client(employee)
        else:
            logger.info(
                "Training scheduling not performed; no learning integration configured for %s",
                employee_id,
            )

        # All processing succeeded; only now add employee to the database.
        _employees_db.append(employee)

        logger.info(
            "Onboarding workflow for %s %s (employee ID: %s) completed successfully.",
            candidate.first_name,
            candidate.last_name,
            employee_id,
        )
        return True

    except Exception as e:
        logger.error("Onboarding Failed: Error during employee creation - %s", e)
        return False


def generate_quarterly_people_report(quarter: str, year: int) -> str:
    """
    Generates a comprehensive quarterly people operations report.
    - Gathers HR metrics (headcount, attrition, diversity).
    - Gathers CRM metrics (customer satisfaction, new leads).
    - Gathers Talent metrics (time-to-hire, offer acceptance rate).
    - Compiles into a single report.
    """
    quarter_label = _normalize_quarter_label(quarter)
    logger.info("Generating quarterly people report for %s %s...", quarter_label, year)

    hr_metrics = get_hr_quarterly_metrics(quarter_label, year, _employees_db)
    crm_metrics = get_crm_quarterly_metrics(quarter_label, year, _customers_db)
    talent_metrics = get_talent_quarterly_metrics(quarter_label, year, _candidates_db)

    report_path = create_quarterly_overview(
        hr_metrics=hr_metrics,
        crm_metrics=crm_metrics,
        talent_metrics=talent_metrics,
    )

    logger.info("Quarterly people report generated at %s.", report_path)
    return report_path


def import_hr_data_from_csv(file_path: str) -> list[Employee]:
    """
    Complete HR data pipeline: import, clean, enrich, and store employee data.

    Args:
        file_path: Path to CSV file containing HR data

    Returns:
        List of processed Employee objects

    Raises:
        Exception: any importer/transformer failure is logged and re-raised.
    """
    logger.info("Starting HR data import from %s", file_path)

    # Import raw data
    importer = CSVHRImporter(file_path)
    employees = importer.import_employees()

    if not employees:
        logger.warning("No employee data found in %s", file_path)
        return []

    # Clean and enrich data
    cleaned_employees = clean_employee_data(employees)
    enriched_employees = enrich_employee_data(cleaned_employees)

    # Store in database
    _employees_db.extend(enriched_employees)

    logger.info(
        "Successfully imported and processed %d employee records",
        len(enriched_employees),
    )
    return enriched_employees


def import_crm_data_from_csv(file_path: str) -> list[Customer]:
    """
    Complete CRM data pipeline: import, clean, enrich, and store customer data.

    Args:
        file_path: Path to CSV file containing CRM data

    Returns:
        List of processed Customer objects

    Raises:
        Exception: any importer/transformer failure is logged and re-raised.
    """
    logger.info("Starting CRM data import from %s", file_path)

    # Import raw data
    importer = CSVCRMImporter(file_path)
    customers = importer.import_customers()

    if not customers:
        logger.warning("No customer data found in %s", file_path)
        return []

    # Clean and enrich data
    cleaned_customers = clean_customer_data(customers)
    enriched_customers = enrich_customer_data(cleaned_customers)

    # Store in database
    _customers_db.extend(enriched_customers)

    logger.info(
        "Successfully imported and processed %d customer records",
        len(enriched_customers),
    )
    return enriched_customers


def import_talent_data_from_csv(
    candidates_file: str, requisitions_file: str
) -> dict[str, Any]:
    """
    Complete talent data pipeline: import, clean, enrich, and store talent data.

    Args:
        candidates_file: Path to CSV file containing candidate data
        requisitions_file: Path to CSV file containing job requisition data

    Returns:
        Dictionary with processed candidates and requisitions

    Raises:
        Exception: any importer/transformer failure is logged and re-raised.
    """
    logger.info(
        "Starting talent data import from %s and %s", candidates_file, requisitions_file
    )

    # Import talent data
    importer = CSVTalentImporter(candidates_file, requisitions_file)
    candidates = importer.import_candidates()
    requisitions = importer.import_requisitions()

    if candidates:
        # Clean and enrich candidate data
        cleaned_candidates = clean_candidate_data(candidates)
        enriched_candidates = enrich_candidate_data(cleaned_candidates, requisitions)

        # Store in database
        _candidates_db.extend(enriched_candidates)

        logger.info(
            "Successfully imported and processed %d candidate records",
            len(enriched_candidates),
        )

    if requisitions:
        # Store requisitions so dashboards and enrichment can use them.
        pep_data_manager.add_requisitions(requisitions)
        logger.info(
            "Successfully imported and processed %d requisition records",
            len(requisitions),
        )

    return {
        "candidates": len(_candidates_db),
        "requisitions": len(requisitions),
        "processed_successfully": True,
    }


def generate_comprehensive_hr_dashboard() -> dict[str, Any]:
    """
    Generate comprehensive HR dashboard data combining multiple analytics.

    Returns:
        Dictionary containing various HR metrics and insights
    """
    if not _employees_db:
        return {"message": "No employee data available for dashboard"}

    logger.info("Generating comprehensive HR dashboard...")

    try:
        # Generate various HR reports
        headcount_report = generate_headcount_report(
            _employees_db, group_by=["department"]
        )
        diversity_report = generate_diversity_report(
            _employees_db, diversity_fields=["gender", "department"]
        )

        # Calculate additional metrics
        total_employees = len(_employees_db)
        active_employees = len(
            [e for e in _employees_db if e.employment_status == EmploymentStatus.ACTIVE]
        )

        # Department breakdown
        dept_breakdown: dict[str, int] = {}
        for emp in _employees_db:
            if emp.employment_status == EmploymentStatus.ACTIVE:
                dept_breakdown[emp.department] = (
                    dept_breakdown.get(emp.department, 0) + 1
                )

        dashboard_data = {
            "total_employees": total_employees,
            "active_employees": active_employees,
            "headcount_by_department": dept_breakdown,
            "headcount_report": headcount_report,
            "diversity_report": diversity_report,
            "generated_at": datetime.now().isoformat(),
            "data_freshness": f"Based on {len(_employees_db)} employee records",
        }

        logger.info("HR dashboard generated successfully")
        return dashboard_data

    except Exception as e:
        logger.error("Error generating HR dashboard: %s", e)
        return {"error": str(e), "message": "Failed to generate dashboard"}


def generate_comprehensive_crm_dashboard() -> dict[str, Any]:
    """
    Generate comprehensive CRM dashboard data combining multiple analytics.

    Returns:
        Dictionary containing various CRM metrics and insights
    """
    if not _customers_db:
        return {"message": "No customer data available for dashboard"}

    logger.info("Generating comprehensive CRM dashboard...")

    try:
        # Generate CRM reports
        segmentation_report = generate_customer_segmentation_report(_customers_db)
        conversion_report = generate_lead_conversion_report(_customers_db)

        # Calculate additional metrics
        total_customers = len(_customers_db)
        active_customers = len([c for c in _customers_db if c.status == "active"])

        # Status breakdown
        status_breakdown: dict[str, int] = {}
        for cust in _customers_db:
            status = cust.status or "unknown"
            status_breakdown[status] = status_breakdown.get(status, 0) + 1

        dashboard_data = {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "status_breakdown": status_breakdown,
            "segmentation_report": segmentation_report,
            "conversion_report": conversion_report,
            "generated_at": datetime.now().isoformat(),
            "data_freshness": f"Based on {len(_customers_db)} customer records",
        }

        logger.info("CRM dashboard generated successfully")
        return dashboard_data

    except Exception as e:
        logger.error("Error generating CRM dashboard: %s", e)
        return {"error": str(e), "message": "Failed to generate dashboard"}


def generate_comprehensive_talent_dashboard() -> dict[str, Any]:
    """
    Generate comprehensive talent dashboard data combining multiple analytics.

    Returns:
        Dictionary containing various talent metrics and insights
    """
    if not _candidates_db:
        return {"message": "No candidate data available for dashboard"}

    logger.info("Generating comprehensive talent dashboard...")

    try:
        # Generate talent reports (requisitions come from the shared store)
        pipeline_report = generate_candidate_pipeline_report(
            _candidates_db, list(pep_data_manager.requisitions)
        )
        time_to_hire_report = calculate_time_to_hire(_candidates_db)

        # Calculate additional metrics
        total_candidates = len(_candidates_db)

        # Status breakdown
        status_breakdown: dict[str, int] = {}
        for cand in _candidates_db:
            status_breakdown[cand.status.value] = (
                status_breakdown.get(cand.status.value, 0) + 1
            )

        dashboard_data = {
            "total_candidates": total_candidates,
            "status_breakdown": status_breakdown,
            "pipeline_report": pipeline_report,
            "time_to_hire_report": time_to_hire_report,
            "generated_at": datetime.now().isoformat(),
            "data_freshness": f"Based on {len(_candidates_db)} candidate records",
        }

        logger.info("Talent dashboard generated successfully")
        return dashboard_data

    except Exception as e:
        logger.error("Error generating talent dashboard: %s", e)
        return {"error": str(e), "message": "Failed to generate dashboard"}


def get_all_employees() -> list[Employee]:
    """Get all employees from the shared in-memory store."""
    return list(_employees_db)


def get_all_candidates() -> list[Candidate]:
    """Get all candidates from the shared in-memory store."""
    return list(_candidates_db)


def get_all_customers() -> list[Customer]:
    """Get all customers from the shared in-memory store."""
    return list(_customers_db)


def clear_all_data() -> bool:
    """Clear all data from the shared in-memory store.

    Destructive: removes every employee, customer, candidate, and requisition
    record visible to both the methods layer and the FastAPI layer.
    Intended for tests and controlled resets.
    """
    logger.warning("clear_all_data(): deleting all in-memory PEP records")
    pep_data_manager.clear_all_data()
    return True
