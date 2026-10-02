"""GEO-INFER-PEP: People, Engagement, Performance operations module.

Public surface: pydantic domain models, the high-level ``methods`` workflow
functions, the core engine/data-manager classes, and the FastAPI routers.
"""

from .models import (
    Address,
    Candidate,
    CandidateStatus,
    Compensation,
    Customer,
    Employee,
    EmploymentStatus,
    Gender,
    InteractionLog,
    Interview,
    InterviewFeedback,
    InterviewType,
    JobHistoryEntry,
    JobRequisition,
    JobRequisitionStatus,
    Offer,
    PerformanceReview,
)
from .core import PEPEngine, PEPDataManager, PEPOrchestrator, PEPValidator
from .core.data_store import pep_data_manager
from .methods import (
    clear_all_data,
    generate_comprehensive_crm_dashboard,
    generate_comprehensive_hr_dashboard,
    generate_comprehensive_talent_dashboard,
    generate_quarterly_people_report,
    get_all_candidates,
    get_all_customers,
    get_all_employees,
    import_crm_data_from_csv,
    import_hr_data_from_csv,
    import_talent_data_from_csv,
    process_employee_onboarding_workflow,
)
from .api import api_router, crm_router, hr_router, talent_router

__version__ = "0.4.0"

__all__ = [
    # Models
    "Address",
    "Candidate",
    "CandidateStatus",
    "Compensation",
    "Customer",
    "Employee",
    "EmploymentStatus",
    "Gender",
    "InteractionLog",
    "Interview",
    "InterviewFeedback",
    "InterviewType",
    "JobHistoryEntry",
    "JobRequisition",
    "JobRequisitionStatus",
    "Offer",
    "PerformanceReview",
    # Core
    "PEPEngine",
    "PEPDataManager",
    "PEPOrchestrator",
    "PEPValidator",
    "pep_data_manager",
    # Methods
    "clear_all_data",
    "generate_comprehensive_crm_dashboard",
    "generate_comprehensive_hr_dashboard",
    "generate_comprehensive_talent_dashboard",
    "generate_quarterly_people_report",
    "get_all_candidates",
    "get_all_customers",
    "get_all_employees",
    "import_crm_data_from_csv",
    "import_hr_data_from_csv",
    "import_talent_data_from_csv",
    "process_employee_onboarding_workflow",
    # API routers
    "api_router",
    "crm_router",
    "hr_router",
    "talent_router",
]
