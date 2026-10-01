# API endpoints for GEO-INFER-PEP
from fastapi import APIRouter, HTTPException
from typing import Any, Dict, Optional

from pydantic import ValidationError

from ..models.hr_models import PerformanceReview
from ..performance_store import performance_review_store
from ..models.learning_models import LearningCourse, LearningEnrollment
from ..models.conflict_models import ConflictCase
from ..models.survey_models import Survey, SurveyResponse
from ..core.data_store import pep_data_manager
from .crm_endpoints import router as crm_router
from .hr_endpoints import router as hr_router
from .talent_endpoints import router as talent_router
from .errors import ErrorHandlerMiddleware, register_error_handlers

# Create the main API router
api_router = APIRouter(prefix="/pep")

# Include module-specific routers
api_router.include_router(crm_router)
api_router.include_router(hr_router)
api_router.include_router(talent_router)


# Health and system endpoints
@api_router.get("/health", response_model=dict[str, Any])
async def health_check() -> dict[str, Any]:
    """System health check endpoint."""
    from ..core.pep_engine import PEPEngine

    engine = PEPEngine()
    health_status = engine.run_health_check()
    return health_status


@api_router.get("/status", response_model=dict[str, Any])
async def system_status() -> dict[str, Any]:
    """Get comprehensive system status."""
    from ..core.pep_engine import PEPEngine

    engine = PEPEngine()
    return engine.get_system_status()


@api_router.get("/dashboard", response_model=dict[str, Any])
async def system_dashboard() -> dict[str, Any]:
    """Get comprehensive system dashboard."""
    from ..core.pep_engine import PEPEngine

    engine = PEPEngine()
    return engine.generate_all_dashboards()


# Onboarding workflow endpoints
@api_router.post("/workflows/onboarding/{candidate_id}", response_model=dict[str, Any])
async def create_onboarding_workflow(candidate_id: str) -> dict[str, Any]:
    """Create an onboarding workflow for a candidate."""
    from ..core.orchestrator import PEPOrchestrator

    orchestrator = PEPOrchestrator()
    workflow_id = orchestrator.create_employee_onboarding_workflow(candidate_id)
    return {"workflow_id": workflow_id, "status": "created"}


@api_router.get("/workflows/{workflow_id}", response_model=dict[str, Any])
async def get_workflow_status(workflow_id: str) -> dict[str, Any]:
    """Get workflow status."""
    from ..core.orchestrator import PEPOrchestrator

    orchestrator = PEPOrchestrator()
    return orchestrator.get_workflow_status(workflow_id)


@api_router.post("/workflows/{workflow_id}/execute", response_model=dict[str, Any])
async def execute_workflow(workflow_id: str) -> dict[str, Any]:
    """Execute a workflow."""
    from ..core.orchestrator import PEPOrchestrator

    orchestrator = PEPOrchestrator()
    return orchestrator.execute_workflow(workflow_id)


# Performance management endpoints
@api_router.post("/performance/reviews", response_model=dict[str, Any])
async def create_performance_review(review_data: dict[str, Any]) -> dict[str, Any]:
    """Create a performance review."""
    employee_id = review_data.get("employee_id")
    if not employee_id:
        raise HTTPException(status_code=400, detail="employee_id is required")
    try:
        review = PerformanceReview(**review_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    performance_review_store.add(employee_id, review)
    return {
        "message": "Performance review created",
        "employee_id": employee_id,
        "data": review.model_dump(mode="json"),
    }


@api_router.get("/performance/reviews/{employee_id}", response_model=dict[str, Any])
async def get_performance_reviews(employee_id: str) -> dict[str, Any]:
    """Get performance reviews for an employee."""
    return {
        "employee_id": employee_id,
        "reviews": [
            review.model_dump(mode="json")
            for review in performance_review_store.list_for_employee(employee_id)
        ],
    }


# Learning & Development endpoints
@api_router.post("/learning/courses", response_model=dict[str, Any])
async def create_learning_course(course_data: dict[str, Any]) -> dict[str, Any]:
    """Create a learning course in the shared store."""
    try:
        course = LearningCourse(**course_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if any(
        existing.course_id == course.course_id
        for existing in pep_data_manager.learning_courses
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Learning course with id '{course.course_id}' already exists.",
        )
    pep_data_manager.add_learning_courses([course])
    return {
        "message": "Learning course created",
        "data": course.model_dump(mode="json"),
    }


@api_router.get("/learning/courses", response_model=dict[str, Any])
async def get_learning_courses() -> dict[str, Any]:
    """Get all learning courses stored in the shared store."""
    return {
        "courses": [
            course.model_dump(mode="json")
            for course in pep_data_manager.learning_courses
        ]
    }


@api_router.post("/learning/enrollments", response_model=dict[str, Any])
async def enroll_employee(enrollment_data: dict[str, Any]) -> dict[str, Any]:
    """Enroll an employee in a stored learning course."""
    try:
        enrollment = LearningEnrollment(**enrollment_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not pep_data_manager.get_learning_courses({"course_id": enrollment.course_id}):
        raise HTTPException(
            status_code=404,
            detail=f"Learning course with id '{enrollment.course_id}' not found.",
        )
    if any(
        existing.enrollment_id == enrollment.enrollment_id
        for existing in pep_data_manager.enrollments
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Learning enrollment with id '{enrollment.enrollment_id}' "
                "already exists."
            ),
        )
    pep_data_manager.add_enrollments([enrollment])
    return {
        "message": "Employee enrolled",
        "data": enrollment.model_dump(mode="json"),
    }


@api_router.get("/learning/enrollments", response_model=dict[str, Any])
async def get_learning_enrollments(
    employee_id: str | None = None, course_id: str | None = None
) -> dict[str, Any]:
    """Get learning enrollments with optional employee/course filters."""
    filters: dict[str, Any] = {}
    if employee_id is not None:
        filters["employee_id"] = employee_id
    if course_id is not None:
        filters["course_id"] = course_id
    return {
        "enrollments": [
            enrollment.model_dump(mode="json")
            for enrollment in pep_data_manager.get_enrollments(filters or None)
        ]
    }


# Conflict resolution endpoints
@api_router.post("/conflicts/cases", response_model=dict[str, Any])
async def create_conflict_case(case_data: dict[str, Any]) -> dict[str, Any]:
    """Create a conflict resolution case in the shared store."""
    try:
        case = ConflictCase(**case_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if any(
        existing.case_id == case.case_id for existing in pep_data_manager.conflict_cases
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Conflict case with id '{case.case_id}' already exists.",
        )
    pep_data_manager.add_conflict_cases([case])
    return {
        "message": "Conflict case created",
        "data": case.model_dump(mode="json"),
    }


@api_router.get("/conflicts/cases", response_model=dict[str, Any])
async def get_conflict_cases() -> dict[str, Any]:
    """Get all conflict resolution cases stored in the shared store."""
    return {
        "cases": [
            case.model_dump(mode="json") for case in pep_data_manager.conflict_cases
        ]
    }


@api_router.put("/conflicts/cases/{case_id}", response_model=dict[str, Any])
async def update_conflict_case(
    case_id: str, update_data: dict[str, Any]
) -> dict[str, Any]:
    """Update a stored conflict resolution case."""
    existing_index = next(
        (
            index
            for index, existing in enumerate(pep_data_manager.conflict_cases)
            if existing.case_id == case_id
        ),
        None,
    )
    if existing_index is None:
        raise HTTPException(
            status_code=404, detail=f"Conflict case with id '{case_id}' not found."
        )
    if "case_id" in update_data and update_data["case_id"] != case_id:
        raise HTTPException(status_code=400, detail="case_id cannot be changed")
    merged_data = pep_data_manager.conflict_cases[existing_index].model_dump()
    merged_data.update(update_data)
    try:
        updated = ConflictCase(**merged_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    pep_data_manager.conflict_cases[existing_index] = updated
    return {
        "message": f"Conflict case {case_id} updated",
        "data": updated.model_dump(mode="json"),
    }


# Survey endpoints
@api_router.post("/surveys", response_model=dict[str, Any])
async def create_survey(survey_data: dict[str, Any]) -> dict[str, Any]:
    """Create a survey in the shared store."""
    try:
        survey = Survey(**survey_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if any(
        existing.survey_id == survey.survey_id for existing in pep_data_manager.surveys
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Survey with id '{survey.survey_id}' already exists.",
        )
    pep_data_manager.add_surveys([survey])
    return {
        "message": "Survey created",
        "data": survey.model_dump(mode="json"),
    }


@api_router.get("/surveys/{survey_id}/responses", response_model=dict[str, Any])
async def get_survey_responses(survey_id: str) -> dict[str, Any]:
    """Get responses stored for a survey."""
    if not pep_data_manager.get_surveys({"survey_id": survey_id}):
        raise HTTPException(
            status_code=404, detail=f"Survey with id '{survey_id}' not found."
        )
    return {
        "survey_id": survey_id,
        "responses": [
            response.model_dump(mode="json")
            for response in pep_data_manager.get_survey_responses(
                {"survey_id": survey_id}
            )
        ],
    }


@api_router.post("/surveys/{survey_id}/responses", response_model=dict[str, Any])
async def submit_survey_response(
    survey_id: str, response_data: dict[str, Any]
) -> dict[str, Any]:
    """Submit and store a response for a survey (path survey_id wins)."""
    if not pep_data_manager.get_surveys({"survey_id": survey_id}):
        raise HTTPException(
            status_code=404, detail=f"Survey with id '{survey_id}' not found."
        )
    try:
        response = SurveyResponse(**{**response_data, "survey_id": survey_id})
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if any(
        existing.response_id == response.response_id
        for existing in pep_data_manager.survey_responses
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Survey response with id '{response.response_id}' already exists."
            ),
        )
    pep_data_manager.add_survey_responses([response])
    return {
        "message": f"Response submitted for survey {survey_id}",
        "data": response.model_dump(mode="json"),
    }


# Data validation endpoints
@api_router.post("/validate/employee", response_model=dict[str, Any])
async def validate_employee_data(employee_data: dict[str, Any]) -> dict[str, Any]:
    """Validate employee data."""
    from ..core.validator import PEPValidator
    from ..models.hr_models import Employee

    try:
        employee = Employee(**employee_data)
        validator = PEPValidator()
        result = validator.validate_employee(employee)
        return result.to_dict()
    except Exception as e:
        return {"is_valid": False, "errors": [str(e)], "error_count": 1}


@api_router.post("/validate/customer", response_model=dict[str, Any])
async def validate_customer_data(customer_data: dict[str, Any]) -> dict[str, Any]:
    """Validate customer data."""
    from ..core.validator import PEPValidator
    from ..models.crm_models import Customer

    try:
        customer = Customer(**customer_data)
        validator = PEPValidator()
        result = validator.validate_customer(customer)
        return result.to_dict()
    except Exception as e:
        return {"is_valid": False, "errors": [str(e)], "error_count": 1}


@api_router.post("/validate/candidate", response_model=dict[str, Any])
async def validate_candidate_data(candidate_data: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate data."""
    from ..core.validator import PEPValidator
    from ..models.talent_models import Candidate

    try:
        candidate = Candidate(**candidate_data)
        validator = PEPValidator()
        result = validator.validate_candidate(candidate)
        return result.to_dict()
    except Exception as e:
        return {"is_valid": False, "errors": [str(e)], "error_count": 1}


__all__ = ["api_router", "ErrorHandlerMiddleware", "register_error_handlers"]
