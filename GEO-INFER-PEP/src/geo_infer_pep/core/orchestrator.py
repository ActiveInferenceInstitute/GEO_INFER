"""
PEP Orchestrator

This module provides orchestration capabilities for complex PEP workflows,
coordinating between multiple modules and handling multi-step processes.
"""

from typing import Any
from collections.abc import Callable
from datetime import datetime
import logging
from enum import StrEnum

from .pep_engine import PEPEngine
from ..models.talent_models import CandidateStatus

logger = logging.getLogger(__name__)


class WorkflowStatus(StrEnum):
    """Status of workflow execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(
        self,
        name: str,
        description: str,
        step_function: Callable[[dict[str, Any]], Any],
        dependencies: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.step_function = step_function
        self.dependencies = dependencies or []
        self.status = WorkflowStatus.PENDING
        self.result: Any = None
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

    def execute(self, context: dict[str, Any]) -> bool:
        """Execute the workflow step."""
        self.status = WorkflowStatus.IN_PROGRESS
        self.started_at = datetime.now()

        try:
            logger.info(f"Executing workflow step: {self.name}")
            self.result = self.step_function(context)
            self.status = WorkflowStatus.COMPLETED
            self.completed_at = datetime.now()
            logger.info(f"Workflow step completed: {self.name}")
            return True

        except Exception as e:
            logger.error(f"Workflow step failed: {self.name} - {str(e)}")
            self.status = WorkflowStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.now()
            return False


class PEPOrchestrator:
    """
    Orchestrates complex PEP workflows and processes.

    Provides capabilities for:
    - Multi-step workflow execution
    - Dependency management
    - Error handling and recovery
    - Workflow state tracking
    - Complex business process automation
    """

    def __init__(self, pep_engine: PEPEngine | None = None):
        self.engine = pep_engine or PEPEngine()
        self.workflows: dict[str, Any] = {}
        self.active_workflows: dict[str, Any] = {}

    def create_employee_onboarding_workflow(self, candidate_id: str) -> str:
        """Create a comprehensive employee onboarding workflow."""

        workflow_id = (
            f"onboarding_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        workflow = {
            "id": workflow_id,
            "name": "Employee Onboarding",
            "description": f"Complete onboarding process for candidate {candidate_id}",
            "status": WorkflowStatus.PENDING,
            "created_at": datetime.now(),
            "steps": [],
            "context": {"candidate_id": candidate_id},
        }

        # Define workflow steps
        steps = [
            WorkflowStep(
                "validate_candidate",
                "Validate candidate data and eligibility",
                self._validate_candidate_step,
            ),
            WorkflowStep(
                "process_background_check",
                "Initiate background check process",
                self._process_background_check_step,
                dependencies=["validate_candidate"],
            ),
            WorkflowStep(
                "create_employee_record",
                "Create employee record in HR system",
                self._create_employee_record_step,
                dependencies=["validate_candidate"],
            ),
            WorkflowStep(
                "setup_payroll",
                "Setup payroll and compensation",
                self._setup_payroll_step,
                dependencies=["create_employee_record"],
            ),
            WorkflowStep(
                "schedule_training",
                "Schedule initial training and orientation",
                self._schedule_training_step,
                dependencies=["create_employee_record"],
            ),
            WorkflowStep(
                "setup_equipment",
                "Setup workstation and equipment",
                self._setup_equipment_step,
                dependencies=["create_employee_record"],
            ),
            WorkflowStep(
                "send_welcome_package",
                "Send welcome package and communications",
                self._send_welcome_package_step,
                dependencies=["create_employee_record"],
            ),
            WorkflowStep(
                "final_verification",
                "Final verification and workflow completion",
                self._final_verification_step,
                dependencies=[
                    "process_background_check",
                    "setup_payroll",
                    "schedule_training",
                    "setup_equipment",
                    "send_welcome_package",
                ],
            ),
        ]

        workflow["steps"] = steps
        self.workflows[workflow_id] = workflow

        logger.info(
            f"Created onboarding workflow {workflow_id} for candidate {candidate_id}"
        )
        return workflow_id

    def create_bulk_hire_workflow(self, candidate_ids: list[str]) -> str:
        """Create a bulk hiring workflow for multiple candidates."""

        workflow_id = f"bulk_hire_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        workflow = {
            "id": workflow_id,
            "name": "Bulk Hiring Process",
            "description": f"Bulk hiring process for {len(candidate_ids)} candidates",
            "status": WorkflowStatus.PENDING,
            "created_at": datetime.now(),
            "steps": [],
            "context": {"candidate_ids": candidate_ids},
        }

        # Define bulk workflow steps
        steps = [
            WorkflowStep(
                "validate_all_candidates",
                "Validate all candidates in bulk",
                self._validate_bulk_candidates_step,
            ),
            WorkflowStep(
                "prepare_hiring_documents",
                "Prepare bulk hiring documents",
                self._prepare_bulk_documents_step,
                dependencies=["validate_all_candidates"],
            ),
            WorkflowStep(
                "bulk_employee_creation",
                "Create employee records in bulk",
                self._bulk_employee_creation_step,
                dependencies=["prepare_hiring_documents"],
            ),
            WorkflowStep(
                "bulk_system_setup",
                "Setup systems and access in bulk",
                self._bulk_system_setup_step,
                dependencies=["bulk_employee_creation"],
            ),
            WorkflowStep(
                "bulk_communications",
                "Send bulk welcome communications",
                self._bulk_communications_step,
                dependencies=["bulk_system_setup"],
            ),
        ]

        workflow["steps"] = steps
        self.workflows[workflow_id] = workflow

        logger.info(
            f"Created bulk hire workflow {workflow_id} for {len(candidate_ids)} candidates"
        )
        return workflow_id

    def execute_workflow(self, workflow_id: str) -> dict[str, Any]:
        """Execute a workflow by ID."""

        if workflow_id not in self.workflows:
            return {
                "success": False,
                "error": f"Workflow {workflow_id} not found",
                "workflow_id": workflow_id,
            }

        workflow = self.workflows[workflow_id]
        workflow["status"] = WorkflowStatus.IN_PROGRESS
        workflow["started_at"] = datetime.now()

        logger.info(f"Starting execution of workflow {workflow_id}")

        # Execute steps in dependency order
        executed_steps = set()
        max_iterations = len(workflow["steps"]) * 2  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            progress_made = False

            for step in workflow["steps"]:
                if step.status != WorkflowStatus.PENDING:
                    continue

                # Check if all dependencies are completed
                dependencies_met = all(
                    any(
                        s.name == dep and s.status == WorkflowStatus.COMPLETED
                        for s in workflow["steps"]
                    )
                    for dep in step.dependencies
                )

                if dependencies_met:
                    logger.info(f"Executing step {step.name} in workflow {workflow_id}")
                    success = step.execute(workflow["context"])

                    if not success:
                        workflow["status"] = WorkflowStatus.FAILED
                        workflow["failed_step"] = step.name
                        workflow["error"] = step.error

                        return {
                            "success": False,
                            "error": f"Workflow failed at step {step.name}: {step.error}",
                            "workflow_id": workflow_id,
                            "failed_step": step.name,
                        }

                    executed_steps.add(step.name)
                    progress_made = True

            # Check if all steps are completed
            all_completed = all(
                step.status == WorkflowStatus.COMPLETED for step in workflow["steps"]
            )

            if all_completed:
                break

            # Check if we're stuck (no progress made in this iteration)
            if not progress_made and iteration > 1:
                pending_steps = [
                    s for s in workflow["steps"] if s.status == WorkflowStatus.PENDING
                ]
                if pending_steps:
                    workflow["status"] = WorkflowStatus.FAILED
                    return {
                        "success": False,
                        "error": f"Workflow stuck. Pending steps with unmet dependencies: {[s.name for s in pending_steps]}",
                        "workflow_id": workflow_id,
                    }

        workflow["status"] = WorkflowStatus.COMPLETED
        workflow["completed_at"] = datetime.now()

        logger.info(f"Workflow {workflow_id} completed successfully")

        return {
            "success": True,
            "workflow_id": workflow_id,
            "steps_executed": len(executed_steps),
            "total_steps": len(workflow["steps"]),
            "context": workflow["context"],
        }

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get the status of a workflow."""

        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]

        return {
            "success": True,
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"].value,
            "created_at": workflow["created_at"].isoformat(),
            "started_at": workflow.get("started_at", "").isoformat()
            if workflow.get("started_at")
            else None,
            "completed_at": workflow.get("completed_at", "").isoformat()
            if workflow.get("completed_at")
            else None,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "status": step.status.value,
                    "dependencies": step.dependencies,
                    "started_at": step.started_at.isoformat()
                    if step.started_at
                    else None,
                    "completed_at": step.completed_at.isoformat()
                    if step.completed_at
                    else None,
                    "error": step.error,
                }
                for step in workflow["steps"]
            ],
        }

    # Workflow step implementations

    def _validate_candidate_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Validate candidate data and eligibility."""
        candidate_id = context["candidate_id"]

        # Get candidate from data manager
        candidates = self.engine.data_manager.get_candidates(
            {"candidate_id": candidate_id}
        )

        if not candidates:
            raise ValueError(f"candidate {candidate_id} not found")

        candidate = candidates[0]

        # Validation checks
        if candidate.status != CandidateStatus.OFFER_ACCEPTED:
            raise ValueError(
                f"candidate {candidate_id} status is {candidate.status}, expected OFFER_ACCEPTED"
            )

        if not candidate.email:
            raise ValueError(f"candidate {candidate_id} missing email address")

        return {
            "candidate_id": candidate_id,
            "validation_status": "passed",
            "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        }

    def _process_background_check_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process background check (simplified implementation)."""
        candidate_id = context["candidate_id"]

        # In real implementation, this would integrate with background check service
        logger.info(f"Processing background check for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "background_check_status": "initiated",
            "estimated_completion": "3-5 business days",
        }

    def _create_employee_record_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Create employee record in HR system."""
        candidate_id = context["candidate_id"]

        # Use the existing onboarding workflow
        from ..methods import process_employee_onboarding_workflow

        employee_data = {"candidate_id": candidate_id}
        success = process_employee_onboarding_workflow(employee_data)

        if not success:
            raise RuntimeError(
                f"Failed to create employee record for candidate {candidate_id}"
            )

        return {"candidate_id": candidate_id, "employee_record_created": True}

    def _setup_payroll_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Setup payroll and compensation."""
        candidate_id = context["candidate_id"]

        # In real implementation, this would integrate with payroll system
        logger.info(f"Setting up payroll for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "payroll_setup_status": "completed",
            "compensation_processed": True,
        }

    def _schedule_training_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Schedule training and orientation."""
        candidate_id = context["candidate_id"]

        # In real implementation, this would integrate with LMS
        logger.info(f"Scheduling training for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "training_scheduled": True,
            "orientation_date": "Next Monday",
            "training_modules": [
                "Company Overview",
                "Security Training",
                "Product Training",
            ],
        }

    def _setup_equipment_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Setup workstation and equipment."""
        candidate_id = context["candidate_id"]

        # In real implementation, this would integrate with IT service management
        logger.info(f"Setting up equipment for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "equipment_setup_status": "ordered",
            "items": ["Laptop", "Monitor", "Headphones", "Access Card"],
            "estimated_delivery": "2 business days",
        }

    def _send_welcome_package_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Send welcome package and communications."""
        candidate_id = context["candidate_id"]

        # In real implementation, this would integrate with email/communication system
        logger.info(f"Sending welcome package for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "welcome_email_sent": True,
            "welcome_package_sent": True,
            "communication_items": [
                "Welcome Email",
                "New Hire Portal Access",
                "Benefits Guide",
            ],
        }

    def _final_verification_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Final verification and workflow completion."""
        candidate_id = context["candidate_id"]

        # Verify all previous steps completed successfully
        logger.info(f"Final verification for candidate {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "verification_status": "completed",
            "onboarding_status": "complete",
            "start_date": "Next Monday",
        }

    # Bulk workflow step implementations

    def _validate_bulk_candidates_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Validate all candidates in bulk."""
        candidate_ids = context["candidate_ids"]

        valid_candidates = []
        invalid_candidates = []

        for candidate_id in candidate_ids:
            try:
                candidates = self.engine.data_manager.get_candidates(
                    {"candidate_id": candidate_id}
                )
                if (
                    candidates
                    and candidates[0].status == CandidateStatus.OFFER_ACCEPTED
                ):
                    valid_candidates.append(candidate_id)
                else:
                    invalid_candidates.append(candidate_id)
            except Exception:
                invalid_candidates.append(candidate_id)

        if invalid_candidates:
            raise ValueError(f"Invalid candidates: {invalid_candidates}")

        return {
            "valid_candidates": valid_candidates,
            "total_validated": len(valid_candidates),
        }

    def _prepare_bulk_documents_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Prepare bulk hiring documents."""
        candidate_ids = context["candidate_ids"]

        # In real implementation, this would generate bulk documents
        logger.info(f"Preparing documents for {len(candidate_ids)} candidates")

        return {
            "documents_prepared": len(candidate_ids),
            "document_types": ["Offer Letters", "Tax Forms", "Benefits Enrollment"],
        }

    def _bulk_employee_creation_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Create employee records in bulk."""
        candidate_ids = context["candidate_ids"]

        success_count = 0
        failed_count = 0

        for candidate_id in candidate_ids:
            try:
                employee_data = {"candidate_id": candidate_id}
                from ..methods import process_employee_onboarding_workflow

                success = process_employee_onboarding_workflow(employee_data)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        return {
            "employees_created": success_count,
            "failed_creations": failed_count,
            "success_rate": success_count / len(candidate_ids),
        }

    def _bulk_system_setup_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Setup systems and access in bulk."""
        candidate_ids = context["candidate_ids"]

        # In real implementation, this would setup multiple systems
        logger.info(f"Setting up systems for {len(candidate_ids)} candidates")

        return {
            "systems_setup": len(candidate_ids),
            "systems": [
                "Email",
                "HR Portal",
                "Project Management",
                "Development Tools",
            ],
        }

    def _bulk_communications_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Send bulk welcome communications."""
        candidate_ids = context["candidate_ids"]

        # In real implementation, this would send bulk emails
        logger.info(f"Sending communications to {len(candidate_ids)} candidates")

        return {
            "communications_sent": len(candidate_ids),
            "communication_types": [
                "Welcome Email",
                "Portal Access",
                "Start Date Confirmation",
            ],
        }
