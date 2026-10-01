"""
Core PEP Engine

This module provides the central engine for orchestrating all PEP (People, Engagement, Performance)
operations. It serves as the main entry point for the PEP system and coordinates between
different modules and data flows.
"""

from typing import Any
from datetime import datetime
import logging

from ..models.hr_models import Employee, EmploymentStatus
from .data_store import PEPDataManager, pep_data_manager  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)


class PEPEngine:
    """
    Main PEP Engine

    Provides high-level operations and orchestrates complex workflows across
    all PEP modules (HR, CRM, Talent).

    By default the engine operates on the process-wide shared store
    (``core.data_store.pep_data_manager``) so it sees the same data as the
    ``geo_infer_pep.methods`` layer and the FastAPI endpoints. Pass an explicit
    ``data_manager`` to isolate the engine (e.g. in tests).
    """

    def __init__(self, data_manager: PEPDataManager | None = None):
        self.data_manager = (
            data_manager if data_manager is not None else pep_data_manager
        )
        self._initialized = False

        logger.info("PEP Engine initialized")

    def initialize(self) -> bool:
        """Initialize the PEP engine."""
        if self._initialized:
            logger.warning("PEP Engine already initialized")
            return True

        try:
            self._initialized = True
            logger.info("PEP Engine successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PEP Engine: {str(e)}")
            return False

    def import_hr_data(self, file_path: str) -> dict[str, Any]:
        """Import HR data from CSV file."""
        try:
            from ..methods import import_hr_data_from_csv

            # import_hr_data_from_csv already stores records in the shared
            # in-memory store; do not add them again to avoid duplicates.
            employees = import_hr_data_from_csv(file_path)

            return {
                "success": True,
                "records_imported": len(employees),
                "data_type": "employees",
                "file_path": file_path,
            }
        except Exception as e:
            logger.error(f"HR data import failed: {str(e)}")
            return {"success": False, "error": str(e), "data_type": "employees"}

    def import_crm_data(self, file_path: str) -> dict[str, Any]:
        """Import CRM data from CSV file."""
        try:
            from ..methods import import_crm_data_from_csv

            # import_crm_data_from_csv already stores records in the shared
            # in-memory store; do not add them again to avoid duplicates.
            customers = import_crm_data_from_csv(file_path)

            return {
                "success": True,
                "records_imported": len(customers),
                "data_type": "customers",
                "file_path": file_path,
            }
        except Exception as e:
            logger.error(f"CRM data import failed: {str(e)}")
            return {"success": False, "error": str(e), "data_type": "customers"}

    def import_talent_data(
        self, candidates_file: str, requisitions_file: str
    ) -> dict[str, Any]:
        """Import talent data from CSV files."""
        try:
            from ..methods import import_talent_data_from_csv

            result = import_talent_data_from_csv(candidates_file, requisitions_file)

            if result.get("processed_successfully"):
                # Imported candidates and requisitions are already stored in
                # the shared in-memory store by the methods layer.
                return {
                    "success": True,
                    "candidates_imported": result.get("candidates", 0),
                    "requisitions_imported": result.get("requisitions", 0),
                    "data_type": "talent",
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "data_type": "talent",
                }

        except Exception as e:
            logger.error(f"Talent data import failed: {str(e)}")
            return {"success": False, "error": str(e), "data_type": "talent"}

    def process_onboarding_workflow(
        self, employee_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process employee onboarding workflow."""
        try:
            from ..methods import process_employee_onboarding_workflow

            success = process_employee_onboarding_workflow(employee_data)

            # The workflow already stores the new employee in the shared
            # in-memory store; no direct data-manager mutation needed.
            return {
                "success": success,
                "workflow": "onboarding",
                "candidate_id": employee_data.get("candidate_id"),
                "employee_data": employee_data,
            }

        except Exception as e:
            logger.error(f"Onboarding workflow failed: {str(e)}")
            return {"success": False, "error": str(e), "workflow": "onboarding"}

    def generate_hr_dashboard(self) -> dict[str, Any]:
        """Generate comprehensive HR dashboard."""
        try:
            from ..methods import generate_comprehensive_hr_dashboard

            dashboard = generate_comprehensive_hr_dashboard()
            return {"success": True, "dashboard_type": "hr", "data": dashboard}
        except Exception as e:
            logger.error(f"HR dashboard generation failed: {str(e)}")
            return {"success": False, "error": str(e), "dashboard_type": "hr"}

    def generate_crm_dashboard(self) -> dict[str, Any]:
        """Generate comprehensive CRM dashboard."""
        try:
            from ..methods import generate_comprehensive_crm_dashboard

            dashboard = generate_comprehensive_crm_dashboard()
            return {"success": True, "dashboard_type": "crm", "data": dashboard}
        except Exception as e:
            logger.error(f"CRM dashboard generation failed: {str(e)}")
            return {"success": False, "error": str(e), "dashboard_type": "crm"}

    def generate_talent_dashboard(self) -> dict[str, Any]:
        """Generate comprehensive talent dashboard."""
        try:
            from ..methods import generate_comprehensive_talent_dashboard

            dashboard = generate_comprehensive_talent_dashboard()
            return {"success": True, "dashboard_type": "talent", "data": dashboard}
        except Exception as e:
            logger.error(f"Talent dashboard generation failed: {str(e)}")
            return {"success": False, "error": str(e), "dashboard_type": "talent"}

    def generate_all_dashboards(self) -> dict[str, Any]:
        """Generate all dashboards and return combined results."""
        results: dict[str, Any] = {
            "overall_success": True,
            "dashboards": {},
            "generated_at": datetime.now().isoformat(),
        }

        # Generate each dashboard
        hr_result = self.generate_hr_dashboard()
        crm_result = self.generate_crm_dashboard()
        talent_result = self.generate_talent_dashboard()

        results["dashboards"]["hr"] = hr_result
        results["dashboards"]["crm"] = crm_result
        results["dashboards"]["talent"] = talent_result

        # Check if any failed
        if not all(
            [hr_result["success"], crm_result["success"], talent_result["success"]]
        ):
            results["overall_success"] = False
            results["errors"] = []

            for name, result in [
                ("HR", hr_result),
                ("CRM", crm_result),
                ("Talent", talent_result),
            ]:
                if not result["success"]:
                    results["errors"].append(
                        f"{name}: {result.get('error', 'Unknown error')}"
                    )

        return results

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        data_summary = self.data_manager.get_data_summary()

        return {
            "system_status": "operational",
            "initialized": self._initialized,
            "data_summary": data_summary,
            "engine_version": "1.0.0",
            "last_check": datetime.now().isoformat(),
        }

    def run_health_check(self) -> dict[str, Any]:
        """Run comprehensive health check."""
        health_status: dict[str, Any] = {
            "overall_health": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check data integrity
        data_summary = self.data_manager.get_data_summary()
        health_status["checks"]["data_integrity"] = {
            "status": "healthy",
            "details": f"Data store contains {data_summary['employees']['total']} employees, {data_summary['customers']['total']} customers, {data_summary['candidates']['total']} candidates",
        }

        # Check module availability
        try:
            import pandas as pd  # noqa: F401

            health_status["checks"]["dependencies"] = {
                "status": "healthy",
                "details": "All required dependencies available",
            }
        except ImportError as e:
            health_status["checks"]["dependencies"] = {
                "status": "unhealthy",
                "details": f"Missing dependency: {str(e)}",
            }
            health_status["overall_health"] = "unhealthy"

        # Check data processing capabilities
        try:
            test_data = Employee(
                employee_id="test001",
                first_name="Test",
                last_name="User",
                email="test@example.com",
                employment_status=EmploymentStatus.ACTIVE,
                job_title="Test Role",
                department="Test Dept",
            )
            self.data_manager.add_employees([test_data])
            health_status["checks"]["data_processing"] = {
                "status": "healthy",
                "details": "Data processing functions operational",
            }
        except Exception as e:
            health_status["checks"]["data_processing"] = {
                "status": "unhealthy",
                "details": f"Data processing error: {str(e)}",
            }
            health_status["overall_health"] = "unhealthy"

        return health_status

    def shutdown(self) -> bool:
        """Shutdown the PEP engine gracefully.

        Clears the engine's data store. With the default shared store this
        wipes every employee, customer, candidate, and requisition record
        visible to the methods and API layers.
        """
        try:
            logger.info("Shutting down PEP Engine")
            self.data_manager.clear_all_data()
            self._initialized = False
            return True
        except Exception as e:
            logger.error(f"Error during PEP Engine shutdown: {str(e)}")
            return False
