"""HR Data Importers."""

import logging
import csv
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime
from ..models.hr_models import (
    Employee,
    EmploymentStatus,
    Gender,
)  # Adjusted import path

logger = logging.getLogger(__name__)


class BaseHRImporter(ABC):
    """Abstract base class for HR data importers."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to the HR data source."""
        raise RuntimeError("HR importer subclasses must implement connect()")

    @abstractmethod
    def fetch_employees(
        self, last_sync_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch raw employee data from the HR source."""
        raise RuntimeError("HR importer subclasses must implement fetch_employees()")

    @abstractmethod
    def transform_employees(self, raw_data: list[dict[str, Any]]) -> list[Employee]:
        """Transform raw employee data into Employee Pydantic models."""
        raise RuntimeError(
            "HR importer subclasses must implement transform_employees()"
        )

    def import_employees(
        self, last_sync_date: datetime | None = None, **kwargs: Any
    ) -> list[Employee]:
        """Orchestrates the import process: connect, fetch, transform for employees."""
        self.connect(**kwargs)
        raw_data = self.fetch_employees(last_sync_date=last_sync_date)
        transformed_data = self.transform_employees(raw_data)
        logger.info(
            f"Successfully imported and transformed {len(transformed_data)} employee records."
        )
        return transformed_data


# Example: CSVHRImporter (Now functional for basic fields)
class CSVHRImporter(BaseHRImporter):
    """Imports HR data from a CSV file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.connection: str | None = None
        logger.info(f"CSV HR Importer initialized for file: {self.file_path}")

    def connect(self, **kwargs: Any) -> None:
        try:
            # Check accessibility
            with open(self.file_path, encoding="utf-8") as f:
                f.read(0)
            self.connection = "connected"
            logger.info(f"Successfully connected to HR CSV file: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"Error: HR CSV file not found at {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to HR CSV file {self.file_path}: {e}")
            raise

    def fetch_employees(
        self, last_sync_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        if not self.connection:
            raise ConnectionError("Not connected to HR CSV file. Call connect() first.")

        records: list[dict[str, Any]] = []
        try:
            with open(self.file_path, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Add date filtering if last_sync_date and relevant date column in CSV
                    records.append(dict(row))
            logger.info(
                f"Fetched {len(records)} employee records from {self.file_path}"
            )
            return records
        except Exception as e:
            logger.error(f"Error fetching employee data from CSV {self.file_path}: {e}")
            return []

    def transform_employees(self, raw_data: list[dict[str, Any]]) -> list[Employee]:
        employees: list[Employee] = []
        for record in raw_data:
            try:
                # Basic date parsing, assuming YYYY-MM-DD format
                hire_dt = None
                if record.get("hire_date"):
                    try:
                        hire_dt = datetime.strptime(
                            record["hire_date"], "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        logger.warning(
                            f"Warning: Could not parse hire_date for record: {record.get('employee_id')}"
                        )

                status_raw = (record.get("status") or "").strip()
                gender_raw = (record.get("gender") or "").strip()

                employee_data: dict[str, Any] = {
                    "employee_id": record.get("employee_id"),
                    "first_name": record.get("first_name"),
                    "last_name": record.get("last_name"),
                    "email": record.get("email"),
                    "hire_date": hire_dt,
                    "employment_status": (
                        EmploymentStatus(status_raw.lower())
                        if status_raw
                        else EmploymentStatus.ACTIVE
                    ),
                    "job_title": record.get("job_title"),
                    "department": record.get("department"),
                    "gender": (Gender(gender_raw.lower()) if gender_raw else None),
                    # Add other fields as necessary from your CSV
                }
                # Filter out None values for fields that are optional and not provided
                employee_data_cleaned: dict[str, Any] = {
                    k: v
                    for k, v in employee_data.items()
                    if v is not None or k in ["hire_date"]
                }

                # Pydantic will validate
                employees.append(Employee(**employee_data_cleaned))
            except Exception as e:
                logger.info(
                    f"Error transforming HR record: {record.get('employee_id', 'Unknown ID')}. Error: {e}"
                )
        logger.info(f"Transformed {len(employees)} employee records.")
        return employees


# Future importers could include:
# - BambooHRImporter
# - WorkdayImporter
# - OtherHRISImporter
