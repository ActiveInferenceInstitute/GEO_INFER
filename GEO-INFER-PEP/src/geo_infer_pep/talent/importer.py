"""Talent Data Importers (e.g., from ATS)."""

import logging
import csv
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from datetime import datetime
from ..models.talent_models import (
    Candidate,
    JobRequisition,
    CandidateStatus,
    JobRequisitionStatus,
)

logger = logging.getLogger(__name__)


class BaseTalentImporter(ABC):
    """Abstract base class for Talent data importers."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to the Talent data source (e.g., ATS API)."""
        raise RuntimeError("Talent importer subclasses must implement connect()")

    @abstractmethod
    def fetch_candidates(
        self,
        last_sync_date: datetime | None = None,
        requisition_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw candidate data."""
        raise RuntimeError(
            "Talent importer subclasses must implement fetch_candidates()"
        )

    @abstractmethod
    def transform_candidates(self, raw_data: list[dict[str, Any]]) -> list[Candidate]:
        """Transform raw data into Candidate Pydantic models."""
        raise RuntimeError(
            "Talent importer subclasses must implement transform_candidates()"
        )

    @abstractmethod
    def fetch_requisitions(
        self, last_sync_date: datetime | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch raw job requisition data."""
        raise RuntimeError(
            "Talent importer subclasses must implement fetch_requisitions()"
        )

    @abstractmethod
    def transform_requisitions(
        self, raw_data: list[dict[str, Any]]
    ) -> list[JobRequisition]:
        """Transform raw data into JobRequisition Pydantic models."""
        raise RuntimeError(
            "Talent importer subclasses must implement transform_requisitions()"
        )

    def import_candidates(
        self,
        last_sync_date: datetime | None = None,
        requisition_id: str | None = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        self.connect(**kwargs)
        raw_data = self.fetch_candidates(
            last_sync_date=last_sync_date, requisition_id=requisition_id
        )
        transformed_data = self.transform_candidates(raw_data)
        logger.info(
            f"Imported and transformed {len(transformed_data)} candidate records."
        )
        return transformed_data

    def import_requisitions(
        self,
        last_sync_date: datetime | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> list[JobRequisition]:
        self.connect(**kwargs)
        raw_data = self.fetch_requisitions(last_sync_date=last_sync_date, status=status)
        transformed_data = self.transform_requisitions(raw_data)
        logger.info(
            f"Imported and transformed {len(transformed_data)} job requisition records."
        )
        return transformed_data


# Example: CSVTalentImporter (Now more functional)
class CSVTalentImporter(BaseTalentImporter):
    """Imports Talent data from CSV files."""

    def __init__(
        self,
        candidate_file_path: str | None = None,
        requisition_file_path: str | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        """Configure CSV sources.

        Args:
            candidate_file_path: Path to the candidate CSV file.
            requisition_file_path: Path to the requisition CSV file.
            clock: Source of the current time, used for missing or malformed
                ``applied_at``/``updated_at``/``opened_at`` values.
        """
        self.candidate_file_path = candidate_file_path
        self.requisition_file_path = requisition_file_path
        self.clock = clock
        self.connection = False

    def connect(self, **kwargs: Any) -> None:
        paths = [
            path
            for path in (self.candidate_file_path, self.requisition_file_path)
            if path
        ]
        if not paths:
            raise ValueError("At least one CSV path is required")
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        self.connection = True

    def _read_csv_file(self, file_path: str) -> list[dict[str, Any]]:
        if not self.connection:
            # Attempt to connect if not already. This is a soft connect.
            self.connect()
            if not self.connection:  # If still not connected (e.g. no paths)
                raise ConnectionError(
                    "Not connected. Call connect() or provide file paths."
                )

        if not file_path:
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    records.append(dict(row))
            logger.info(f"Fetched {len(records)} records from {file_path}")
            return records
        except FileNotFoundError:
            logger.error(f"Error: CSV file not found at {file_path}")
            raise  # Re-raise to be caught by tests or higher level logic
        except Exception as e:
            logger.error(f"Error fetching data from CSV file {file_path}: {e}")
            return []

    def fetch_candidates(
        self,
        last_sync_date: datetime | None = None,
        requisition_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.candidate_file_path:
            logger.warning(
                "Warning: Candidate file path not provided for fetch_candidates."
            )
            return []
        # Basic filtering could be added here post-fetch if needed, e.g., by date or req_id
        return self._read_csv_file(self.candidate_file_path)

    def transform_candidates(self, raw_data: list[dict[str, Any]]) -> list[Candidate]:
        candidates: list[Candidate] = []
        now = self.clock()
        for record in raw_data:
            try:
                applied_at_dt = None
                if record.get("applied_at"):
                    try:
                        applied_at_dt = datetime.fromisoformat(record["applied_at"])
                    except ValueError:
                        logger.warning(
                            f"Warn: Bad applied_at for cand {record.get('candidate_id')}"
                        )

                updated_at_dt = None
                if record.get("updated_at"):
                    try:
                        updated_at_dt = datetime.fromisoformat(record["updated_at"])
                    except ValueError:
                        logger.warning(
                            f"Warn: Bad updated_at for cand {record.get('candidate_id')}"
                        )

                status_raw = (record.get("status") or "").strip()

                candidate_data: dict[str, Any] = {
                    "candidate_id": record.get("candidate_id"),
                    "first_name": record.get("first_name"),
                    "last_name": record.get("last_name"),
                    "email": record.get("email"),
                    "phone_number": record.get("phone_number"),
                    "status": (
                        CandidateStatus(status_raw.lower())
                        if status_raw
                        else CandidateStatus.APPLIED
                    ),
                    "job_requisition_id": record.get("job_requisition_id"),
                    "source": record.get("source"),
                    "applied_at": applied_at_dt or now,  # Default if missing/bad
                    "updated_at": updated_at_dt or now,  # Default if missing/bad
                    "skills": (
                        [
                            s.strip()
                            for s in record.get("skills", "").split(",")
                            if s.strip()
                        ]
                        if record.get("skills") is not None
                        else []
                    ),
                    # Add other fields like linkedin_profile, resume_url etc.
                }
                candidate_data_cleaned: dict[str, Any] = {
                    k: v
                    for k, v in candidate_data.items()
                    if v is not None or k in ["applied_at", "updated_at"]
                }
                candidates.append(Candidate(**candidate_data_cleaned))
            except Exception as e:
                logger.info(
                    f"Error transforming candidate record: {record.get('candidate_id', 'Unknown ID')}. Error: {e}"
                )
        logger.info(f"Transformed {len(candidates)} candidate records.")
        return candidates

    def fetch_requisitions(
        self, last_sync_date: datetime | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        if not self.requisition_file_path:
            logger.warning(
                "Warning: Requisition file path not provided for fetch_requisitions."
            )
            return []
        # Basic filtering could be added here post-fetch if needed
        return self._read_csv_file(self.requisition_file_path)

    def transform_requisitions(
        self, raw_data: list[dict[str, Any]]
    ) -> list[JobRequisition]:
        requisitions: list[JobRequisition] = []
        for record in raw_data:
            try:
                opened_at_date = None
                if record.get("opened_at"):
                    try:
                        opened_at_date = datetime.strptime(
                            record["opened_at"], "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        logger.warning(
                            f"Warn: Bad opened_at for req {record.get('requisition_id')}"
                        )

                closed_at_date = None
                if record.get("closed_at"):
                    try:
                        closed_at_date = datetime.strptime(
                            record["closed_at"], "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        logger.warning(
                            f"Warn: Bad closed_at for req {record.get('requisition_id')}"
                        )

                status_raw = (record.get("status") or "").strip()

                req_data: dict[str, Any] = {
                    "requisition_id": record.get("requisition_id"),
                    "job_title": record.get("job_title"),
                    "department": record.get("department"),
                    "status": (
                        JobRequisitionStatus(status_raw.lower())
                        if status_raw
                        else JobRequisitionStatus.OPEN
                    ),
                    "opened_at": opened_at_date
                    or self.clock().date(),  # Default if missing/bad
                    "closed_at": closed_at_date,
                    "hiring_manager_id": record.get("hiring_manager_id"),
                    # Add other fields like location, description, etc.
                }
                req_data_cleaned: dict[str, Any] = {
                    k: v
                    for k, v in req_data.items()
                    if v is not None or k == "closed_at"
                }
                requisitions.append(JobRequisition(**req_data_cleaned))
            except Exception as e:
                logger.info(
                    f"Error transforming requisition record: {record.get('requisition_id', 'Unknown ID')}. Error: {e}"
                )
        logger.info(f"Transformed {len(requisitions)} job requisition records.")
        return requisitions
