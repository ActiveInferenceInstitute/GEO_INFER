"""CRM Data Importers."""

import logging
import csv
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime
from ..models.crm_models import (
    Customer,
    Address,
    InteractionLog,
)  # Adjusted import path

logger = logging.getLogger(__name__)


class BaseCRMImporter(ABC):
    """Abstract base class for CRM importers."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to the CRM data source."""
        raise RuntimeError("CRM importer subclasses must implement connect()")

    @abstractmethod
    def fetch_data(
        self, last_sync_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch raw data from the CRM."""
        raise RuntimeError("CRM importer subclasses must implement fetch_data()")

    @abstractmethod
    def transform_data(self, raw_data: list[dict[str, Any]]) -> list[Customer]:
        """Transform raw data into Customer Pydantic models."""
        raise RuntimeError("CRM importer subclasses must implement transform_data()")

    def import_customers(
        self, last_sync_date: datetime | None = None, **kwargs: Any
    ) -> list[Customer]:
        """Orchestrates the import process: connect, fetch, transform."""
        self.connect(**kwargs)
        raw_data = self.fetch_data(last_sync_date=last_sync_date)
        transformed_data = self.transform_data(raw_data)
        logger.info(
            f"Successfully imported and transformed {len(transformed_data)} customer records."
        )
        return transformed_data


class CSVCRMImporter(BaseCRMImporter):
    """Imports CRM data from a CSV file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.connection: str | None = None
        logger.info(f"CSV CRM Importer initialized for file: {self.file_path}")

    def connect(self, **kwargs: Any) -> None:
        """Open and validate access to the CSV file."""
        try:
            # In a real scenario, you might keep the file open or check its existence.
            with open(self.file_path, encoding="utf-8") as f:
                f.read(0)
            self.connection = "connected"
            logger.info(f"Successfully connected to CSV file: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"Error: CSV file not found at {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to CSV file {self.file_path}: {e}")
            raise

    def fetch_data(
        self, last_sync_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Reads data from the CSV file."""
        if not self.connection:
            raise ConnectionError("Not connected to CSV file. Call connect() first.")

        records: list[dict[str, Any]] = []
        try:
            with open(self.file_path, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Basic filtering by date if applicable
                    # This assumes a 'updated_at' or similar column in the CSV
                    if last_sync_date and "updated_at" in row:
                        try:
                            record_date = datetime.fromisoformat(row["updated_at"])
                            if record_date <= last_sync_date:
                                continue
                        except ValueError:
                            # Row cannot be date-filtered reliably, so it is
                            # skipped: incremental syncs must not append rows
                            # whose recency is unknown.
                            logger.warning(
                                "Skipping row with unparseable updated_at during incremental fetch: %s",
                                row.get("id", row),
                            )
                            continue
                    records.append(dict(row))
            logger.info(f"Fetched {len(records)} records from {self.file_path}")
            return records
        except Exception as e:
            logger.error(f"Error fetching data from CSV file {self.file_path}: {e}")
            return []

    def transform_data(self, raw_data: list[dict[str, Any]]) -> list[Customer]:
        """Transforms CSV rows into Customer Pydantic models."""
        customers: list[Customer] = []
        for record in raw_data:
            try:
                address = Address(
                    street=record.get("address_street"),
                    city=record.get("address_city"),
                    state=record.get("address_state"),
                    postal_code=record.get("address_postal_code"),
                    country=record.get("address_country"),
                )

                # Example: simple interaction log from a notes field (highly simplified)
                interactions = []
                if record.get("notes"):
                    interactions.append(
                        InteractionLog(summary=record["notes"], channel="csv_import")
                    )

                customer_data = {
                    "customer_id": record.get(
                        "id", record.get("customer_id", f"csv-{hash(str(record))}")
                    ),
                    "first_name": record.get("first_name"),
                    "last_name": record.get(
                        "last_name", "N/A"
                    ),  # last_name is mandatory in model
                    "email": record.get("email"),
                    "phone_number": record.get("phone"),
                    "company": record.get("company_name"),
                    "job_title": record.get("title"),
                    "address": address,
                    "created_at": (
                        datetime.fromisoformat(record["created_at"])
                        if record.get("created_at")
                        else datetime.now()
                    ),
                    "updated_at": (
                        datetime.fromisoformat(record["updated_at"])
                        if record.get("updated_at")
                        else datetime.now()
                    ),
                    "source": record.get("lead_source", "CSV Import"),
                    "status": record.get("status", "active"),
                    "tags": (
                        record.get("tags", "").split(",") if record.get("tags") else []
                    ),
                    "interaction_history": interactions,
                    "notes": record.get("notes_detail"),
                }
                # Pydantic will validate the data
                customers.append(Customer(**customer_data))
            except Exception as e:
                # Log the error and problematic record, then continue if possible
                logger.error(f"Error transforming record: {record}. Error: {e}")
                # Optionally, add to an error list or re-raise if critical
        return customers
