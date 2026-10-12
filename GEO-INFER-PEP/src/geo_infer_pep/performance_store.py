"""In-memory performance-review store for GEO-INFER-PEP.

This module follows the ``core/data_store.py`` singleton pattern: a single
process-wide store shared by the FastAPI ``geo_infer_pep.api`` layer (and
available to the ``geo_infer_pep.methods`` layer). ``PerformanceReview``
instances created directly remain instance-scoped for isolated use (e.g.
tests).
"""

import logging

from .models.hr_models import PerformanceReview

logger = logging.getLogger(__name__)


class PerformanceReviewStore:
    """In-memory store of performance reviews keyed by employee."""

    def __init__(self) -> None:
        self.reviews_by_employee: dict[str, list[PerformanceReview]] = {}

    def add(self, employee_id: str, review: PerformanceReview) -> None:
        """Persist ``review`` under ``employee_id``."""
        self.reviews_by_employee.setdefault(employee_id, []).append(review)
        logger.info(
            "Stored performance review %s for employee %s",
            review.review_id,
            employee_id,
        )

    def list_for_employee(self, employee_id: str) -> list[PerformanceReview]:
        """Return a copy of the reviews for ``employee_id`` (empty if unknown)."""
        return list(self.reviews_by_employee.get(employee_id, []))

    def clear(self) -> None:
        """Remove all stored reviews."""
        self.reviews_by_employee.clear()


#: Process-wide shared store used by the FastAPI layer in
#: ``geo_infer_pep.api`` so performance-review data persists across requests.
performance_review_store = PerformanceReviewStore()
