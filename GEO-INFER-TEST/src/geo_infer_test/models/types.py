from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class TestOutcome:
    """Test outcome with detailed information.

    Renamed from ``TestResult``: the canonical ``TestResult`` in
    ``core.test_runner`` is the runner's execution record; this dataclass is
    the outcome shape used by model-level reporting.
    """

    test_name: str
    passed: bool
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    category: str = "general"


@dataclass
class ValidationRule:
    """Data validation rule."""

    name: str
    field: str
    rule_type: str  # range, format, custom
    parameters: dict[str, Any]
    severity: str = "error"  # error, warning, info
    description: str = ""
