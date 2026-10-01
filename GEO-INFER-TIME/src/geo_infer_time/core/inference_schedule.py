"""Explicit fixed-interval UTC schedules for sequential inference."""

from datetime import datetime, timedelta, UTC
from itertools import islice
import math
from collections.abc import Iterable


def inference_schedule(
    timestamps: Iterable[str | datetime],
    *,
    step_seconds: float,
    max_steps: int = 10_000,
) -> tuple[datetime, ...]:
    """Validate aware timestamps, order and exact model-step alignment.

    Missing intervals require an explicit model prediction and therefore raise;
    the function never fills gaps, sorts observations or discards duplicates.
    Offsets are normalized to UTC before comparing instants.
    """
    if (
        isinstance(step_seconds, bool)
        or not isinstance(step_seconds, (int, float))
        or not math.isfinite(step_seconds)
        or step_seconds <= 0
    ):
        raise ValueError("step_seconds must be finite and positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    interval = timedelta(seconds=step_seconds)
    if interval.total_seconds() != step_seconds:
        raise ValueError("step_seconds must be representable at microsecond precision")
    result = []
    for timestamp in islice(timestamps, max_steps + 1):
        if len(result) == max_steps:
            raise ValueError("Inference schedule exceeds max_steps")
        instant = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if isinstance(timestamp, str)
            else timestamp
        )
        if (
            not isinstance(instant, datetime)
            or instant.tzinfo is None
            or instant.utcoffset() is None
        ):
            raise ValueError("Inference timestamps must include a timezone")
        instant = instant.astimezone(UTC)
        if result and instant - result[-1] != interval:
            raise ValueError("Timestamps must be ordered and separated by step_seconds")
        result.append(instant)
    if not result:
        raise ValueError("Inference schedule must not be empty")
    return tuple(result)
