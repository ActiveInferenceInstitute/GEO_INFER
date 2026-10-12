"""Explicit event-time contracts at the active inference boundary."""

from datetime import UTC
import pytest
from geo_infer_time.core.inference_schedule import inference_schedule


def test_epoch_and_equivalent_offsets():
    result = inference_schedule(
        ["1970-01-01T00:00:00Z", "1969-12-31T16:01:00-08:00"], step_seconds=60
    )
    assert result[0].timestamp() == 0
    assert all(t.tzinfo == UTC for t in result)
    assert result[1].timestamp() == 60


@pytest.mark.parametrize(
    "times",
    [
        [],
        ["2026-01-01T00:00:00"],
        ["2026-01-01T00:00:00Z"] * 2,
        ["2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"],
        ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"],
    ],
)
def test_invalid_schedules(times):
    with pytest.raises(ValueError):
        inference_schedule(times, step_seconds=60)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, 1e-8])
def test_invalid_step(seconds):
    with pytest.raises(ValueError):
        inference_schedule(["2026-01-01T00:00:00Z"], step_seconds=seconds)


def test_bounded_iterable():
    def endless():
        while True:
            yield "2026-01-01T00:00:00Z"

    with pytest.raises(ValueError, match="max_steps"):
        inference_schedule(endless(), step_seconds=1, max_steps=1)
