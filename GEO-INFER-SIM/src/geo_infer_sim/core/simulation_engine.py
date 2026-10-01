"""
Core simulation engine for GEO-INFER-SIM.

This module provides the foundational simulation engine that supports
multiple simulation paradigms including ABM, system dynamics, and CA.
"""

import logging
import threading
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, UTC
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimulationState(StrEnum):
    """Simulation execution states."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""

    time_step: float = 1.0
    max_time: float = 100.0
    output_interval: float = 1.0
    random_seed: int | None = None
    save_state_history: bool = True
    checkpoint_interval: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
        if self.output_interval <= 0:
            raise ValueError("output_interval must be positive")


class SimulationEngine:
    """
    Core simulation engine for geospatial simulations.

    Provides a unified interface for running simulations across different
    paradigms with state management, event scheduling, and result collection.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """
        Initialize the simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.state = SimulationState.INITIALIZED
        self.current_time = 0.0
        self._current_state: dict[str, Any] = {}
        self.state_history: list[dict[str, Any]] = []
        self.metrics: dict[str, list[float]] = {}
        self.events: list[dict[str, Any]] = []

        # Pause signaling: the run loop honors this event between steps so a
        # pause can never surface as a step error (and thus a FAILED state).
        self._pause_requested = threading.Event()

        # Seeded RNG for stochastic step functions; deterministic-by-default
        # when config.random_seed is set.
        self.rng: np.random.Generator = np.random.default_rng(self.config.random_seed)

    def initialize(self, initial_state: dict[str, Any]) -> None:
        """
        Initialize the simulation with initial state.

        Args:
            initial_state: Dictionary containing initial simulation state
        """
        logger.info("Initializing simulation")

        self.current_time = 0.0
        self.state = SimulationState.INITIALIZED
        self._current_state = initial_state.copy()
        self.state_history = []
        self.metrics = {}
        self.events = []

        # Store initial state
        if self.config.save_state_history:
            self.state_history.append(
                {
                    "time": self.current_time,
                    "state": initial_state.copy(),
                }
            )

        logger.info("Simulation initialized")

    def step(
        self, step_func: Callable[[float, dict[str, Any]], dict[str, Any]]
    ) -> None:
        """
        Execute a single simulation step.

        Args:
            step_func: Function that takes (time, current_state) and returns new_state
        """
        if (
            self.state != SimulationState.RUNNING
            and self.state != SimulationState.INITIALIZED
        ):
            raise ValueError(f"Cannot step simulation in state: {self.state}")

        if self.state == SimulationState.INITIALIZED:
            self.state = SimulationState.RUNNING

        # Get current state
        current_state = self._current_state.copy()

        # Execute step function
        try:
            new_state = step_func(self.current_time, current_state)
            if not isinstance(new_state, dict):
                raise TypeError("step_func must return a state dictionary")
            self._current_state = new_state.copy()

            # Update time
            self.current_time += self.config.time_step

            # Save state if needed
            if self.config.save_state_history:
                if (
                    len(self.state_history) == 0
                    or self.current_time
                    >= self.state_history[-1]["time"] + self.config.output_interval
                ):
                    self.state_history.append(
                        {
                            "time": self.current_time,
                            "state": new_state.copy(),
                        }
                    )

        except Exception as e:
            logger.error(f"Simulation step failed at time {self.current_time}: {e}")
            self.state = SimulationState.FAILED
            raise

    def run(
        self, step_func: Callable[[float, dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Run the complete simulation.

        Args:
            step_func: Function that takes (time, current_state) and returns new_state

        Returns:
            Simulation results dictionary
        """
        logger.info(f"Starting simulation (max_time={self.config.max_time})")

        self.state = SimulationState.RUNNING
        self._pause_requested.clear()
        start_time = datetime.now(UTC).replace(tzinfo=None)

        try:
            while self.current_time < self.config.max_time:
                if self.state == SimulationState.CANCELLED:
                    break

                if self._pause_requested.is_set():
                    break

                self.step(step_func)

            if self.state == SimulationState.RUNNING:
                self.state = SimulationState.COMPLETED
            end_time = datetime.now(UTC).replace(tzinfo=None)
            duration = (end_time - start_time).total_seconds()

            results = {
                "status": self.state.value,
                "final_time": self.current_time,
                "duration_seconds": duration,
                "state_history": self.state_history,
                "metrics": self.metrics,
                "events": self.events,
            }

            logger.info(
                f"Simulation completed in {duration:.2f}s "
                f"(final_time={self.current_time})"
            )

            return results

        except Exception as e:
            self.state = SimulationState.FAILED
            logger.error(f"Simulation failed: {e}")
            raise

    def pause(self) -> None:
        """Pause the simulation at the next step boundary."""
        if self.state == SimulationState.RUNNING:
            self._pause_requested.set()
            self.state = SimulationState.PAUSED
            logger.info("Simulation paused")

    def resume(self) -> None:
        """Resume a paused simulation from the paused step."""
        if self.state == SimulationState.PAUSED:
            self._pause_requested.clear()
            self.state = SimulationState.RUNNING
            logger.info("Simulation resumed")

    def cancel(self) -> None:
        """Cancel the simulation."""
        self.state = SimulationState.CANCELLED
        logger.info("Simulation cancelled")

    def get_state(self) -> dict[str, Any]:
        """
        Get current simulation state.

        Returns:
            Current state dictionary
        """
        return {
            "state": self.state.value,
            "current_time": self.current_time,
            "config": {
                "time_step": self.config.time_step,
                "max_time": self.config.max_time,
                "output_interval": self.config.output_interval,
            },
        }

    def record_metric(self, name: str, value: float) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)

    def record_event(
        self, event_type: str, time: float, data: dict[str, Any] | None = None
    ) -> None:
        """
        Record a simulation event.

        Args:
            event_type: Type of event
            time: Event time
            data: Optional event data
        """
        self.events.append(
            {
                "type": event_type,
                "time": time,
                "data": data or {},
            }
        )

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save simulation checkpoint to file.

        Args:
            filepath: Path to save checkpoint
        """
        import json

        checkpoint = {
            "current_time": self.current_time,
            "state": self.state.value,
            "state_history": self.state_history,
            "metrics": self.metrics,
            "events": self.events,
            "config": {
                "time_step": self.config.time_step,
                "max_time": self.config.max_time,
                "output_interval": self.config.output_interval,
                "random_seed": self.config.random_seed,
            },
            "rng_state": self.rng.bit_generator.state,
        }

        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load simulation checkpoint from file.

        Args:
            filepath: Path to checkpoint file
        """
        import json

        with open(filepath) as f:
            checkpoint = json.load(f)

        self.current_time = checkpoint["current_time"]
        self.state = SimulationState(checkpoint["state"])
        self.state_history = checkpoint["state_history"]
        self.metrics = checkpoint["metrics"]
        self.events = checkpoint["events"]

        # Restore the embedded configuration and the exact RNG stream state so
        # a resumed run is bit-reproducible from the checkpoint.
        saved_config = checkpoint.get("config")
        if saved_config is not None:
            self.config = SimulationConfig(
                time_step=saved_config["time_step"],
                max_time=saved_config["max_time"],
                output_interval=saved_config["output_interval"],
                random_seed=saved_config["random_seed"],
            )

        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        else:
            # Legacy checkpoint without an RNG snapshot: fall back to seeding
            # from the saved random_seed (deterministic, but restarts stream).
            self.rng = np.random.default_rng(
                self.config.random_seed if saved_config else None
            )

        logger.info(f"Checkpoint loaded from {filepath}")

    def export_results(self, format: str = "dataframe") -> Any:
        """
        Export simulation results in various formats.

        Args:
            format: Export format ('dataframe', 'dict', 'json')

        Returns:
            Exported results
        """
        if format == "dataframe":
            # Create DataFrame from state history
            records = []
            for entry in self.state_history:
                record = {"time": entry["time"]}
                record.update(entry["state"])
                records.append(record)

            return pd.DataFrame(records)

        elif format == "json":
            import json

            return json.dumps(
                {
                    "state_history": self.state_history,
                    "metrics": self.metrics,
                    "events": self.events,
                },
                indent=2,
                default=str,
            )

        elif format == "dict":
            return {
                "state_history": self.state_history,
                "metrics": self.metrics,
                "events": self.events,
                "current_time": self.current_time,
                "status": self.state.value,
            }

        raise ValueError(
            f"Unsupported export format: {format!r}; expected 'dataframe', "
            "'dict', or 'json'"
        )

    def get_metric_statistics(self, metric_name: str) -> dict[str, Any]:
        """
        Get statistics for a recorded metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Statistical summary
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {"error": "Metric not found or empty"}

        values = self.metrics[metric_name]

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "median": float(np.median(values)),
            "sum": float(sum(values)),
            "first": values[0],
            "last": values[-1],
            "trend": (
                "increasing"
                if values[-1] > values[0]
                else "decreasing"
                if values[-1] < values[0]
                else "stable"
            ),
        }
