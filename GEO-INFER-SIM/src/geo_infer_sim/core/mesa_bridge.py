"""
Mesa-backed simulation bridge for GEO-INFER-SIM.

This module provides :class:`MesaModelBridge`, which wraps a ``mesa.Model``
instance so that it can be driven through the same
:class:`~geo_infer_sim.core.simulation_engine.SimulationEngine` interface used
by the deterministic engine. Mesa is an *optional* dependency: the bridge module
imports cleanly without Mesa installed (``HAS_MESA`` is ``False``) and only
raises when a :class:`MesaModelBridge` is actually constructed.

The bridge is responsible for:

* advancing the wrapped Mesa model one ``step()`` at a time,
* snapshotting the model state into :attr:`state_history` after each step,
* pulling model-level metrics from a ``mesa.DataCollector`` (when present) and
  recording them through :meth:`record_metric`,
* preserving :class:`SimulationState.CANCELLED` / :class:`SimulationState.FAILED`
  states so callers can distinguish a clean stop from a completed run, and
* exposing the same JSON / DataFrame / dict export surface as the base engine.
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd

from geo_infer_sim.core.simulation_engine import (
    SimulationConfig,
    SimulationEngine,
    SimulationState,
)

logger = logging.getLogger(__name__)


# --- Optional Mesa import ----------------------------------------------------
#
# We probe for Mesa exactly once at module import time and expose the result as
# ``HAS_MESA``. The module itself must remain importable when Mesa is absent so
# that the rest of geo_infer_sim (and its tests) can load without the optional
# dependency installed. The bridge only fails when someone actually tries to
# *construct* a MesaModelBridge without Mesa available.

try:  # pragma: no cover - import probe
    import mesa as _mesa

    HAS_MESA: bool = True
    _MesaModel = _mesa.Model
    _MesaDataCollector = _mesa.DataCollector
except ImportError:  # pragma: no cover - exercised when mesa is missing
    _mesa = None
    HAS_MESA = False
    _MesaModel = None
    _MesaDataCollector = None


# Type aliases for clarity (kept permissive since Mesa is optional).
StateExtractor = Callable[[Any], dict[str, Any]]
MetricExtractors = dict[str, Callable[[Any], float]]


class MesaModelBridge(SimulationEngine):
    """Wrap a ``mesa.Model`` in the :class:`SimulationEngine` interface.

    The bridge drives the Mesa model forward one ``step()`` at a time, maps
    each Mesa step onto a ``time_step`` increment in the parent engine, and
    records state/metric history so that the standard
    :meth:`~SimulationEngine.export_results` and
    :meth:`~SimulationEngine.save_checkpoint` machinery works unchanged.

    Parameters
    ----------
    model:
        A ``mesa.Model`` instance (or any object exposing ``step()``,
        ``running``, ``steps``, and ``time`` attributes). The bridge does not
        take ownership of the model's lifecycle; callers may reuse it.
    config:
        Simulation configuration. ``time_step`` is the amount of simulation
        time credited per Mesa ``step()``; ``max_time`` bounds the run. If
        ``None`` a default :class:`SimulationConfig` is used.
    state_extractor:
        Optional callable ``(model) -> dict`` snapshotting the model state
        after each step. When omitted, the bridge tries (in order) the
        model's ``DataCollector`` model-level reporters, then a generic
        ``{steps, time, num_agents}`` snapshot.
    metric_extractors:
        Optional mapping ``{metric_name: (model) -> float}`` recording custom
        metrics after each step. When the model carries a ``DataCollector`` its
        model-level reporters are also recorded automatically (the
        ``DataCollector`` takes precedence for names that collide).
    """

    def __init__(
        self,
        model: Any,
        config: SimulationConfig | None = None,
        state_extractor: StateExtractor | None = None,
        metric_extractors: MetricExtractors | None = None,
    ) -> None:
        """Initialize the Mesa-backed bridge.

        Raises
        ------
        ImportError
            If Mesa is not installed (``HAS_MESA`` is ``False``).
        TypeError
            If ``model`` does not look like a Mesa model (missing ``step``).
        """
        if not HAS_MESA:
            raise ImportError(
                "MesaModelBridge requires the optional 'mesa' dependency. "
                "Install it via `uv sync --extra mesa` (GEO-INFER-SIM) or "
                "`uv pip install -e ./GEO-INFER-SIM[mesa]`."
            )
        if model is None or not hasattr(model, "step") or not callable(model.step):
            raise TypeError(
                "MesaModelBridge expects a Mesa Model (or duck-typed object) "
                f"with a callable step(); got {type(model).__name__!r}."
            )

        super().__init__(config=config)
        self.model = model
        self._state_extractor: StateExtractor = (
            state_extractor or self._default_state_extractor
        )
        self._metric_extractors: MetricExtractors = dict(metric_extractors or {})

    # ------------------------------------------------------------------
    # Default extractors
    # ------------------------------------------------------------------
    def _default_state_extractor(self, model: Any) -> dict[str, Any]:
        """Snapshot model state, preferring DataCollector model reporters."""
        collector = getattr(model, "datacollector", None)
        if isinstance(collector, _MesaDataCollector):
            try:
                df = collector.get_model_vars_dataframe()
                if df is not None and not df.empty:
                    row = df.iloc[-1].to_dict()
                    # Coerce numpy scalars to plain Python types for JSON.
                    return {k: _to_jsonable(v) for k, v in row.items()}
                # DataCollector is empty (no steps collected yet). Compute the
                # reporter values directly so the initial snapshot has the same
                # schema as post-step snapshots.
                reporters = getattr(collector, "model_reporters", None) or {}
                if reporters:
                    snapshot: dict[str, Any] = {}
                    for name, reporter in reporters.items():
                        try:
                            snapshot[str(name)] = _to_jsonable(reporter(model))
                        except Exception:  # pragma: no cover - defensive
                            logger.debug(
                                "Reporter %r failed during initial snapshot",
                                name,
                                exc_info=True,
                            )
                    # Always include bookkeeping fields for consistency.
                    snapshot.setdefault("steps", int(getattr(model, "steps", 0)))
                    snapshot.setdefault("time", float(getattr(model, "time", 0.0)))
                    return snapshot
            except Exception:  # pragma: no cover - defensive
                logger.debug("DataCollector model vars unavailable", exc_info=True)
        # Generic fallback: expose bookkeeping fields every Mesa Model has.
        return {
            "steps": int(getattr(model, "steps", 0)),
            "time": float(getattr(model, "time", 0.0)),
            "num_agents": len(getattr(model, "agents", []) or []),
            "running": bool(getattr(model, "running", True)),
        }

    def _collect_metrics(self) -> None:
        """Record metrics from the DataCollector and custom extractors."""
        collector = getattr(self.model, "datacollector", None)
        if isinstance(collector, _MesaDataCollector):
            try:
                df = collector.get_model_vars_dataframe()
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    for name, value in last.items():
                        try:
                            self.record_metric(str(name), float(value))
                        except (TypeError, ValueError):
                            # Skip non-numeric reporter values silently.
                            logger.debug(
                                "Skipping non-numeric DataCollector reporter %r=%r",
                                name,
                                value,
                            )
            except Exception:  # pragma: no cover - defensive
                logger.debug("DataCollector metric pull failed", exc_info=True)
        # Custom extractors override / extend DataCollector metrics.
        for name, extractor in self._metric_extractors.items():
            try:
                self.record_metric(name, float(extractor(self.model)))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Metric extractor %r raised: %s", name, exc)

    # ------------------------------------------------------------------
    # SimulationEngine interface
    # ------------------------------------------------------------------
    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        """Initialize the bridge.

        ``initial_state`` is optional for Mesa-backed runs because the model
        already carries its own state; when provided it is stored verbatim as
        the first history entry (mirroring the base engine contract).
        """
        logger.info("Initializing Mesa-backed simulation")
        self.current_time = 0.0
        self.state = SimulationState.INITIALIZED
        self.state_history = []
        self.metrics = {}
        self.events = []

        snapshot = (
            initial_state.copy()
            if initial_state is not None
            else self._state_extractor(self.model)
        )
        self._current_state = snapshot.copy()

        if self.config.save_state_history:
            self.state_history.append(
                {"time": self.current_time, "state": snapshot.copy()}
            )
        logger.info("Mesa-backed simulation initialized")

    def step(self, step_func: Callable[..., Any] | None = None) -> None:
        """Advance the Mesa model by one ``step()``.

        ``step_func`` is accepted for API parity with
        :meth:`SimulationEngine.step` but is **not** used to drive the model:
        the Mesa model's own ``step()`` is the source of truth. Passing a
        non-None value logs a warning.
        """
        if step_func is not None:
            logger.warning(
                "MesaModelBridge.step ignores the supplied step_func; the "
                "Mesa model's step() is the execution primitive."
            )

        if self.state not in (SimulationState.INITIALIZED, SimulationState.RUNNING):
            raise ValueError(f"Cannot step simulation in state: {self.state}")

        if self.state == SimulationState.INITIALIZED:
            self.state = SimulationState.RUNNING

        try:
            self.model.step()
        except Exception as exc:
            logger.error(
                "Mesa model step failed at time %s: %s", self.current_time, exc
            )
            self.state = SimulationState.FAILED
            raise

        # Advance engine time by the configured time_step per Mesa step.
        self.current_time += self.config.time_step

        # Pull metrics first (DataCollector is populated during model.step).
        self._collect_metrics()

        # Snapshot state for history.
        new_state = self._state_extractor(self.model)
        self._current_state = new_state.copy()
        if self.config.save_state_history:
            if (
                len(self.state_history) == 0
                or self.current_time
                >= self.state_history[-1]["time"] + self.config.output_interval
            ):
                self.state_history.append(
                    {"time": self.current_time, "state": new_state.copy()}
                )

    def run(self, step_func: Callable[..., Any] | None = None) -> dict[str, Any]:
        """Run the Mesa model until ``max_time`` or the model stops.

        The loop terminates when *any* of the following holds:

        * ``self.current_time >= self.config.max_time`` (normal completion),
        * ``self.state == SimulationState.CANCELLED`` (caller cancelled), or
        * the Mesa model's ``running`` flag is ``False`` (model self-stopped).

        On completion the result dict matches :meth:`SimulationEngine.run`,
        including the preserved ``status`` field.
        """
        if step_func is not None:
            logger.warning(
                "MesaModelBridge.run ignores the supplied step_func; the "
                "Mesa model's step() is the execution primitive."
            )

        logger.info(
            "Starting Mesa-backed simulation (max_time=%s)", self.config.max_time
        )
        # If initialize() was never called, seed the initial snapshot now so
        # that bare run() produces a complete history (initial + per-step
        # entries). This mirrors the ergonomic expectation for a Mesa model
        # that already carries its own state, while preserving the base
        # engine's contract when initialize() is called explicitly.
        if self.state == SimulationState.INITIALIZED and not self.state_history:
            self.initialize()
        if self.state == SimulationState.INITIALIZED:
            self.state = SimulationState.RUNNING
        start_time = datetime.now(UTC).replace(tzinfo=None)

        try:
            while self.current_time < self.config.max_time:
                if self.state == SimulationState.CANCELLED:
                    break
                # Honor a model that flips its own `running` flag to False.
                if getattr(self.model, "running", True) is False:
                    logger.info(
                        "Mesa model set running=False; stopping at time %s",
                        self.current_time,
                    )
                    break
                # step() may set state to FAILED or CANCELLED.
                self.step()

            if self.state not in (SimulationState.CANCELLED, SimulationState.FAILED):
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
                # Mesa-specific extras for downstream consumers.
                "mesa_steps": int(getattr(self.model, "steps", 0)),
                "mesa_time": float(getattr(self.model, "time", 0.0)),
                "mesa_running": bool(getattr(self.model, "running", True)),
            }

            logger.info(
                "Mesa-backed simulation finished in %.2fs (final_time=%s, status=%s)",
                duration,
                self.current_time,
                self.state.value,
            )
            return results

        except Exception as exc:
            self.state = SimulationState.FAILED
            logger.error("Mesa-backed simulation failed: %s", exc)
            raise

    def cancel(self) -> None:
        """Cancel the simulation and signal the Mesa model to stop."""
        self.state = SimulationState.CANCELLED
        # Polite: ask the Mesa model to stop as well so an in-flight step can
        # observe the flag on its next iteration if it polls `running`.
        if getattr(self.model, "running", None) is not None:
            self.model.running = False
        logger.info("Mesa-backed simulation cancelled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-serializable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, dict, tuple)):
        return value
    # Catch pandas/numpy NaN and other NA sentinels without misclassifying
    # containers (pd.isna on a list/dict returns an array, not a bool).
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
