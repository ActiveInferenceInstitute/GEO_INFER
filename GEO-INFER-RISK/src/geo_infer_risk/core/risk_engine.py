"""
Enhanced RiskEngine: Advanced orchestrator for comprehensive risk modeling and analysis.

This module provides an enhanced RiskEngine class that serves as the main entry point
for running sophisticated risk analyses, coordinating interactions between multiple
hazard, vulnerability, and exposure models while integrating with other GEO-INFER modules.

Key enhancements:
- Integration with GEO-INFER-SPACE for advanced spatial analysis
- Integration with GEO-INFER-TIME for temporal dynamics
- Integration with GEO-INFER-AI for machine learning applications
- Integration with GEO-INFER-MATH for advanced statistical methods
- Integration with GEO-INFER-BAYES for Bayesian inference
- Advanced Monte Carlo simulation with convergence criteria
- Real-time risk monitoring and streaming analytics
- Portfolio optimization and risk aggregation
- Climate change scenario integration
- Comprehensive uncertainty quantification
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any, cast
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

# GEO-INFER module imports with error handling
try:
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
    from geo_infer_space.core.dispatcher import configure_backends
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

try:
    from geo_infer_time.core.temporal_analysis import TemporalAnalysisInterface

    TIME_AVAILABLE = True
except ImportError:
    TIME_AVAILABLE = False
    TemporalAnalysisInterface = None

try:
    from geo_infer_math.core.interpolation import (  # type: ignore[import-untyped]
        InterpolationManager,
    )
    from geo_infer_math.core.spatial_statistics import (  # type: ignore[import-untyped]
        GearysC,
        MoranI,
    )

    MATH_AVAILABLE = True
except ImportError:
    MATH_AVAILABLE = False
    GearysC = None
    MoranI = None
    InterpolationManager = None

try:
    from geo_infer_bayes.core.inference import (  # type: ignore[import-untyped]
        BayesianInference,
    )

    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    BayesianInference = None

# Local imports
from geo_infer_risk.core.catastrophe_models import (
    CatastropheModelManager,
    MultiHazardInteractionMatrix,
)
from geo_infer_risk.core.exposure_model import EnhancedExposureModel
from geo_infer_risk.core.hazard_model import EnhancedHazardModel
from geo_infer_risk.core.insurance_models import InsuranceManager
from geo_infer_risk.core.vulnerability_model import EnhancedVulnerabilityModel
from geo_infer_risk.utils.config_loader import load_config_with_defaults
from geo_infer_risk.utils.rng import resolve_rng
from geo_infer_risk.utils.validation import validate_config


@dataclass
class AnalysisJob:
    """Represents an analysis job with metadata and status."""

    job_id: str
    job_type: str
    status: str = "queued"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    results: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelIntegrationStatus:
    """Status of integration with external GEO-INFER modules."""

    space_integration: bool = SPACE_AVAILABLE
    time_integration: bool = TIME_AVAILABLE
    math_integration: bool = MATH_AVAILABLE
    bayes_integration: bool = BAYES_AVAILABLE
    spatial_indexing_available: bool = False
    temporal_analysis_available: bool = False
    advanced_statistics_available: bool = False
    bayesian_inference_available: bool = False


class EnhancedRiskEngine:
    """
    Enhanced risk analysis engine with advanced capabilities and module integration.

    The EnhancedRiskEngine provides:
    - Integration with all GEO-INFER modules (SPACE, TIME, AI, MATH, BAYES)
    - Advanced Monte Carlo simulation with convergence monitoring
    - Real-time risk monitoring and streaming analytics
    - Portfolio optimization and risk aggregation
    - Climate change scenario analysis
    - Comprehensive uncertainty quantification
    - Asynchronous processing capabilities
    - Model calibration and validation
    - Advanced spatial and temporal analysis
    """

    # Runtime-integration singletons; typed Optional[Any] so mypy does not
    # collapse them to None (they are populated at runtime if modules present).
    spatial_interface: Any | None = None
    temporal_interface: Any | None = None

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the EnhancedRiskEngine with comprehensive configuration.

        Args:
            config: Configuration dictionary. If None, loads from default locations.
        """
        # Load configuration with defaults
        if config is None:
            config = load_config_with_defaults()

        self.config = config
        validation_result = validate_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")

        self.random_seed = config.get("risk_model", {}).get(
            "random_seed", config.get("general", {}).get("random_seed")
        )
        # All stochastic draws come from self.rng; resolve_rng owns seed
        # validation and never touches the numpy.random singleton.
        self.rng: np.random.Generator = resolve_rng(self.random_seed)
        self._file_handler: logging.FileHandler | None = None
        self._closed = False
        # Logging needs the output directory before it creates its file handler.
        # Resolve and create it here so initialization is deterministic and the
        # first startup does not emit a misleading "output_dir" warning.
        self.output_dir = config.get("general", {}).get("output_directory", "./outputs")
        self.cache_dir = config.get("general", {}).get("cache_directory", "./cache")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = self._setup_enhanced_logging()

        # Initialize integration status
        self.integration_status = self._check_module_integrations()

        # Initialize core components
        self._initialize_core_components()

        # Initialize model containers
        self.hazard_models: dict[str, Any] = {}
        self.vulnerability_models: dict[str, Any] = {}
        self.exposure_models: dict[str, Any] = {}
        self.catastrophe_manager = CatastropheModelManager()
        self.insurance_manager = InsuranceManager()

        # Initialize results and job management
        self.event_losses = None
        self.aggregated_metrics = None
        self.active_jobs: dict[str, AnalysisJob] = {}
        self.job_counter = 0

        # Initialize spatial and temporal interfaces
        self._initialize_spatial_interface()
        self._initialize_temporal_interface()

        # Initialize threading and async support
        self.executor = ThreadPoolExecutor(
            max_workers=config.get("general", {}).get("num_workers", 4)
        )
        self.async_enabled = config.get("general", {}).get("enable_async", True)

        self.logger.info(
            "EnhancedRiskEngine initialized successfully with %d module integrations",
            len(self.integration_status.__dict__),
        )

    def __enter__(self) -> "EnhancedRiskEngine":
        """Return the engine as a context manager with deterministic cleanup."""
        if self._closed:
            raise RuntimeError("Cannot enter a closed EnhancedRiskEngine")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Shut down worker resources when leaving a context manager."""
        self.close()

    def close(self) -> None:
        """Release worker and file-handler resources; safe to call repeatedly."""
        if self._closed:
            return
        self.executor.shutdown(wait=True)
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        self._closed = True

    def _check_module_integrations(self) -> ModelIntegrationStatus:
        """Check availability of external module integrations."""
        status = ModelIntegrationStatus()

        # Check SPACE integration
        if SPACE_AVAILABLE:
            try:
                # Try to configure backends
                configure_backends(
                    {"default_backends": {"indexing": "h3", "analytics": "srai"}}
                )
                SpatialIndexingInterface()
                SpatialAnalyticsInterface()
                status.spatial_indexing_available = True
                status.space_integration = True
            except Exception as e:
                self.logger.warning(f"SPACE integration check failed: {e}")
                status.space_integration = False

        # Check TIME integration
        if TIME_AVAILABLE:
            try:
                # Try to initialize temporal analysis
                TemporalAnalysisInterface()
                status.temporal_analysis_available = True
                status.time_integration = True
            except Exception as e:
                self.logger.warning(f"TIME integration check failed: {e}")
                status.time_integration = False

        # Check MATH integration
        if MATH_AVAILABLE:
            try:
                MoranI()
                InterpolationManager()
                status.advanced_statistics_available = True
                status.math_integration = True
            except Exception as e:
                self.logger.warning(f"MATH integration check failed: {e}")
                status.math_integration = False

        # Check BAYES integration
        if BAYES_AVAILABLE:
            try:
                status.bayesian_inference_available = all(
                    callable(getattr(BayesianInference, method, None))
                    for method in ("__init__", "run", "update")
                )
                if not status.bayesian_inference_available:
                    raise TypeError(
                        "BayesianInference does not expose the required API"
                    )
                status.bayes_integration = True
            except Exception as e:
                self.logger.warning(f"BAYES integration check failed: {e}")
                status.bayes_integration = False

        return status

    def _initialize_core_components(self) -> None:
        """Initialize core risk analysis components."""
        # Initialize catastrophe and insurance managers
        self.catastrophe_manager = CatastropheModelManager()
        self.insurance_manager = InsuranceManager()

        # Initialize spatial and temporal interfaces if available
        self.spatial_interface = None
        self.temporal_interface = None
        self.math_interface = None
        self.bayes_interface = None

    def _initialize_spatial_interface(self) -> None:
        """Initialize spatial analysis interface."""
        if self.integration_status.space_integration:
            try:
                self.spatial_interface = SpatialIndexingInterface()
                self.spatial_analytics = SpatialAnalyticsInterface()
                self.logger.info("Spatial interface initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize spatial interface: {e}")
                self.spatial_interface = None

    def _initialize_temporal_interface(self) -> None:
        """Initialize temporal analysis interface."""
        if self.integration_status.time_integration:
            try:
                self.temporal_interface = TemporalAnalysisInterface()
                self.logger.info("Temporal interface initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize temporal interface: {e}")
                self.temporal_interface = None

    def _ensure_open(self) -> None:
        """Reject work submitted after the engine has released its resources."""
        if self._closed:
            raise RuntimeError("EnhancedRiskEngine is closed")

    def _setup_enhanced_logging(self) -> logging.Logger:
        """Set up enhanced logging with structured output."""
        log_level = self.config.get("general", {}).get("log_level", "INFO")
        level = getattr(logging, log_level)

        logger = logging.getLogger("geo_infer_risk_enhanced")
        logger.setLevel(level)

        # Create formatter with more detailed information
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        )

        # Reuse handlers when multiple engines are created in one process.
        # This prevents duplicate log lines and repeated file descriptors.
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler if output directory is available
        log_file = os.path.abspath(os.path.join(self.output_dir, "risk_engine.log"))
        if not any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == log_file
            for handler in logger.handlers
        ):
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                self._file_handler = file_handler
            except OSError as exc:
                logger.warning("Could not create log file %s: %s", log_file, exc)

        return logger

    def get_integration_status(self) -> dict[str, bool]:
        """Get status of all module integrations."""
        return {
            "space_integration": self.integration_status.space_integration,
            "time_integration": self.integration_status.time_integration,
            "math_integration": self.integration_status.math_integration,
            "bayes_integration": self.integration_status.bayes_integration,
            "spatial_indexing": self.integration_status.spatial_indexing_available,
            "temporal_analysis": self.integration_status.temporal_analysis_available,
            "advanced_statistics": self.integration_status.advanced_statistics_available,
            "bayesian_inference": self.integration_status.bayesian_inference_available,
        }

    def run_enhanced_analysis(
        self, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Run enhanced risk analysis with advanced capabilities.

        Args:
            analysis_type: Type of analysis ('comprehensive', 'portfolio', 'climate', 'stress_test')
            **kwargs: Additional analysis parameters

        Returns:
            Comprehensive risk analysis results
        """
        self._ensure_open()
        self.logger.info(f"Starting enhanced {analysis_type} risk analysis")

        job_id = self._create_analysis_job(analysis_type, **kwargs)

        try:
            if analysis_type == "comprehensive":
                results = self._run_comprehensive_analysis(**kwargs)
            elif analysis_type == "portfolio":
                results = self._run_portfolio_analysis(**kwargs)
            elif analysis_type == "climate":
                results = self._run_climate_analysis(**kwargs)
            elif analysis_type == "stress_test":
                results = self._run_stress_test(**kwargs)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            # Update job status
            self._update_job_status(job_id, "completed", results=results)

            self.logger.info(
                f"Enhanced {analysis_type} analysis completed successfully"
            )
            return results

        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            self._update_job_status(job_id, "failed", error_message=str(e))
            raise

    def _run_comprehensive_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run comprehensive multi-hazard risk analysis."""
        # Load and validate models
        self._load_configured_models()

        # Run spatial analysis if available
        spatial_results: dict[str, Any] = {}
        if self.spatial_interface:
            spatial_results = self._run_spatial_analysis(**kwargs)

        # Run temporal analysis if available
        temporal_results: dict[str, Any] = {}
        if self.temporal_interface:
            temporal_results = self._run_temporal_analysis(**kwargs)

        # Report the configured core models. Monte Carlo simulation is opt-in so
        # an analysis without event and exposure data cannot present fabricated
        # losses as measured risk.
        core_results = self._run_core_analysis(**kwargs)

        # Combine results
        combined_results = {
            "core_analysis": core_results,
            "spatial_analysis": spatial_results,
            "temporal_analysis": temporal_results,
            "integration_metadata": self.get_integration_status(),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return combined_results

    def _run_core_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Summarize configured models and optionally run their loss simulation."""
        results: dict[str, Any] = {
            "analysis_type": "core",
            "model_status": self.get_model_status(),
        }
        if kwargs.get("run_monte_carlo", False):
            results["monte_carlo"] = self.run_monte_carlo_analysis(
                num_iterations=int(kwargs.get("monte_carlo_iterations", 1000)),
                convergence_threshold=kwargs.get("convergence_threshold", 0.01),
            )
        return results

    def _run_spatial_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run advanced spatial analysis using GEO-INFER-SPACE."""
        if not self.spatial_interface:
            return {}

        region = kwargs.get("region", {})
        if not region:
            return {}

        try:
            # Convert region to spatial format
            bounds = region.get("bounds", {})
            if bounds:
                min_lon, max_lon = (
                    bounds.get("min_lon", -180),
                    bounds.get("max_lon", 180),
                )
                min_lat, max_lat = bounds.get("min_lat", -90), bounds.get("max_lat", 90)

                # Create spatial analysis region
                spatial_region = {
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                }

                # Run spatial analytics
                if hasattr(self.spatial_analytics, "analyze_risk_concentration"):
                    concentration_analysis = (
                        self.spatial_analytics.analyze_risk_concentration(
                            region=spatial_region,
                            resolution=kwargs.get("spatial_resolution", 9),
                        )
                    )
                else:
                    concentration_analysis = {}

                # Run spatial statistics if available
                if self.integration_status.advanced_statistics_available:
                    statistical_analysis = self._run_spatial_statistics(spatial_region)
                else:
                    statistical_analysis = {}

                return {
                    "concentration_analysis": concentration_analysis,
                    "statistical_analysis": statistical_analysis,
                    "spatial_indexing": "h3" if self.spatial_interface else "none",
                }

            return {}

        except Exception as e:
            self.logger.warning(f"Spatial analysis failed: {e}")
            return {}

    def _run_spatial_statistics(self, region: dict[str, Any]) -> dict[str, Any]:
        """Run spatial statistics using GEO-INFER-MATH."""
        if not self.integration_status.advanced_statistics_available:
            return {}

        try:
            # Extract location pairs from region features
            coords = []
            values = []
            for feat in region.get("features", []):
                loc = feat.get("location", {})
                if "latitude" in loc and "longitude" in loc:
                    coords.append([loc["latitude"], loc["longitude"]])
                    values.append(feat.get("risk_value", 0.0))

            if len(coords) < 3:
                return {
                    "spatial_autocorrelation": 0.0,
                    "morans_i": 0.0,
                    "geary_c": 0.0,
                    "local_indicators": [],
                }

            coords_arr = np.array(coords)
            vals = np.array(values, dtype=float)
            dev = vals - np.mean(vals)

            # Delegate to GEO-INFER-MATH's spatial statistics when the
            # integrations extra is installed; otherwise fall back to an
            # inline inverse-distance implementation with the same outputs.
            if MATH_AVAILABLE and MoranI is not None and GearysC is not None:
                moran_result = MoranI().compute(vals, coords_arr)
                geary_result = GearysC().compute(vals, coords_arr)
                morans_i = float(moran_result["I"])
                geary_c = float(geary_result["C"])
                return {
                    "spatial_autocorrelation": morans_i,
                    "morans_i": morans_i,
                    "geary_c": geary_c,
                    "local_indicators": [float(d) for d in dev[:10]],
                    "source": "geo_infer_math",
                }

            n = len(vals)

            # Distance-based spatial weights (inverse distance)
            from scipy.spatial.distance import pdist, squareform

            dists = squareform(pdist(coords_arr))
            np.fill_diagonal(dists, np.inf)
            W = 1.0 / dists
            np.fill_diagonal(W, 0.0)
            W_sum = W.sum()

            # Moran's I
            morans_num = n * np.sum(W * np.outer(dev, dev))
            morans_den = W_sum * np.sum(dev**2)
            morans_i = morans_num / morans_den if morans_den != 0 else 0.0

            # Geary's C
            geary_num = (n - 1) * np.sum(W * (np.subtract.outer(vals, vals) ** 2))
            geary_den = 2 * W_sum * np.sum(dev**2)
            geary_c = geary_num / geary_den if geary_den != 0 else 0.0

            return {
                "spatial_autocorrelation": float(morans_i),
                "morans_i": float(morans_i),
                "geary_c": float(geary_c),
                "local_indicators": [float(d) for d in dev[:10]],
                "source": "inline_fallback",
            }
        except ImportError:
            self.logger.warning("scipy not available for spatial statistics")
            return {
                "spatial_autocorrelation": 0.0,
                "morans_i": 0.0,
                "geary_c": 0.0,
                "local_indicators": [],
            }
        except Exception as e:
            self.logger.warning(f"Spatial statistics failed: {e}")
            return {}

    def _run_temporal_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run temporal analysis using GEO-INFER-TIME."""
        if not self.temporal_interface:
            return {}

        try:
            time_horizon = kwargs.get("time_horizon", 50)

            # Analyse loss history to extract seasonal and trend components
            history = kwargs.get("loss_history", [])
            if not history:
                return {
                    "seasonal_patterns": {},
                    "trend_analysis": {},
                    "time_series_decomposition": {},
                    "forecast_scenarios": [],
                }

            values = np.array([h.get("value", 0) for h in history], dtype=float)
            # Trend via simple linear regression
            x = np.arange(len(values), dtype=float)
            if len(values) > 1:
                slope = np.polyfit(x, values, 1)[0]
            else:
                slope = 0.0

            # Seasonal: group by month index (mod 12)
            seasonal: dict[int, list[Any]] = {}
            for i, v in enumerate(values):
                month = i % 12
                seasonal.setdefault(month, []).append(v)
            seasonal_means = {k: float(np.mean(v)) for k, v in seasonal.items()}

            return {
                "seasonal_patterns": seasonal_means,
                "trend_analysis": {
                    "slope": float(slope),
                    "direction": "increasing" if slope > 0 else "decreasing",
                },
                "time_series_decomposition": {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                },
                "forecast_scenarios": [
                    {
                        "horizon": time_horizon,
                        "projected_mean": float(np.mean(values) + slope * time_horizon),
                    }
                ],
            }
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {e}")
            return {}

    def _create_analysis_job(self, job_type: str, **kwargs: Any) -> str:
        """Create a new analysis job."""
        self.job_counter += 1
        job_id = f"analysis_{self.job_counter}_{int(time.time())}"

        job = AnalysisJob(
            job_id=job_id, job_type=job_type, status="queued", metadata=kwargs
        )

        self.active_jobs[job_id] = job
        return job_id

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        progress: float | None = None,
        results: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job status and progress."""
        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]
        job.status = status

        if progress is not None:
            job.progress = progress

        if status == "running" and job.started_at is None:
            job.started_at = datetime.now()
        elif status in ["completed", "failed"]:
            job.completed_at = datetime.now()

        if error_message:
            job.error_message = error_message

        if results:
            job.results = results

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of an analysis job."""
        if job_id not in self.active_jobs:
            return None

        job = self.active_jobs[job_id]
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "metadata": job.metadata,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running analysis job."""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        if job.status in ["running"]:
            job.status = "cancelled"
            job.completed_at = datetime.now()
            return True

        return False

    def get_model_status(self) -> dict[str, Any]:
        """Get status of all loaded models."""
        return {
            "hazard_models": list(self.hazard_models.keys()),
            "vulnerability_models": list(self.vulnerability_models.keys()),
            "exposure_models": list(self.exposure_models.keys()),
            "catastrophe_manager": hasattr(self.catastrophe_manager, "models"),
            "insurance_manager": hasattr(self.insurance_manager, "models"),
            "integration_status": self.get_integration_status(),
            "loaded_models_count": len(self.hazard_models)
            + len(self.vulnerability_models)
            + len(self.exposure_models),
        }

    def configure_hazard_interactions(
        self,
        hazards: Sequence[str],
        interaction_matrix: Sequence[Sequence[float]],
    ) -> MultiHazardInteractionMatrix:
        """Configure directed multi-hazard interactions for compound analysis."""
        self._ensure_open()
        return self.catastrophe_manager.configure_hazard_interactions(
            hazards, interaction_matrix
        )

    def set_hazard_interaction(self, source: str, target: str, strength: float) -> None:
        """Set one directed source-to-target hazard interaction."""
        self._ensure_open()
        self.catastrophe_manager.set_hazard_interaction(source, target, strength)

    def calculate_compound_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequence: Sequence[str] | None = None,
    ) -> float:
        """Estimate joint exceedance along a configured compound-hazard path."""
        self._ensure_open()
        return self.catastrophe_manager.calculate_compound_exceedance_probability(
            exceedance_probabilities, hazard_sequence
        )

    def calculate_joint_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequences: Sequence[Sequence[str]],
    ) -> float:
        """Estimate the union exceedance across several compound paths."""
        self._ensure_open()
        return self.catastrophe_manager.joint_exceedance_probability(
            exceedance_probabilities, hazard_sequences
        )

    def get_branch_exceedance_probabilities(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> dict[tuple[str, ...], float]:
        """Enumerate directed compound paths and their joint probabilities."""
        self._ensure_open()
        return self.catastrophe_manager.branch_exceedance_probabilities(
            exceedance_probabilities, max_path_length
        )

    def get_dominant_hazard_path(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> tuple[tuple[str, ...], float]:
        """Return the compound path with the largest joint exceedance."""
        self._ensure_open()
        return self.catastrophe_manager.dominant_exceedance_path(
            exceedance_probabilities, max_path_length
        )

    def calibrate_models(
        self, calibration_data: dict[str, Any], method: str = "cross_validation"
    ) -> dict[str, Any]:
        """
        Calibrate model parameters using historical data.

        Args:
            calibration_data: Historical data for calibration
            method: Only 'cross_validation' is implemented and accepted.
                'bayesian' and 'maximum_likelihood' are deliberately rejected:
                a generic Bayesian calibration needs a model-specific
                BayesianModel adapter with explicit likelihood and prior per
                configured risk component, and a maximum-likelihood fit needs
                a model-specific likelihood adapter; no such adapters exist,
                so options that would hard-fail are not advertised.

        Returns:
            Calibration results and updated parameters
        """
        self._ensure_open()
        method = method.lower()
        if method != "cross_validation":
            raise ValueError(
                "method must be 'cross_validation' — the only implemented "
                "calibration method. 'bayesian' is not accepted because a "
                "generic Bayesian calibration requires a model-specific "
                "BayesianModel adapter (explicit likelihood and prior per "
                "configured risk component), and 'maximum_likelihood' "
                "requires a model-specific likelihood adapter; no such "
                "adapters exist yet"
            )
        self.logger.info("Starting model calibration using cross_validation method")
        return self._calibrate_with_cross_validation(calibration_data)

    def _calibrate_with_cross_validation(
        self, calibration_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Fit and cross-validate an empirical mean-loss baseline.

        The generic risk engine cannot safely mutate heterogeneous hazard,
        vulnerability, or exposure models without a model-specific adapter.
        This method therefore fits the parameters its input contract actually
        identifies: the mean and sample standard deviation of finite observed
        losses. Held-out folds evaluate a mean-loss predictor fitted only on
        each fold's training samples; the returned calibrated parameters are
        then fitted once on the complete supplied calibration set.
        """
        if not isinstance(calibration_data, dict):
            raise TypeError("calibration_data must be a mapping")
        samples = calibration_data.get("samples", [])
        if len(samples) < 2:
            raise ValueError("at least two calibration samples are required")
        if not all(
            isinstance(sample, dict)
            and "loss" in sample
            and np.isfinite(float(sample["loss"]))
            for sample in samples
        ):
            raise ValueError("calibration samples must contain finite loss values")
        losses = np.asarray([float(sample["loss"]) for sample in samples], dtype=float)
        k = min(5, max(1, len(samples)))
        folds: list[dict[str, Any]] = []
        for i in range(k):
            train_values = losses[np.arange(len(losses)) % k != i]
            test_values = losses[np.arange(len(losses)) % k == i]
            if train_values.size and test_values.size:
                training_mean = float(np.mean(train_values))
                mse = float(np.mean((test_values - training_mean) ** 2))
                folds.append(
                    {
                        "fold": i,
                        "training_sample_count": int(train_values.size),
                        "validation_sample_count": int(test_values.size),
                        "fitted_loss_mean": training_mean,
                        "validation_mse": mse,
                    }
                )
        fold_scores = [fold["validation_mse"] for fold in folds]
        avg_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        calibrated_parameters = {
            "loss_mean": float(np.mean(losses)),
            "loss_standard_deviation": float(np.std(losses, ddof=1)),
            "sample_count": int(losses.size),
        }
        return {
            "method": "cross_validation",
            "calibrated_parameters": calibrated_parameters,
            "validation_scores": {"average_mse": avg_score},
            "cross_validation_results": {
                "k": k,
                "fold_scores": fold_scores,
                "folds": folds,
            },
        }

    def run_monte_carlo_analysis(
        self, num_iterations: int | None = None, convergence_threshold: float = 0.01
    ) -> dict[str, Any]:
        """
        Run advanced Monte Carlo analysis with convergence monitoring.

        Args:
            num_iterations: Number of Monte Carlo iterations
            convergence_threshold: Convergence threshold for stopping criteria

        Returns:
            Monte Carlo analysis results with convergence information
        """
        self._ensure_open()
        if num_iterations is None:
            num_iterations = self.config.get("risk_model", {}).get(
                "monte_carlo_iterations", 1000
            )
        if not isinstance(num_iterations, int) or num_iterations < 1:
            raise ValueError("num_iterations must be a positive integer")
        if convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")

        self.logger.info(
            f"Running Monte Carlo analysis with {num_iterations} iterations"
        )

        # Initialize results tracking
        all_losses = []
        running_means = []
        running_stds = []

        # Run Monte Carlo simulation in batches
        batch_size = max(1, min(100, num_iterations // 10))

        for i in range(0, num_iterations, batch_size):
            batch_iterations = min(batch_size, num_iterations - i)

            # Run batch simulation
            batch_results = self._run_monte_carlo_batch(batch_iterations)
            all_losses.extend(batch_results["losses"])

            # Update running statistics
            running_mean = np.mean(all_losses)
            running_std = np.std(all_losses)

            running_means.append(running_mean)
            running_stds.append(running_std)

            # Check convergence
            if len(running_means) > 10:
                recent_means = running_means[-10:]
                denominator = max(abs(recent_means[0]), np.finfo(float).eps)
                mean_change = abs(recent_means[-1] - recent_means[0]) / denominator

                if mean_change < convergence_threshold:
                    self.logger.info(
                        f"Convergence reached at iteration {i + batch_iterations}"
                    )
                    break

        # Calculate final statistics
        final_mean = np.mean(all_losses)
        final_std = np.std(all_losses)
        confidence_interval = stats.norm.interval(
            0.95, loc=final_mean, scale=final_std / np.sqrt(len(all_losses))
        )

        return {
            "total_iterations": len(all_losses),
            "final_aal": final_mean,
            "standard_deviation": final_std,
            "confidence_interval_95": confidence_interval,
            "convergence_info": {
                "converged": len(running_means) > 10
                and abs(running_means[-1] - running_means[-10])
                / max(abs(running_means[-10]), np.finfo(float).eps)
                < convergence_threshold,
                "threshold": convergence_threshold,
                "running_means": running_means,
                "running_stds": running_stds,
            },
            "loss_distribution": {
                "mean": final_mean,
                "median": np.median(all_losses),
                "percentiles": {
                    "5th": np.percentile(all_losses, 5),
                    "95th": np.percentile(all_losses, 95),
                    "99th": np.percentile(all_losses, 99),
                },
            },
        }

    def _run_monte_carlo_batch(self, batch_size: int) -> dict[str, Any]:
        """Run a batch of Monte Carlo simulations."""
        losses = []

        for _ in range(batch_size):
            # Generate random event
            event = self._generate_random_event()

            # Calculate loss for this event
            loss = self._calculate_event_loss(event)
            losses.append(loss)

        return {
            "batch_size": batch_size,
            "losses": losses,
            "batch_mean": np.mean(losses),
            "batch_std": np.std(losses),
        }

    def _generate_random_event(self) -> dict[str, Any]:
        """Sample an event from a configured historical hazard catalogue."""
        candidates = [
            (hazard_type, model)
            for hazard_type, model in self.hazard_models.items()
            if getattr(model, "historical_data", None) is not None
            and not model.historical_data.empty
        ]
        if not candidates:
            raise ValueError(
                "Monte Carlo analysis requires at least one fitted hazard model"
            )

        hazard_type, model = candidates[self.rng.integers(len(candidates))]
        row = model.historical_data.iloc[self.rng.integers(len(model.historical_data))]
        event = row.to_dict()
        event["hazard_type"] = hazard_type
        if "magnitude" not in event:
            intensity_column = next(
                (
                    column
                    for column in ("intensity", "water_depth", "wind_speed", "pga")
                    if column in event
                ),
                None,
            )
            if intensity_column is None:
                raise ValueError(
                    f"Hazard catalogue for {hazard_type} has no usable intensity column"
                )
            event["magnitude"] = event[intensity_column]
        return cast(dict[str, Any], event)

    def _calculate_event_loss(self, event: dict[str, Any]) -> float:
        """Calculate loss using configured exposure and vulnerability models."""
        exposure_records = [
            model.exposure_data
            for model in self.exposure_models.values()
            if getattr(model, "exposure_data", None) is not None
            and not model.exposure_data.empty
        ]
        if not exposure_records:
            raise ValueError("Monte Carlo analysis requires configured exposure data")

        value_columns = ("value", "replacement_cost", "property_value")
        exposure_values = [
            frame[column].to_numpy(dtype=float)
            for frame in exposure_records
            for column in value_columns
            if column in frame.columns
        ]
        if not exposure_values:
            raise ValueError(
                "Configured exposure data must contain a numeric value column"
            )
        base_exposure = float(np.concatenate(exposure_values).sum())

        vulnerability_models = list(self.vulnerability_models.values())
        if not vulnerability_models:
            raise ValueError(
                "Monte Carlo analysis requires a configured vulnerability model"
            )
        vulnerability_model = vulnerability_models[0]
        damage = vulnerability_model.calculate_enhanced_damage(
            event["hazard_type"],
            float(event["magnitude"]),
            {"asset_type": "building"},
            include_uncertainty=False,
        )
        return float(
            base_exposure * damage["damage_ratio"] * self.rng.lognormal(0, 0.3)
        )

    def save_enhanced_results(
        self, results: dict[str, Any], filename: str | None = None
    ) -> str:
        """
        Save enhanced analysis results with comprehensive metadata.

        Args:
            results: Analysis results to save
            filename: Output filename (optional)

        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_risk_analysis_{timestamp}.json"

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.output_dir, filename)

        # Add metadata to results
        enhanced_results = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "engine_version": "2.0.0",
                "integration_status": self.get_integration_status(),
                "configuration_summary": {
                    "monte_carlo_iterations": self.config.get("risk_model", {}).get(
                        "monte_carlo_iterations", 1000
                    ),
                    "confidence_level": self.config.get("risk_model", {}).get(
                        "confidence_level", 0.95
                    ),
                    "spatial_resolution": self.config.get("risk_model", {}).get(
                        "spatial_resolution", 1.0
                    ),
                },
            },
            "results": results,
        }

        # Save with custom encoder for datetime and numpy objects
        with open(filepath, "w") as f:
            json.dump(enhanced_results, f, indent=2, default=self._json_encoder)

        self.logger.info(f"Enhanced results saved to {filepath}")
        return filepath

    def _json_encoder(self, obj: Any) -> Any:
        """Convert datetime and numpy values to JSON-serializable equivalents.

        Args:
            obj: Value ``json`` could not serialize on its own.

        Returns:
            A JSON-serializable equivalent.

        Raises:
            TypeError: If the value has no known JSON representation.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _load_configured_models(self) -> None:
        """Load and initialize enabled models from the validated configuration."""
        self.logger.info("Loading risk models...")

        # Load hazard models
        hazard_config = self.config.get("hazards", {})
        for hazard_type, hazard_params in hazard_config.items():
            if hazard_params.get("enabled", False):
                self.logger.info(f"Loading {hazard_type} hazard model")
                self.hazard_models[hazard_type] = EnhancedHazardModel(
                    hazard_type=hazard_type, params=hazard_params
                )

        # Load vulnerability models
        vuln_config = self.config.get("vulnerability", {})
        for vuln_type, vuln_params in vuln_config.items():
            if vuln_params.get("enabled", False):
                self.logger.info(f"Loading {vuln_type} vulnerability model")
                self.vulnerability_models[vuln_type] = EnhancedVulnerabilityModel(
                    vulnerability_type=vuln_type, params=vuln_params
                )

        # Load exposure models
        exposure_config = self.config.get("exposure", {})
        for exp_type, exp_params in exposure_config.items():
            if exp_params.get("enabled", False):
                self.logger.info(f"Loading {exp_type} exposure model")
                self.exposure_models[exp_type] = EnhancedExposureModel(
                    exposure_type=exp_type, params=exp_params
                )

        self.logger.info(
            f"Models loaded: {len(self.hazard_models)} hazard, "
            f"{len(self.vulnerability_models)} vulnerability, "
            f"{len(self.exposure_models)} exposure"
        )

    def _run_portfolio_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run portfolio risk analysis.

        The inter-peril correlation is an ILLUSTRATIVE default (0.3 when more
        than one hazard model is configured), not an empirically fitted
        statistic. Override it per analysis via the ``inter_peril_correlation``
        keyword or persistently via ``config["portfolio"]["inter_peril_correlation"]``.
        """
        # Aggregate risk across loaded models
        hazard_count = len(self.hazard_models)
        vuln_count = len(self.vulnerability_models)
        default_correlation = float(
            self.config.get("portfolio", {}).get("inter_peril_correlation", 0.3)
        )
        inter_peril_correlation = float(
            kwargs.get("inter_peril_correlation", default_correlation)
        )
        return {
            "portfolio_id": kwargs.get("portfolio_id", "default"),
            "analysis_type": "portfolio",
            "aggregation_level": kwargs.get("aggregation_level", "location"),
            "risk_metrics": {
                "hazard_model_count": hazard_count,
                "vulnerability_model_count": vuln_count,
            },
            "correlation_analysis": {
                "inter_peril_correlation": (
                    inter_peril_correlation if hazard_count > 1 else 0.0
                ),
                "inter_peril_correlation_is_illustrative_default": True,
            },
            "diversification_benefits": {
                "benefit_ratio": max(0, 1 - 1 / max(hazard_count, 1))
            },
        }

    def _run_climate_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run climate risk analysis.

        The scenario factors, linear year scaling, and adaptation cost-benefit
        ratio are ILLUSTRATIVE defaults for first-order scenario comparison —
        they are not downscaled climate projections. Override per analysis via
        the ``scenario_factors`` / ``cost_benefit_ratio`` keywords, or
        persistently via ``config["climate"]["scenario_factors"]`` /
        ``config["climate"]["cost_benefit_ratio"]``.
        """
        # Climate projection using simple scaling factors per scenario
        baseline_year = kwargs.get("baseline_year", 2023)
        target_years = kwargs.get("target_years", [2050, 2100])
        scenarios = kwargs.get("scenarios", ["rcp4.5", "rcp8.5"])
        climate_config = self.config.get("climate", {})
        scenario_factors = dict(
            climate_config.get(
                "scenario_factors",
                {"rcp2.6": 1.1, "rcp4.5": 1.3, "rcp6.0": 1.5, "rcp8.5": 2.0},
            )
        )
        scenario_factors.update(kwargs.get("scenario_factors", {}))
        cost_benefit_ratio = float(
            kwargs.get(
                "cost_benefit_ratio", climate_config.get("cost_benefit_ratio", 2.5)
            )
        )
        projected = {}
        for sc in scenarios:
            factor = scenario_factors.get(sc, 1.2)
            projected[sc] = {
                str(yr): {
                    "risk_multiplier": 1 + (factor - 1) * (yr - baseline_year) / 100
                }
                for yr in target_years
            }
        return {
            "baseline_year": baseline_year,
            "target_years": target_years,
            "scenarios": scenarios,
            "projected_risks": projected,
            "adaptation_analysis": {"cost_benefit_ratio": cost_benefit_ratio},
        }

    def _run_stress_test(self, **kwargs: Any) -> dict[str, Any]:
        """Run stress testing analysis."""
        # Stress test by scaling losses by severity multiplier
        severity_map = {"low": 1.5, "moderate": 2.0, "high": 3.0, "extreme": 5.0}
        severity = kwargs.get("severity_level", "moderate")
        multiplier = severity_map.get(severity, 2.0)
        baseline_loss = kwargs.get("baseline_loss")
        if baseline_loss is None or baseline_loss < 0:
            raise ValueError("stress_test requires a non-negative baseline_loss")
        stressed_loss = baseline_loss * multiplier
        return {
            "scenario_type": kwargs.get("scenario_type", "historical"),
            "severity_level": severity,
            "baseline_metrics": {"expected_loss": float(baseline_loss)},
            "stressed_metrics": {
                "expected_loss": float(stressed_loss),
                "multiplier": multiplier,
            },
            "impact_analysis": {"loss_increase_pct": (multiplier - 1) * 100},
        }
