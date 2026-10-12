"""
MLflow integration for MLOps workflows.

This module provides integration with MLflow for experiment tracking,
model versioning, and model deployment in geospatial AI workflows.

MLflow is an optional dependency: install the ``mlops`` extra
(``pip install geo-infer-ai[mlops]`` / ``uv sync --extra mlops``) to enable
tracking; without it the pipeline degrades gracefully to disabled mode.

"""

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Try to import MLflow, but make it optional
try:
    import mlflow  # type: ignore[import-not-found]
    import mlflow.sklearn  # type: ignore[import-not-found]

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("MLflow not available. Install with: uv pip install mlflow")


class MLflowPipeline:
    """
    MLflow pipeline for experiment tracking and model management.

    Provides integration with MLflow for tracking experiments, logging models,
    and managing model versions in geospatial AI workflows.
    """

    def __init__(
        self,
        experiment_name: str = "geospatial_ai",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the MLflow pipeline.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local file system)
            enabled: Whether MLflow tracking is enabled
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.current_run: Any | None = None

        if self.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow experiment and tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Tracking disabled.")
            self.enabled = False
            return

        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Get or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    _experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created MLflow experiment: {self.experiment_name}")
                else:
                    _experiment_id = experiment.experiment_id
                    logger.info(
                        f"Using existing MLflow experiment: {self.experiment_name}"
                    )
            except Exception as e:
                logger.warning(f"Could not set up MLflow experiment: {e}")
                self.enabled = False

        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Tracking disabled.")
            self.enabled = False

    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Optional tags for the run
        """
        if not self.enabled:
            return

        try:
            self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.enabled = False

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled or self.current_run is None:
            return

        try:
            mlflow.end_run()
            self.current_run = None
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.enabled:
            return

        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metrics
        """
        if not self.enabled:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: Trained model to log
            artifact_path: Path within the run to store the model
            registered_model_name: Optional name for model registration
        """
        if not self.enabled:
            return

        try:
            # Log sklearn models
            if hasattr(model, "predict") and hasattr(model, "fit"):
                mlflow.sklearn.log_model(model, artifact_path)
                logger.info(f"Logged model to MLflow: {artifact_path}")

                # Register model if name provided
                if registered_model_name and self.current_run is not None:
                    try:
                        mlflow.register_model(
                            f"runs:/{self.current_run.info.run_id}/{artifact_path}",
                            registered_model_name,
                        )
                        logger.info(f"Registered model: {registered_model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to register model: {e}")
            else:
                logger.warning("Model type not supported for MLflow logging")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_artifacts(
        self, local_dir: str | Path, artifact_path: str | None = None
    ) -> None:
        """
        Log artifacts (files) to MLflow.

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Optional path within the run to store artifacts
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifacts(str(local_dir), artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow.

        Args:
            model_uri: MLflow model URI (e.g., "runs:/run_id/model" or "models:/model_name/version")

        Returns:
            Loaded model
        """
        if not self.enabled:
            raise ValueError("MLflow not enabled")

        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise
