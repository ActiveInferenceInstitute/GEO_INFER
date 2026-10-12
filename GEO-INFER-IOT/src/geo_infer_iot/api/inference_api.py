"""
Bayesian Inference API Module

This module provides API endpoints for Bayesian spatial inference operations,
including spatial prediction, uncertainty quantification, and model management.
"""

import logging
from typing import Any
from datetime import datetime, UTC
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks

# BayesianSpatialInference lives in geo_infer_iot.core.inference; a broken
# installation must fail at import time instead of silently 503-ing every
# inference endpoint.
from geo_infer_iot.core.inference import BayesianSpatialInference

logger = logging.getLogger(__name__)


class BayesianInferenceAPI:
    """
    API for Bayesian spatial inference operations.

    Provides endpoints for:
    - Running spatial inference on sensor data
    - Retrieving posterior distributions
    - Managing inference models and configurations
    - Real-time inference updates
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.app = FastAPI(
            title="GEO-INFER-IOT Bayesian Inference API", version="1.0.0"
        )

        # Initialize the inference engine. The import above fails loudly when
        # the package is broken, so the engine is always available here.
        self.inference_engine: Any | None = BayesianSpatialInference(
            variable="default",
            spatial_resolution=8,
            temporal_window="1h",
            config=config,
        )

        # Inference cache and history
        self.inference_history: list[dict[str, Any]] = []
        self.model_cache: dict[str, Any] = {}

        # Setup API routes
        self._setup_routes()

        logger.info("BayesianInferenceAPI initialized")

    def _setup_routes(self) -> None:
        """Setup API routes and endpoints."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """API root endpoint."""
            return {
                "service": "GEO-INFER-IOT Bayesian Inference API",
                "version": "1.0.0",
                "status": "operational",
                "inference_available": self.inference_engine is not None,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @self.app.post("/inference/spatial")
        async def run_spatial_inference(
            sensor_data: list[dict],
            variable: str = Query(..., description="Variable to infer"),
            spatial_resolution: int = Query(
                8, description="H3 resolution for inference"
            ),
            confidence_levels: list[float] = Query(
                [0.68, 0.95], description="Confidence levels"
            ),
        ) -> dict[str, Any]:
            """Run Bayesian spatial inference on sensor data."""
            if self.inference_engine is None:
                raise HTTPException(
                    status_code=503, detail="Bayesian inference not available"
                )

            try:
                # Configure inference engine for this variable
                self.inference_engine.variable = variable
                self.inference_engine.spatial_resolution = spatial_resolution

                # Run inference
                result = self.inference_engine.infer_spatial_distribution(
                    sensor_data=sensor_data, update_interval="15min"
                )

                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])

                # Get posterior map with requested confidence levels
                posterior_map = self.inference_engine.get_posterior_map(
                    confidence_levels
                )

                # Store in history
                inference_record = {
                    "variable": variable,
                    "spatial_resolution": spatial_resolution,
                    "sensor_count": result.get("sensor_count", 0),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "result": result,
                    "posterior_map": posterior_map,
                }
                self.inference_history.append(inference_record)

                return {
                    "inference_result": result,
                    "posterior_map": posterior_map,
                    "inference_id": len(self.inference_history) - 1,
                }

            except Exception as e:
                logger.error(f"Error in spatial inference: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Inference failed: {str(e)}"
                ) from e

        @self.app.get("/inference/{inference_id}")
        async def get_inference_result(inference_id: int) -> dict[str, Any]:
            """Get results of a specific inference operation."""
            if inference_id >= len(self.inference_history) or inference_id < 0:
                raise HTTPException(
                    status_code=404, detail=f"Inference {inference_id} not found"
                )

            return {
                "inference_id": inference_id,
                "result": self.inference_history[inference_id],
            }

        @self.app.get("/inference/history")
        async def get_inference_history(
            limit: int = Query(50, description="Maximum history entries to return"),
            variable: str | None = Query(None, description="Filter by variable"),
        ) -> dict[str, Any]:
            """Get history of inference operations."""
            history = self.inference_history

            # Filter by variable if specified
            if variable:
                history = [h for h in history if h.get("variable") == variable]

            # Apply limit and return most recent
            recent_history = history[-limit:] if len(history) > limit else history

            return {
                "history": recent_history,
                "total_entries": len(self.inference_history),
                "returned_entries": len(recent_history),
                "filters": {"variable": variable},
            }

        @self.app.get("/models")
        async def list_models() -> dict[str, Any]:
            """List available inference models."""
            models = {
                "bayesian_spatial": {
                    "type": "Gaussian Process",
                    "variables": [
                        "temperature",
                        "humidity",
                        "air_quality",
                        "soil_moisture",
                    ],
                    "spatial_resolution": [5, 6, 7, 8, 9, 10],
                    "available": self.inference_engine is not None,
                    "description": "Bayesian spatial inference using Gaussian processes",
                }
            }

            return {"models": models, "total_models": len(models)}

        @self.app.post("/models/{model_type}/configure")
        async def configure_model(
            model_type: str, config: dict, background_tasks: BackgroundTasks
        ) -> dict[str, Any]:
            """Configure an inference model."""
            if model_type != "bayesian_spatial":
                raise HTTPException(
                    status_code=404, detail=f"Model type {model_type} not found"
                )

            if self.inference_engine is None:
                raise HTTPException(
                    status_code=503, detail="Bayesian inference not available"
                )

            try:
                # Update inference engine configuration
                self.inference_engine.config.update(config)

                # Store in model cache
                self.model_cache[model_type] = {
                    "config": config,
                    "last_updated": datetime.now(UTC).isoformat(),
                }

                return {
                    "message": f"Model {model_type} configured successfully",
                    "config": config,
                    "status": "configured",
                }

            except Exception as e:
                logger.error(f"Error configuring model: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Model configuration failed: {str(e)}"
                ) from e

        @self.app.get("/predictions/spatial")
        async def get_spatial_predictions(
            variable: str = Query(..., description="Variable for predictions"),
            h3_resolution: int = Query(8, description="H3 resolution"),
            confidence_level: float = Query(0.95, description="Confidence level"),
        ) -> dict[str, Any]:
            """Get current spatial predictions for a variable."""
            if self.inference_engine is None:
                raise HTTPException(
                    status_code=503, detail="Bayesian inference not available"
                )

            try:
                # Ensure inference engine is configured for this variable
                self.inference_engine.variable = variable

                # Get posterior map
                posterior_map = self.inference_engine.get_posterior_map(
                    [confidence_level]
                )

                if "error" in posterior_map:
                    raise HTTPException(status_code=404, detail=posterior_map["error"])

                return {
                    "variable": variable,
                    "h3_resolution": h3_resolution,
                    "confidence_level": confidence_level,
                    "predictions": posterior_map,
                    "generated_at": datetime.now(UTC).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error getting spatial predictions: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Prediction retrieval failed: {str(e)}"
                ) from e

        @self.app.post("/inference/batch")
        async def run_batch_inference(
            inference_requests: list[dict], background_tasks: BackgroundTasks
        ) -> dict[str, Any]:
            """Run multiple inference operations in batch."""
            if self.inference_engine is None:
                raise HTTPException(
                    status_code=503, detail="Bayesian inference not available"
                )

            results: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []

            for request in inference_requests:
                try:
                    sensor_data = request.get("sensor_data", [])
                    variable = request.get("variable", "default")
                    spatial_resolution = request.get("spatial_resolution", 8)

                    # Configure inference engine
                    self.inference_engine.variable = variable
                    self.inference_engine.spatial_resolution = spatial_resolution

                    # Run inference
                    result = self.inference_engine.infer_spatial_distribution(
                        sensor_data
                    )

                    results.append(
                        {
                            "request_id": len(results),
                            "variable": variable,
                            "success": "error" not in result,
                            "result": result,
                        }
                    )

                except Exception as e:
                    error_result = {
                        "request_id": len(errors) + len(results),
                        "variable": request.get("variable", "unknown"),
                        "success": False,
                        "error": str(e),
                    }
                    errors.append(error_result)
                    results.append(error_result)

            return {
                "batch_results": results,
                "total_requests": len(inference_requests),
                "successful_requests": len(
                    [r for r in results if r.get("success", False)]
                ),
                "failed_requests": len(errors),
                "processed_at": datetime.now(UTC).isoformat(),
            }

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
