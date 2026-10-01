"""
Economic Modeling Engine - Core orchestration and execution framework.
"""

from typing import Any, cast
from dataclasses import dataclass
import logging


@dataclass
class ModelConfiguration:
    """Configuration settings for economic models."""

    model_type: str
    parameters: dict[str, Any]
    spatial_config: dict[str, Any] | None = None
    temporal_config: dict[str, Any] | None = None


class EconomicModelingEngine:
    """
    Core engine for orchestrating and executing economic models.

    This class provides the central framework for managing economic model
    lifecycles, from initialization through execution to results processing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the Economic Modeling Engine.

        Args:
            config: Optional configuration dictionary for the engine
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.active_models: dict[str, Any] = {}
        self.model_registry: dict[str, Any] = {}

    def register_model(self, model_name: str, model_class: type) -> None:
        """
        Register a model class for use by the engine.

        Args:
            model_name: Unique identifier for the model
            model_class: The model class to register
        """
        self.model_registry[model_name] = model_class
        self.logger.info(f"Registered model: {model_name}")

    def create_model(self, model_name: str, model_config: ModelConfiguration) -> Any:
        """
        Create and initialize a model instance.

        Args:
            model_name: Name of the registered model to create
            model_config: Configuration for the model instance

        Returns:
            Initialized model instance
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not registered")

        model_class = self.model_registry[model_name]
        model_instance = model_class(model_config)

        instance_id = f"{model_name}_{len(self.active_models)}"
        self.active_models[instance_id] = model_instance

        self.logger.info(f"Created model instance: {instance_id}")
        return model_instance

    def execute_model(
        self, model_instance: Any, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a model with provided data.

        Args:
            model_instance: The model instance to execute
            data: Input data for the model

        Returns:
            Model execution results
        """
        try:
            if hasattr(model_instance, "validate_inputs"):
                model_instance.validate_inputs(data)

            if hasattr(model_instance, "run"):
                results = model_instance.run(data)
            elif hasattr(model_instance, "execute"):
                results = model_instance.execute(data)
            else:
                raise AttributeError("Model must implement 'run' or 'execute' method")

            return cast(dict[str, Any], results)

        except Exception as e:
            self.logger.error(f"Model execution failed: {str(e)}")
            raise

    def batch_execute(
        self, models: list[tuple], common_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute multiple models with common data.

        Args:
            models: List of (model_instance, specific_config) tuples
            common_data: Data available to all models

        Returns:
            Dictionary of results keyed by model identifier
        """
        results = {}

        for i, (model_instance, specific_config) in enumerate(models):
            model_id = f"model_{i}"
            try:
                # Merge common data with model-specific config
                model_data = {**common_data, **(specific_config or {})}
                result = self.execute_model(model_instance, model_data)
                results[model_id] = result

            except Exception as e:
                self.logger.error(f"Failed to execute {model_id}: {str(e)}")
                results[model_id] = {"error": str(e)}

        return results

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about a registered model.

        Args:
            model_name: Name of the model to query

        Returns:
            Model information dictionary
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not registered")

        model_class = self.model_registry[model_name]

        info = {
            "name": model_name,
            "class": model_class.__name__,
            "docstring": model_class.__doc__,
            "methods": [
                method for method in dir(model_class) if not method.startswith("_")
            ],
        }

        return info

    def list_models(self) -> list[str]:
        """
        List all registered models.

        Returns:
            List of registered model names
        """
        return list(self.model_registry.keys())

    def cleanup(self) -> None:
        """Clean up resources and active model instances."""
        for model_id, model_instance in self.active_models.items():
            if hasattr(model_instance, "cleanup"):
                try:
                    model_instance.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up {model_id}: {str(e)}")

        self.active_models.clear()
        self.logger.info("Engine cleanup completed")
