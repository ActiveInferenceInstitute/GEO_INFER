"""
Cross-Module Integration Convenience Methods

This module provides convenience methods for integrating across
different GEO-INFER modules.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


def cross_module_helper(
    module_name: str, operation: str, data: dict[str, Any], **kwargs: Any
) -> Any:
    """
    Helper for cross-module operations.

    Args:
        module_name: Name of module to use
        operation: Operation name
        data: Input data
        **kwargs: Additional parameters

    Returns:
        Operation result
    """
    if module_name == "act":
        from geo_infer_math.api.convenience.act_convenience import (
            ActiveInferenceConvenience,
        )

        act_conv = ActiveInferenceConvenience()
        if operation == "free_energy":
            obs = data.get("observations")
            beliefs = data.get("beliefs")
            assert obs is not None and beliefs is not None
            return act_conv.calculate_free_energy(obs, beliefs, **kwargs)

    elif module_name == "bayes":
        from geo_infer_math.api.convenience.bayes_convenience import BayesianConvenience

        bayes_conv = BayesianConvenience()
        if operation == "posterior":
            prior = data.get("prior")
            likelihood = data.get("likelihood")
            data_val = data.get("data")
            assert prior is not None and likelihood is not None and data_val is not None
            return bayes_conv.calculate_posterior(prior, likelihood, data_val, **kwargs)

    elif module_name == "ai":
        from geo_infer_math.api.convenience.ai_convenience import AIConvenience

        ai_conv = AIConvenience()
        if operation == "gradient":
            fn = data.get("function")
            params = data.get("parameters")
            assert fn is not None and params is not None
            return ai_conv.compute_gradient(fn, params, **kwargs)

    else:
        raise ValueError(f"Unknown module: {module_name}")

    return None


class IntegrationConvenience:
    """
    Convenience class for cross-module integration.

    Provides methods for integrating operations across modules.
    """

    def __init__(self) -> None:
        """Initialize integration convenience class."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._module_registry: dict[str, Any] = {}
        self.logger.debug("IntegrationConvenience initialized")

    def execute_cross_module(
        self, module_name: str, operation: str, data: dict[str, Any], **kwargs: Any
    ) -> Any:
        """
        Execute cross-module operation.

        Args:
            module_name: Module name
            operation: Operation name
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Operation result
        """
        return cross_module_helper(module_name, operation, data, **kwargs)
