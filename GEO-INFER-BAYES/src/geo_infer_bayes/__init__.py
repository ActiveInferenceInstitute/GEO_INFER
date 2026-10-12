"""
GEO-INFER-BAYES: Bayesian inference for geospatial applications
======================================================================

This module provides a comprehensive framework for Bayesian inference
processes within the GEO-INFER ecosystem, implementing probabilistic modeling,
uncertainty quantification, and Bayesian computational methods for geospatial
applications.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__email__ = "geo-infer@activeinference.institute"


# Submodule aliases
from . import api, core, models, utils  # noqa: F401

# Expose key classes for easy import
from .models.spatial_gp import SparseSpatialGP, SpatialGP
from .models.gaussian_process import GaussianProcess, SpatialCovariance
from .core.inference import BayesianInference
from .core.posterior import PosteriorAnalysis
from .core.variational import VariationalInference
from .core.mcmc import MCMC as MCMCSampler
from .civic_intel import (
    HazardCategoricalPrior,
    build_hazard_categorical_prior,
    build_hazard_prior_table,
    load_crescent_city_intel,
)
from .geo_observations import (
    CRESCENT_CITY_OBSERVATIONS_SCHEMA,
    load_crescent_city_geo_observations,
)

__all__ = [
    "SpatialGP",
    "SparseSpatialGP",
    "BayesianInference",
    "PosteriorAnalysis",
    "GaussianProcess",
    "SpatialCovariance",
    "VariationalInference",
    "MCMCSampler",
    "HazardCategoricalPrior",
    "build_hazard_categorical_prior",
    "build_hazard_prior_table",
    "CRESCENT_CITY_OBSERVATIONS_SCHEMA",
    "load_crescent_city_geo_observations",
    "load_crescent_city_intel",
]
