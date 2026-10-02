"""
GEO-INFER-SPM: Statistical Parametric Mapping for Geospatial Analysis

This module implements Statistical Parametric Mapping (SPM) methodology adapted
for geospatial analysis, providing rigorous statistical inference for spatially
and temporally continuous data fields while preserving spatiotemporal relationships.

The implementation is grounded in Active Inference principles, using Bayesian
inference for uncertainty quantification and free energy minimization for
optimal model selection.

Core Components:
- General Linear Model (GLM) for geospatial data analysis
- Random Field Theory (RFT) for multiple comparison correction
- Spatial autocorrelation modeling and cluster-based inference
- Bayesian extensions with hierarchical models
- Comprehensive visualization and statistical mapping tools

Example:
    >>> import geo_infer_spm as gispm
    >>> # Load geospatial data
    >>> data = gispm.load_data("temperature_data.tif")
    >>> # Create design matrix (factors maps factor name to its levels)
    >>> design = gispm.create_design_matrix(data, factors={"season": ["winter", "spring"]})
    >>> # Fit GLM and compute SPM
    >>> model = gispm.fit_glm(data, design)
    >>> contrast = gispm.contrast(model, "season_winter > 0")
    >>> spm_map = gispm.compute_spm(model, contrast, correction="RFT")
"""

# Core SPM functionality
from .core.glm import GeneralLinearModel, fit_glm
from .core.rft import RandomFieldTheory, compute_spm
from .core.contrasts import Contrast, contrast

from .core.spatial_analysis import SpatialAnalyzer
from .core.temporal_analysis import TemporalAnalyzer
from .core.bayesian import BayesianSPM

# Data models
from .models.data_models import SPMData, SPMResult, ContrastResult

# Utilities
from .utils.data_io import load_data, save_spm
from .utils.helpers import create_design_matrix, generate_synthetic_data

# Visualization (re-exported from the visualization subpackage)
from .visualization import create_statistical_map  # noqa: F401

# Advanced modeling methods
from .core.advanced import (
    MixedEffectsSPM,
    NonparametricSPM,
    ModelValidator,
    SpatialRegression,
)
from .core.advanced.mixed_effects import fit_mixed_effects
from .core.advanced.nonparametric import fit_nonparametric
from .core.advanced.model_validation import validate_spm_model
from .core.advanced.spatial_regression import fit_spatial_model

# API
from .api.endpoints import SPMAPI

__version__ = "0.4.0"
__author__ = "GEO-INFER Framework"
__description__ = "Statistical Parametric Mapping for Geospatial Analysis"

# Make version accessible as module attribute
VERSION = __version__

__all__ = [
    # Core SPM functionality
    "GeneralLinearModel",
    "fit_glm",
    "RandomFieldTheory",
    "compute_spm",
    "Contrast",
    "contrast",
    # Analysis tools
    "SpatialAnalyzer",
    "TemporalAnalyzer",
    "BayesianSPM",
    # Advanced modeling methods
    "MixedEffectsSPM",
    "NonparametricSPM",
    "ModelValidator",
    "SpatialRegression",
    "fit_mixed_effects",
    "fit_nonparametric",
    "validate_spm_model",
    "fit_spatial_model",
    # Data models
    "SPMData",
    "SPMResult",
    "ContrastResult",
    # Utilities
    "load_data",
    "save_spm",
    "create_design_matrix",
    "generate_synthetic_data",
    "create_statistical_map",
    # API
    "SPMAPI",
]
