"""
GEO-INFER-MATH

A comprehensive mathematical library for geospatial data analysis and inference.
This module provides specialized mathematical tools, models, and algorithms that
are optimized for processing and analyzing geographical and spatial data.

Key components:
- Spatial statistics and probability distributions
- Geospatial optimization algorithms
- Spatial interpolation and extrapolation methods
- Vector and raster math operations
- Coordinate transformations and projections
- Geometric operations and calculations
- Tensor operations for multi-dimensional geospatial data
"""

from geo_infer_math.core.spatial_statistics import (
    SpatialDescriptiveStats,
    MoranI,
    GearysC,
    GetisOrd,
    getis_ord_g,
    ripley_k,
    semivariogram,
    spatial_descriptive_statistics,
    spatial_entropy,
    local_indicators_spatial_association,
)

from geo_infer_math.core.interpolation import (
    InterpolationConfig,
    SpatialInterpolator,
    IDWInterpolator,
    KrigingInterpolator,
    RBFInterpolator,
    LinearInterpolator,
    CubicInterpolator,
    InterpolationManager,
    create_interpolation_manager,
    interpolate_spatial_data,
    create_interpolation_grid,
)

from geo_infer_math.core.optimization import (
    OptimizationConfig,
    Optimizer,
    GradientDescentOptimizer,
    GeneticAlgorithmOptimizer,
    ScipyOptimizer,
    MultiObjectiveOptimizer,
    OptimizationManager,
    create_optimization_manager,
    optimize_function,
    compare_optimization_methods,
)

from geo_infer_math.core.geometry import (
    Point,
    LineString,
    Polygon,
    haversine_distance,
    vincenty_distance,
    bearing,
    destination_point,
    point_in_polygon,
    buffer_point,
    line_intersection,
    polygon_area_spherical,
)

from geo_infer_math.models import clustering as clustering
from geo_infer_math.models import regression as regression

from geo_infer_math.core.information_theory import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy,
    mutual_information,
    kl_divergence,
    EntropyCalculator,
    MutualInformationCalculator,
    KLDivergenceCalculator,
)

from geo_infer_math.core.theorem_proving import (
    TheoremProver,
    ProofResult,
    create_prover,
    TheoremDatabase,
)

# GPU acceleration (CPU fallback; cupy/torch are optional)
from geo_infer_math.core.gpu_acceleration import (
    GPUAccelerator,
    is_gpu_available,
    get_gpu_info,
)

from geo_infer_math.api.convenience import (
    ActiveInferenceConvenience,
    BayesianConvenience,
    AIConvenience,
    InformationTheoryConvenience,
    SpatialConvenience,
    IntegrationConvenience,
)

from geo_infer_math.integration.ai import (
    AIGradientHelpers,
    SpatialLossFunctions,
    OptimizationBridges,
)
from geo_infer_math.integration.act import (
    FreeEnergyCalculator,
    VariationalInferenceHelpers,
    BeliefUpdating,
)
from geo_infer_math.integration.bayes import (
    PosteriorHelpers,
    PriorBuilders,
    MCMCHelpers,
)

__all__ = [
    # Spatial statistics
    "SpatialDescriptiveStats",
    "MoranI",
    "GearysC",
    "GetisOrd",
    "getis_ord_g",
    "ripley_k",
    "semivariogram",
    "spatial_descriptive_statistics",
    "spatial_entropy",
    "local_indicators_spatial_association",
    # Interpolation
    "InterpolationConfig",
    "SpatialInterpolator",
    "IDWInterpolator",
    "KrigingInterpolator",
    "RBFInterpolator",
    "LinearInterpolator",
    "CubicInterpolator",
    "InterpolationManager",
    "create_interpolation_manager",
    "interpolate_spatial_data",
    "create_interpolation_grid",
    # Optimization
    "OptimizationConfig",
    "Optimizer",
    "GradientDescentOptimizer",
    "GeneticAlgorithmOptimizer",
    "ScipyOptimizer",
    "MultiObjectiveOptimizer",
    "OptimizationManager",
    "create_optimization_manager",
    "optimize_function",
    "compare_optimization_methods",
    # Geometry
    "Point",
    "LineString",
    "Polygon",
    "haversine_distance",
    "vincenty_distance",
    "bearing",
    "destination_point",
    "point_in_polygon",
    "buffer_point",
    "line_intersection",
    "polygon_area_spherical",
    # Information theory
    "shannon_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    "mutual_information",
    "kl_divergence",
    "EntropyCalculator",
    "MutualInformationCalculator",
    "KLDivergenceCalculator",
    # Theorem proving
    "TheoremProver",
    "ProofResult",
    "create_prover",
    "TheoremDatabase",
    # GPU acceleration
    "GPUAccelerator",
    "is_gpu_available",
    "get_gpu_info",
    # Convenience APIs
    "ActiveInferenceConvenience",
    "BayesianConvenience",
    "AIConvenience",
    "InformationTheoryConvenience",
    "SpatialConvenience",
    "IntegrationConvenience",
    # Module integration
    "AIGradientHelpers",
    "SpatialLossFunctions",
    "OptimizationBridges",
    "FreeEnergyCalculator",
    "VariationalInferenceHelpers",
    "BeliefUpdating",
    "PosteriorHelpers",
    "PriorBuilders",
    "MCMCHelpers",
]


# Version information
__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"
__email__ = "geo-infer@activeinference.institute"
