"""
GEO-INFER-COG: Cognitive Geospatial Processing

This module provides human-centered geospatial tools that model perception,
reasoning, and spatial cognition for intuitive interfaces.

The module implements cognitive models of spatial thinking to enhance
geospatial decision-making and interface design across the GEO-INFER framework.

Main Components:
- Cognitive Processing Engine: Core spatial cognition modeling
- Spatial Perception Models: Human-like spatial understanding
- Spatial Reasoning Systems: Computational spatial reasoning
- Spatial Language Processing: Geographic language understanding
- Visualization Adapters: Cognitively optimized visualizations
- Decision Support Systems: Human-centered spatial decision making

Integration Points:
- GEO-INFER-SPACE: Enhanced spatial operations with cognitive models
- GEO-INFER-APP: Improved interface design with cognitive principles
- GEO-INFER-AGENT: Cognitive models for agent decision-making
- GEO-INFER-AI: Human-like spatial intelligence frameworks
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER-COG Team"

# Core cognitive processing components
from .core.cognitive_engine import CognitiveProcessingEngine
from .core.spatial_perception import SpatialPerceptionModel
from .core.spatial_reasoning import SpatialReasoningEngine
from .core.spatial_memory import SpatialMemoryModel

# Supporting components
from .spatial_language import SpatialLanguageProcessor
from .visualization import HumanCenteredVisualizer
from .decision import SpatialDecisionSupport

# API components
from .api.rest_api import create_cog_api_app

# Utility functions and helpers
from .utils.validation import validate_spatial_data, validate_cognitive_model
from .utils.helpers import load_cognitive_profile, save_cognitive_model

# Configuration and models
from .models.cognitive_models import CognitiveMap, SpatialKnowledgeGraph
from .models.user_profiles import UserCognitiveProfile, ProfileManager

__all__ = [
    # Core components
    "CognitiveProcessingEngine",
    "SpatialPerceptionModel",
    "SpatialReasoningEngine",
    "SpatialMemoryModel",
    # Supporting components
    "SpatialLanguageProcessor",
    "HumanCenteredVisualizer",
    "SpatialDecisionSupport",
    "create_cog_api_app",
    # Utilities
    "validate_spatial_data",
    "validate_cognitive_model",
    "load_cognitive_profile",
    "save_cognitive_model",
    # Models
    "CognitiveMap",
    "SpatialKnowledgeGraph",
    "UserCognitiveProfile",
    "ProfileManager",
    # Version info
    "__version__",
    "__author__",
]
