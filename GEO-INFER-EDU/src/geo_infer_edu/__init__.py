"""
GEO-INFER-EDU: Educational Technology Module

This module provides educational technology capabilities for geospatial systems,
including curriculum design, interactive exercises, progress tracking, and
personalized learning paths.

Key Features:
- Curriculum design with standards alignment
- Interactive exercise generation
- Learning progress tracking and analytics
- Personalized learning recommendations
- Professional development support
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

from .core.curriculum import CurriculumDesigner
from .core.exercises import ExerciseGenerator
from .core.progress import ProgressTracker
from .core.personalization import PersonalizedLearning
from .core.professional import ProfessionalDevelopment

__all__ = [
    "CurriculumDesigner",
    "ExerciseGenerator",
    "ProgressTracker",
    "PersonalizedLearning",
    "ProfessionalDevelopment",
]
