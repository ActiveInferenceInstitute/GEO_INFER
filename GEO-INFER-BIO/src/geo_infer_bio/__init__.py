"""
GEO-INFER-BIO: A bioinformatics module for the GEO-INFER framework.
"""

from .core.sequence_analysis import SequenceAnalyzer
from .utils.validation import DataValidator
from .utils.visualization import BioVisualizer

from .climate import ClimateDataProcessor, ClimateDataset
from .microbiome import MicrobiomeDataLoader, MicrobiomeDataset
from .soil import SoilDataIntegrator, SoilDataset

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__email__ = "team@geo-infer.org"

__all__ = [
    "SequenceAnalyzer",
    "DataValidator",
    "BioVisualizer",
    "ClimateDataProcessor",
    "ClimateDataset",
    "MicrobiomeDataLoader",
    "MicrobiomeDataset",
    "SoilDataIntegrator",
    "SoilDataset",
]
