"""
GEO-INFER-DATA: Geospatial Data Management, ETL, and Storage Optimization

Provides multi-source data ingestion (``MultiSourceDataIngestion``), ETL
pipelines (``IntelligentETLPipeline``), adaptive storage (``AdaptiveDataStorage``),
and data quality management (``DataQualityManager``) for the GEO-INFER framework.
Use ``initialize_data_system()`` to set up all components in one call.
"""

from typing import Any
import logging

__version__ = "0.4.0"

from .core.ingestion import MultiSourceDataIngestion
from .core.pipeline import IntelligentETLPipeline
from .core.storage import AdaptiveDataStorage
from .core.validation import DataQualityManager
from .models.schemas import Dataset, DatasetMetadata, DataQualityReport

# Import submodules for convenience
from . import api as api
from . import connectors as connectors
from . import utils as utils


logger = logging.getLogger(__name__)
__all__ = [
    # Core classes
    "MultiSourceDataIngestion",
    "IntelligentETLPipeline",
    "AdaptiveDataStorage",
    "DataQualityManager",
    # Data models
    "Dataset",
    "DatasetMetadata",
    "DataQualityReport",
    # Utility functions
    "initialize_data_system",
    "validate_data_integrity",
    "optimize_storage_performance",
]


def initialize_data_system(
    storage_backends: list[str] | None = None,
    enable_validation: bool = True,
) -> dict[str, Any]:
    """
    Initialize all GEO-INFER-DATA components and return them in a dict.

    Constructs ``MultiSourceDataIngestion``, ``AdaptiveDataStorage``,
    ``IntelligentETLPipeline``, and ``DataQualityManager`` with sensible defaults,
    then returns them under the keys ``'ingestion'``, ``'storage'``, ``'pipeline'``,
    ``'quality_manager'``, and ``'status'``.

    Args:
        storage_backends: Storage backends to pass to ``AdaptiveDataStorage``
            (default: ``['local']``).
        enable_validation: Enable data validation in the ingestion component.

    Returns:
        Dictionary of the four components plus ``'status': 'initialized'``.
    """
    logger.info("Initializing GEO-INFER-DATA system")

    if storage_backends is None:
        storage_backends = ["local"]

    # Initialize core components
    ingestion = MultiSourceDataIngestion(
        data_sources=["satellite", "sensors", "crowdsourced"],
        format_detection="automatic",
        validation_enabled=enable_validation,
    )

    storage = AdaptiveDataStorage(
        storage_backends=storage_backends,
        optimization_strategy="access_pattern_based",
        compression_enabled=True,
    )

    pipeline = IntelligentETLPipeline(
        workflow_config=None,  # Will be loaded from config
        dependency_resolution="automatic",
        error_recovery="intelligent_retry",
    )

    quality_manager = DataQualityManager(
        validation_rules="comprehensive",
        quality_threshold=0.8,
        real_time_monitoring=True,
    )

    system_components = {
        "ingestion": ingestion,
        "storage": storage,
        "pipeline": pipeline,
        "quality_manager": quality_manager,
        "status": "initialized",
    }

    logger.info(
        f"GEO-INFER-DATA system initialized with {len(system_components)} components"
    )
    return system_components


async def validate_data_integrity(
    datasets: list[str], quality_threshold: float = 0.8
) -> dict[str, Any]:
    """
    Validate data integrity across a list of dataset identifiers.

    Constructs a ``DataQualityManager`` and calls ``validate_dataset`` on each
    entry in *datasets*. Failures are caught and recorded per-dataset rather than
    aborting the whole run.

    Args:
        datasets: Dataset identifiers to validate.
        quality_threshold: Minimum acceptable quality score (0.0–1.0).

    Returns:
        Dict with keys ``'validation_results'``, ``'overall_score'``,
        ``'datasets_validated'``, ``'quality_threshold'``, and ``'validation_passed'``.
    """
    logger.info(f"Validating integrity for {len(datasets)} datasets")

    quality_manager = DataQualityManager(
        validation_rules="comprehensive", quality_threshold=quality_threshold
    )

    validation_results: dict[str, Any] = {}
    overall_scores = []

    for dataset_id in datasets:
        try:
            report = await quality_manager.validate_dataset(dataset_id)
            validation_results[dataset_id] = report
            overall_scores.append(report.overall_score)
        except Exception as e:
            logger.error(f"Failed to validate dataset {dataset_id}: {e}")
            validation_results[dataset_id] = {"error": str(e)}

    overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    return {
        "validation_results": validation_results,
        "overall_score": overall_score,
        "datasets_validated": len(datasets),
        "quality_threshold": quality_threshold,
        "validation_passed": overall_score >= quality_threshold,
    }


def optimize_storage_performance(
    access_patterns: dict[str, Any], time_window: str = "30d"
) -> dict[str, Any]:
    """
    Optimize storage performance based on per-dataset access patterns.

    Delegates to ``AdaptiveDataStorage.optimize_for_patterns``. The *access_patterns*
    dict keys are dataset identifiers; values describe query frequency, spatial/temporal
    query shapes, and growth rate.

    Args:
        access_patterns: Dict mapping dataset IDs to their access-pattern metadata.
        time_window: Look-back window for pattern analysis (e.g. ``'30d'``, ``'7d'``).

    Returns:
        Whatever ``AdaptiveDataStorage.optimize_for_patterns`` returns — typically
        a dict with ``'optimizations'``, ``'actions'``, and ``'performance_impact'`` keys.
    """
    logger.info(f"Optimizing storage performance for {len(access_patterns)} patterns")

    storage = AdaptiveDataStorage(
        storage_backends=["postgresql", "minio", "redis"],
        optimization_strategy="access_pattern_based",
    )

    optimizations = storage.optimize_for_patterns(access_patterns, time_window)

    logger.info(
        f"Applied {len(optimizations.get('actions', []))} storage optimizations"
    )

    return optimizations
