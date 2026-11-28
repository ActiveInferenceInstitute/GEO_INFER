## GEO-INFER-DATA — ETL, Storage, and Data Quality Backbone

**Purpose.** GEO-INFER-DATA is the foundational data backbone: ETL pipelines, storage optimization, and data quality assurance for geospatial datasets. It supplies ingestion, transformation, storage, and validation contracts for workflows that select these components. Other modules may accept native arrays or external data directly; DATA is not a universal dependency. A `config/` directory and `connectors` subpackage support the multi-source character of the module.

**Public API.** The `geo_infer_data` package exports core classes: `MultiSourceDataIngestion` (connector-driven ingestion), `IntelligentETLPipeline` (transformation orchestration), `AdaptiveDataStorage` (storage optimization), and `DataQualityManager` (quality assurance). Data contracts are exported from `models/schemas` as `Dataset`, `DatasetMetadata`, and `DataQualityReport`. Three utility functions — `initialize_data_system`, `validate_data_integrity`, and `optimize_storage_performance` — expose the module's lifecycle operations as callable entry points; `api/` subpackages surface the same capabilities as services.

The tests exercise ingestion, pipelines, storage, quality, and operation-specific dependency boundaries. Native DuckDB and columnar storage acceptance requires the corresponding installed-operation receipt.

**Theme role.** Infrastructure with geospatial-core reach: DATA provides reusable dataset and storage contracts for workflows that use its components. Theme membership does not establish a dependency edge from every other module.
