## GEO-INFER-DATA — ETL, Storage, and Data Quality Backbone

**Purpose.** GEO-INFER-DATA is the foundational data backbone: ETL pipelines, storage optimization, and data quality assurance for geospatial datasets. Its README defines the module as the entry point of the framework's data lifecycle — everything the inference, agent, and domain bands consume passes through ingestion, transformation, storage, and validation here. A `config/` directory and `connectors` subpackage support the multi-source character of the module.

**Public API.** The `geo_infer_data` package exports core classes: `MultiSourceDataIngestion` (connector-driven ingestion), `IntelligentETLPipeline` (transformation orchestration), `AdaptiveDataStorage` (storage optimization), and `DataQualityManager` (quality assurance). Data contracts are exported from `models/schemas` as `Dataset`, `DatasetMetadata`, and `DataQualityReport`. Three utility functions — `initialize_data_system`, `validate_data_integrity`, and `optimize_storage_performance` — expose the module's lifecycle operations as callable entry points; `api/` subpackages surface the same capabilities as services.

The tests exercise ingestion, pipelines, storage, quality, and operation-specific dependency boundaries. Native DuckDB and columnar storage acceptance requires the corresponding installed-operation receipt.

**Theme role.** Infrastructure with geospatial-core reach: DATA sits at the base of the framework — upstream of every module — and anchors the geospatial core band by defining the dataset contracts everything else assumes.
