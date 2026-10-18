## GEO-INFER-AG — Precision Agriculture with Geospatial Intelligence

**Purpose.** GEO-INFER-AG applies geospatial intelligence and active inference principles to agriculture: field analysis, precision farming, and sustainability assessment. Its README describes advanced agricultural analysis driven by the framework's spatial data backbone and inference modules. The module README records package metadata (package `geo_infer_ag`) and routes testing through the unified runner (`uv run python GEO-INFER-TEST/run_unified_tests.py --module AG`).

**Public API.** The `geo_infer_ag` package organizes its exports into three documented groups. Core components: `AgriculturalAnalysis` (the main analysis entry point), `FieldBoundaryManager` (field geometry management), `SeasonalAnalysis`, and `SustainabilityAssessment`. Models: `CropYieldModel`, `SoilHealthModel`, `WaterUsageModel`, and `CarbonSequestrationModel`. `AgriculturalAPI` is an outbound HTTP client using a configured service URL; `FieldsResource`, `CropsResource`, and `YieldResource` are in-process resource classes. The package does not serve those classes as HTTP endpoints or establish availability of a remote service. The subpackage layout (`core/`, `models/`, `api/`) mirrors this triad.

The tests exercise agricultural analysis and its public data models. The shared repository runner selects this module with `--module AG`.

**Theme role.** Domain sciences: AG specializes the framework's geospatial and inference components for agricultural analysis and resource handling.
