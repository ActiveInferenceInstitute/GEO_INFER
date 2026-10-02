## GEO-INFER-CLIMATE — Climate Modeling and Impact Assessment

**Purpose.** GEO-INFER-CLIMATE provides climate modeling, weather analysis, and climate change impact assessment for geospatial systems. Its README establishes the module as the atmospheric-science vertical, covering the full chain from raw climate data through indices, downscaling, projections, and extreme-event analysis to impact assessment.

**Public API.** The `geo_infer_climate` package exports a nine-component pipeline from `core`: `ClimateDataProcessor` (ingest and normalize climate datasets), `ClimateIndicesCalculator` (standard climate indices), `DownscalingMethods` and `ClimateProjections` (scenario processing), `ExtremeEventAnalyzer`, `ClimateImpactAssessor`, and the analysis trio `ClimateClassifier`, `TemperatureTrendAnalyzer`, and `PrecipitationAnalyzer`. Each export is a distinct stage of the climate workflow, so the module reads as a documented processing chain rather than a loose toolkit.

**Verification status.** The `tests/` directory is present with 12 test files, one layer covering each major component family; testing routes through the unified runner (`--module CLIMATE`).

**Theme role.** Domain sciences: CLIMATE is the earth-system band. Its projections and extremes feed hazard-aware consumers across the framework — including the hazard-prior machinery in the inference band — and its datasets flow through the DATA backbone.
