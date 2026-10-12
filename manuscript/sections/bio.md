## GEO-INFER-BIO — Bioinformatics with Geospatial Context

**Purpose.** GEO-INFER-BIO brings bioinformatics and biological data analysis into geospatial context, covering spatial omics, landscape genetics, phylogeography, and microbial ecology. Its README positions the module as the biology vertical of the framework, and it supplies container infrastructure (`Dockerfile`, `docker-compose.yml`) for reproducible biological workloads.

**Public API.** The `geo_infer_bio` package exports `SequenceAnalyzer` (core sequence analysis), validation and visualization utilities via `DataValidator` and `BioVisualizer`, and three environmental dataset families pairing a processor with a dataset type: `ClimateDataProcessor`/`ClimateDataset`, `MicrobiomeDataLoader`/`MicrobiomeDataset`, and `SoilDataIntegrator`/`SoilDataset`. These connect biological sequences and communities to the climatic and soil context in which they occur — the module's distinctive geospatial angle.

The tests exercise sequence analysis, environmental dataset loading, and validation utilities. The shared repository runner selects this module with `--module BIO`.

**Theme role.** Domain sciences: BIO is the life-science band, showing how the framework's data backbone and spatial primitives extend into ecological and molecular domains that require both biological pipelines and geographic context.
