## GEO-INFER-ART — Geospatial Data as Artistic Expression

**Purpose.** GEO-INFER-ART transforms geospatial data into compelling artistic expressions through aesthetic visualizations and generative art systems. Its README defines the module as the creative boundary of the framework: the same spatial datasets analyzed elsewhere can be rendered as cartographic art, stylistic maps, and generative compositions.

**Public API.** The `geo_infer_art` package exports a coherent artistic toolchain: `GeoArt` with `MapStyle` as the core visualization pair; `StyleTransfer` and `ColorPalette` from the aesthetics layer; the generative pair `GenerativeMap` and `ProceduralArt`; the place-based `PlaceArt` and `CulturalMap` for culturally grounded cartography; plus `CustomAlgorithmFramework` for user-defined art algorithms and `PerformanceOptimizer` for rendering throughput. The subpackage layout (`core/`, `utils/`) under `geo_infer_art` mirrors this split; `geopandas>=0.10.0` is the declared geospatial dependency.

The tests exercise visualization, aesthetics, generation, and registry-based persistence. Hostile persistence payloads must fail without executing source or partly changing the registry.

**Theme role.** Domain sciences and human-facing applications: ART demonstrates the framework's reach beyond analytics into communication and culture, consuming standard geospatial inputs from the DATA band and expressing them for non-technical audiences.
