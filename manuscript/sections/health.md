## GEO-INFER-HEALTH — Spatial Epidemiology and Healthcare Accessibility

GEO-INFER-HEALTH covers epidemiology, healthcare accessibility analysis, disease surveillance, and spatial health risk assessment (module README). It is a member of the Governance, Risk & Domain theme, applying the framework's spatial-inference machinery to public health questions where location is a first-class epidemiological variable.

The `geo_infer_health` package exports a coherent three-part surface from `src/geo_infer_health/__init__.py`. Three core analyzers — `DiseaseHotspotAnalyzer` for clustering outbreak signal, `HealthcareAccessibilityAnalyzer` for measuring service reach, and `EnvironmentalHealthAnalyzer` for exposure assessment — carry the domain logic. They operate on typed data models: `Location`, `HealthFacility`, `DiseaseReport`, `PopulationData`, and `EnvironmentalData`, giving health records an explicit geographic schema. Utilities `haversine_distance` and `create_bounding_box` provide the geometric primitives, and `api_router` exposes the analyzers over HTTP, matching the framework-wide pattern of CLI-plus-API surfaces.

The unit and integration tests exercise analyzers and their data-model invariants. Examples demonstrate surveillance and accessibility workflows; clinical or public-health effectiveness requires independent validation.

Within the manuscript, HEALTH illustrates the framework's public-health posture: hotspot detection is a spatial statistics problem, accessibility a network-and-distance problem, and environmental exposure a data-fusion problem — three established geospatial idioms, each realized as a tested, importable class rather than a notebook.
