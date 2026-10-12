## GEO-INFER-MARINE — Marine and Coastal Analysis

GEO-INFER-MARINE provides marine and oceanographic analysis, coastal management, and marine ecosystem monitoring as a Python package (`geo_infer_marine`, under `src/geo_infer_marine/` with `core/` and `utils/` subpackages). It packages the ocean domain as code: observational processing, coastal geometry, ecosystem state, and spatial planning each get dedicated engines rather than ad hoc notebooks.

The public interface, verified from `__init__.py`, centers on exported classes: `OceanographicDataProcessor` for observational data ingestion and cleanup, `CoastalAnalyzer` for shoreline and coastal-zone analysis, `MarineEcosystemModeler` with `MarineHabitatType` and `SpeciesData` for ecosystem representation, `SeaLevelAnalyzer` for sea-level trend work, `MarineSpatialPlanner` for ocean-space allocation, `OceanCurrentModeler` for circulation modeling, `MarineWaterQuality` for contamination assessment, and `CoralReefAssessor` for reef condition scoring.

The tests exercise the exported engines against constructed marine fixtures. They provide numerical and interface checks, without establishing oceanographic calibration or coastal management outcomes.

Within the repository's theme taxonomy (root `README.md`, Module Themes), MARINE belongs to Spatial & Place-based, alongside SPACE, PLACE, TIME, WATER, and TRANSPORT. It extends the spatial theme to the ocean domain: the same discipline of indexed, tested, deterministic geospatial analysis applied to coastlines, currents, and marine habitats.
