## GEO-INFER-PLACE — Place-Based Analysis

GEO-INFER-PLACE is a comprehensive place-based analysis framework providing deep insights into specific geographic locations and regional systems (per its `README.md`). Unlike generic spatial tooling, it ships concrete study regions: `locations/`, `data/`, and `config/` directories at the module root, plus worked outputs (`cascadia_output/`, `del_norte_dashboard/`, `del_norte_output/`). The package `geo_infer_place` contains `core/`, `hydrography/`, `locations/`, `config/`, and `utils/`.

The public interface, verified from `__init__.py`, is broad. Analysis modules: `PlaceInterface`, `PlaceDataManager`, `PlaceTemporalAnalyzer`, `BaseAnalysisModule`, `InteractiveVisualizationEngine`, and region-specific engines — `ForestHealthMonitor`, `CoastalResilienceAnalyzer`, `FireRiskAssessor`, `SeismicHazardAnalyzer`, and the `CascadianAgriculturalH3Backend`. External data: `CaliforniaAPIManager` with `NOAAClient`, `CALFIREClient`, `USGSClient`, `USGSEarthquakeClient`, and `CDECClient`, wrapped by `CachedAPIWrapper`. Spatial indexing: a full H3 v4 helper surface (`latlng_to_cell`, `grid_disk`, `polygon_to_cells`, `compact_cells`, and more).

The tests exercise regional analysis, data acquisition, and the PLACE-owned Cascadia package. Local layers and reproducible fixtures are distinct from licensed datasets and authenticated external services.

Under the root README's Module Themes, PLACE belongs to Spatial & Place-based with SPACE, TIME, MARINE, WATER, and TRANSPORT. Its role is depth over generality: SPACE provides the indexing substrate, while PLACE demonstrates the framework's full evidence loop — real regions, real agency data, hazard analytics, and dashboard outputs.
