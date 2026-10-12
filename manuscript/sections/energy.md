## GEO-INFER-ENERGY — Energy Systems Analysis and Grid Optimization

GEO-INFER-ENERGY covers energy systems analysis, renewable energy optimization, and grid management (module README). It belongs to the Spatial & Place-based theme, since every quantity it models — solar irradiance, wind regimes, grid topology, demand load — is anchored to geography.

The `geo_infer_energy` package exports public symbols from `src/geo_infer_energy/__init__.py`. Resource assessment comes first: `RenewableResourceAssessor` evaluates sites, supported by the enumerations `RenewableType` and `SuitabilityClass` and the dataclass `RenewableSite`. Grid-side analysis is carried by `EnergyGridOptimizer`, `EnergyDemandForecaster`, and `EnergyInfrastructurePlanner`. Sustainability accounting is provided by `CarbonFootprintAnalyzer`, and technology-specific analysis by `SolarAnalyzer` and `WindAnalyzer`. The trio of assessor, optimizer, and forecaster forms the module's analytical spine: site suitability feeds infrastructure planning, which feeds grid optimization under forecast demand.

The unit and integration tests exercise resource assessors, analyzers, and demand forecasting. Examples illustrate assessment-to-optimization workflows; deployment effectiveness and forecast calibration require separate evidence.
