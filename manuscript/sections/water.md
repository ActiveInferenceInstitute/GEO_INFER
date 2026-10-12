## GEO-INFER-WATER — Water Resources and Hydrology

GEO-INFER-WATER provides water resources management, hydrology, and water quality monitoring (per its `README.md`). Its package `geo_infer_water` contains `api/`, `core/`, and `utils/` for its water-domain interfaces.

The public interface, verified from `__init__.py`, exports components spanning modeling, assessment, and planning. Process modeling: `HydrologicalModeler` for catchment-scale water movement, `WatershedDelineator` for deriving basin boundaries from terrain, `WaterBalanceModeler` for precipitation–runoff–storage accounting, and `InfiltrationModeler` for soil-water processes. Assessment: `WaterQualityAssessor` with `WaterSample`, `WaterBodyType`, and `PollutantType` typed records for monitoring contamination. Planning: `WaterInfrastructurePlanner` for infrastructure siting and `FloodDroughtAnalyzer` for hazard-side analysis of both surplus and scarcity.

The tests exercise hydrological modeling and assessment with constructed inputs and numerical reference values. Field calibration and operational water decisions require independent domain evidence.

Under the root README's Module Themes, WATER belongs to Spatial & Place-based with SPACE, PLACE, TIME, MARINE, and TRANSPORT. Its role there is the freshwater complement to MARINE's ocean scope: watershed delineation and water-balance modeling connect terrain (SPACE's indexing, PLACE's regional data) to human outcomes — quality, flooding, drought — that RISK's catastrophe models and TIME's temporal fusion consume. The module rounds out the framework's coverage of the physical environment on which its governance and risk layers operate.
