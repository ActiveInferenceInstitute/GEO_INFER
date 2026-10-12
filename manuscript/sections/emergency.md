## GEO-INFER-EMERGENCY — Emergency Management and Disaster Response

GEO-INFER-EMERGENCY provides emergency management and disaster response capabilities for geospatial systems (module README). It sits in the Spatial & Place-based theme, applying the framework's spatial machinery to time-critical scenarios — evacuation, resource deployment, and search-and-rescue — where geographic accuracy directly affects outcomes.

The module exposes five coordinated public classes from `src/geo_infer_emergency/__init__.py`: `EmergencyCoordinator` as the central orchestrator, `ResourceDeployer` for positioning response assets, `EvacuationPlanner` for routing populations away from hazard zones, `SituationalAwareness` for maintaining a live operational picture, and `SearchAndRescue` for locating and recovering affected people. The narrow API is deliberate: emergency workflows value a small, well-tested command surface over broad configurability, and the module's `pyproject.toml` dependencies are correspondingly focused.

The tests exercise coordination, evacuation, resources, situational awareness, search and rescue, and geographic constraints. `test_acceptance_emergency.py` supplies constructed response scenarios. These local scenarios do not establish operational safety or emergency response efficacy.
