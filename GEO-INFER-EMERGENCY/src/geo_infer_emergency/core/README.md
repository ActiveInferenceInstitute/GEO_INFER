# GEO-INFER-EMERGENCY/src/geo_infer_emergency/core

Core workspace within `GEO-INFER-EMERGENCY`.

## Contents

- `__init__.py`
- `awareness.py`
- `coordinator.py`
- `evacuation.py`
- `geo.py`
- `resources.py`
- `sar.py`

## Public Interface

- `awareness.py:ThreatLevel` (class)
- `awareness.py:DataSource` (class)
- `awareness.py:SensoryInput` (class)
- `awareness.py:LayerConfig` (class)
- `awareness.py:SituationalAwareness` (class)
- `coordinator.py:IncidentType` (class)
- `coordinator.py:IncidentScale` (class)
- `coordinator.py:Incident` (class)
- `coordinator.py:Agency` (class)
- `coordinator.py:IncidentCommand` (class)
- `coordinator.py:EmergencyCoordinator` (class)
- `evacuation.py:EvacuationLevel` (class)
- `evacuation.py:EvacuationZone` (class)
- `evacuation.py:Shelter` (class)
- `evacuation.py:EvacuationRoute` (class)
- `evacuation.py:EvacuationPlanner` (class)
- `geo.py:haversine_distance_km` (function)
- `resources.py:ResourceStatus` (class)
- `resources.py:ResourceType` (class)
- `resources.py:Resource` (class)

## Module Metadata

- Module: `GEO-INFER-EMERGENCY`
- Package: `geo_infer_emergency`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-emergency`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EMERGENCY`

## Dependencies

- `networkx>=2.6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EMERGENCY
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
