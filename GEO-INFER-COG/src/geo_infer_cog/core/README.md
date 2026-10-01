# GEO-INFER-COG/src/geo_infer_cog/core

Core workspace within `GEO-INFER-COG`.

## Contents

- `__init__.py`
- `cognitive_engine.py`
- `spatial_memory.py`
- `spatial_perception.py`
- `spatial_reasoning.py`

## Public Interface

- `cognitive_engine.py:CognitiveState` (class)
- `cognitive_engine.py:CognitiveProcessingEngine` (class)
- `spatial_memory.py:SpatialMemoryItem` (class)
- `spatial_memory.py:MemoryConsolidation` (class)
- `spatial_memory.py:SpatialMemoryModel` (class)
- `spatial_perception.py:SpatialPercept` (class)
- `spatial_perception.py:AttentionModel` (class)
- `spatial_perception.py:SpatialPerceptionModel` (class)
- `spatial_reasoning.py:SpatialRelation` (class)
- `spatial_reasoning.py:ReasoningStep` (class)
- `spatial_reasoning.py:SpatialReasoningEngine` (class)

## Module Metadata

- Module: `GEO-INFER-COG`
- Package: `geo_infer_cog`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-cog`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

## Dependencies

- `numpy>=1.20.0`
- `networkx>=2.6`
- `pyyaml>=5.4`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
