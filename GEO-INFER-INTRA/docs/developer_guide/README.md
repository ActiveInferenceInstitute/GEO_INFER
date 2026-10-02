# GEO-INFER-INTRA/docs/developer_guide

Developer Guide workspace within `GEO-INFER-INTRA`.

## Contents

- `conftest_template.py`
- `autonomous_agent_guide.md`
- `code_structure.md`
- `contributing.md`
- `index.md`
- `repo_guidelines.md`
- `testing_guide.md`

## Public Interface

- `conftest_template.py:sample_coordinates` (function)
- `conftest_template.py:sample_h3_cells` (function)
- `conftest_template.py:sample_geodataframe` (function)
- `conftest_template.py:sample_time_series` (function)
- `conftest_template.py:sample_raster` (function)
- `conftest_template.py:active_inference_state` (function)
- `conftest_template.py:tmp_spatial_dir` (function)

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-intra`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
