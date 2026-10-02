# GEO-INFER-INTRA/src/geo_infer_intra/core/documentation

Documentation workspace within `GEO-INFER-INTRA`.

## Contents

- `__init__.py`
- `visual_preview.py`

## Public Interface

- `visual_preview.py:SpatialPreviewArtifacts` (class)
- `visual_preview.py:render_svg_card` (function)
- `visual_preview.py:render_leaflet_html` (function)
- `visual_preview.py:render_png_card` (function)
- `visual_preview.py:generate_module_preview_suite` (function)
- `visual_preview.py:generate_all_module_previews` (function)
- `visual_preview.py:build_previews` (function)

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
