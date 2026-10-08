# H3 v4 in GEO-INFER

GEO-INFER uses the `h3` Python package with the supported range
`h3>=4.5.0,<5`. H3 cells are compact spatial keys that support multi-resolution
aggregation, neighborhood queries, and reproducible spatial joins.

## Current API names

| Task | H3 v4 API |
| --- | --- |
| Coordinates to cell | `h3.latlng_to_cell(lat, lng, resolution)` |
| Cell to center | `h3.cell_to_latlng(cell)` |
| Cell boundary | `h3.cell_to_boundary(cell)` |
| Neighborhood | `h3.grid_disk(cell, k)` |
| Resolution | `h3.get_resolution(cell)` |
| Parent | `h3.cell_to_parent(cell, resolution)` |
| Children | `h3.cell_to_children(cell, resolution)` |
| Polygon fill | `h3.geo_to_cells(geojson, resolution)` |

The GEO-INFER-SPACE convenience layer exposes the common conversion functions:

```python
from geo_infer_space import cell_to_latlng, latlng_to_cell

cell = latlng_to_cell(45.5231, -122.6765, resolution=9)
latitude, longitude = cell_to_latlng(cell)
print(cell, latitude, longitude)
```

Use the SPACE backend or convenience layer in application code so backend
availability and input validation stay consistent. Direct `h3` calls are
appropriate in focused integration code and tests.

## Nested hierarchies

`geo_infer_space.nested.NestedH3Grid` owns opt-in multi-resolution closure. A
valid hierarchy has ordered resolutions, real H3 cells, complete parent/child
maps, no orphan children, and finite aggregation results.

```python
from geo_infer_space.nested import NestedH3Grid

grid = NestedH3Grid("portland")
hierarchy = grid.build_h3_hierarchy_from_cells(
    ["89283082803ffff"],
    resolutions=[7, 8, 9],
)
assert hierarchy["validation"]["is_valid"]
assert hierarchy["validation"]["orphan_count"] == 0
```

Nested H3 behavior is opt-in in ACT, SPACE, and runner paths. Do not silently
replace a flat grid with a hierarchy because it changes cell counts, memory
use, and output schemas.

## Resolution and coordinate rules

- Coordinates are `(latitude, longitude)` in WGS84 degrees at the H3 boundary.
- Projected CRS coordinates must be transformed before H3 conversion.
- Resolutions are integers from 0 through 15; higher resolution means smaller
  cells and potentially much larger cardinality.
- Validate the expected cell budget before polygon expansion or hierarchy
  construction.
- Treat pentagons and antimeridian-crossing geometries as explicit edge cases.

## Verify the supported runtime

Use the v4 table above. The repository H3 contract gate is:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
```

## Further reading

- [SPACE module guide](../../../modules/geo-infer-space.md)
- [ACT spatial inference guide](../../../modules/geo-infer-act.md)
- [H3 advanced examples](../../../../../GEO-INFER-SPACE/examples/h3_advanced_applications.py)
- [H3 contract validator](../../../../../GEO-INFER-TEST/validate_h3_active_inference_contract.py)
