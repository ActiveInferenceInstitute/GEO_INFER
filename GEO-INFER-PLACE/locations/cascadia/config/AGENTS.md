# Agent Instructions: GEO-INFER-PLACE/locations/cascadia/config

## Scope

- Owning module: `GEO-INFER-PLACE`
- Python package: `geo_infer_place`
- Directory role: Config workspace within `GEO-INFER-PLACE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_place` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `analysis_config.yaml`
- `ca_del_norte_boundary.geojson`
- `ca_humboldt_boundary.geojson`
- `ca_lassen_boundary.geojson`
- `cascadia_climate_zones.yaml`
- `cascadia_config.yaml`
- `cascadia_ecoregions.yaml`
- `cascadia_indigenous_territories.yaml`
- `cascadia_layers.provenance.json`
- `cascadia_major_watersheds.geojson`
- `cascadia_regional_source_metadata.json`
- `cascadia_salmon_esus.yaml`
- `cascadia_subduction_zone.geojson`
- `cascadia_tectonics.metadata.xml`
- `cascadia_tectonics.source.json`
- `cascadia_volcanoes.geojson`
- `cascadia_volcanoes.source.json`
- `cascadia_watersheds.source.json`
- `county_boundaries.yaml`
- `data_cleanup_config.json`
- `data_urls.json`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
