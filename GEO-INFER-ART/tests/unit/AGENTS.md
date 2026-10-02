# Agent Instructions: GEO-INFER-ART/tests/unit

## Scope

- Owning module: `GEO-INFER-ART`
- Python package: `geo_infer_art`
- Directory role: Unit workspace within `GEO-INFER-ART`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_art` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_algorithm_registry_security.py`
- `test_cli.py`
- `test_color_palette.py`
- `test_cultural_map.py`
- `test_generative_map.py`
- `test_geo_art.py`
- `test_place_art.py`
- `test_procedural_art.py`
- `test_style_transfer.py`
- `test_validators.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-ART/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
