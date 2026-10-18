# Getting Started with INTRA Documentation Tools

GEO-INFER-INTRA ships reproducible module previews and configuration utilities.
The documentation tree also contains conceptual material about knowledge
representation and integration. Those concepts do not establish an installed
knowledge-base server, ontology editor, or workflow service.

## Install and inspect

Run commands from the GEO_INFER repository root:

```bash
uv sync --frozen --package geo-infer-intra
uv run --frozen python -c "import geo_infer_intra; print(geo_infer_intra.__version__)"
```

For the shared script, CLI, and MCP adapters, install GEO-INFER-API and the
modules you intend to call; follow the [installation guide](../../getting_started/installation_guide.md)
and [agent interface guide](../../integration/agent_interfaces.md).
`geo-infer-intra` is a package name, not a database or user-administration CLI.
INTRA's package metadata declares no console scripts. PostgreSQL and Redis are
not prerequisites for the local preview/configuration examples below.

## Generate a real preview bundle

```python
from pathlib import Path
from geo_infer_intra import generate_module_preview_suite

artifacts = generate_module_preview_suite("SPACE", Path("output/intra-quickstart"))
assert artifacts.html_path.is_file()
assert artifacts.svg_path.is_file()
assert artifacts.png_path.is_file()
assert artifacts.manifest_path.is_file()
print(artifacts.manifest_path)
```

The suite writes HTML, SVG, PNG, and a JSON manifest containing input provenance
and each rendered artifact's byte count and SHA-256. It uses real H3 geometry
for an **illustrative module overview**, not measured observations or a model
result. Static files remain usable without a server. The interactive HTML
references external Leaflet assets and OpenStreetMap tiles; loading those
resources requires network access. The HTML includes a static fallback.

## Use configuration utilities explicitly

```python
from geo_infer_intra.utils.config import get_config_value, merge_configs

base = {"general": {"debug_mode": False, "log_level": "INFO"}}
override = {"general": {"debug_mode": True}}
merged = merge_configs(base, override)
assert get_config_value(merged, "general.debug_mode") is True
assert get_config_value(base, "general.debug_mode") is False
assert get_config_value(merged, "general.log_level") == "INFO"
```

`load_config(path)` reads JSON or YAML; `validate_config(mapping)` checks the
packaged configuration JSON Schema and returns `(is_valid, error_message)`.
Configuration keys describing a server, ontology, or workflow do not themselves
start a service or implement those components. Call explicit utilities and
inspect their results rather than assuming configuration implies runtime
capability.

## Verify and continue

```bash
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module INTRA
uv run --frozen python GEO-INFER-TEST/validate_documentation.py --strict
```

- [First spatial analysis](../../getting_started/first_analysis.md) uses the
  owning SPACE and ACT packages.
- [Geometric algorithms](../algorithms/geometric_algorithms.md) distinguishes
  planar geometry from geodesic measurement.
- [Spatial indexing](../algorithms/spatial_indexing.md) uses real H3 interfaces.
- [Ontology concepts](../../ontology/index.md) explains design considerations
  without claiming an installed ontology toolkit.
- [Issue tracker](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues)
  is the repository's tracked route for reproducible problems and proposals.
