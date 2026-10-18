# Ontology Design and Available Tooling

The ontology pages are conceptual guidance. GEO-INFER-INTRA currently implements
module preview generation and configuration utilities; its source has no
OntologyManager, SchemaBuilder, graphical concept editor, RDF database, or
ontology-specific console entrypoint. Earlier examples that imported those
classes from an umbrella namespace did not name installed interfaces.

## Installed INTRA surfaces

| Surface | Actual role |
| --- | --- |
| `geo_infer_intra.generate_module_preview_suite` | Write one module's illustrative HTML/SVG/PNG previews and manifest |
| `geo_infer_intra.generate_all_module_previews` | Write preview bundles for the module profile registry |
| `geo_infer_intra.utils.config.load_config` | Read an explicitly selected JSON/YAML configuration file |
| `geo_infer_intra.utils.config.validate_config` | Validate a configuration mapping against the packaged JSON Schema |
| `geo_infer_intra.utils.config.get_config_value` | Read a nested configuration value with an explicit missing-key policy |
| `geo_infer_intra.utils.config.merge_configs` | Merge nested mappings with override precedence |

A configuration section named `ontology`, `knowledge_base`, or `workflow` does
not implement the corresponding runtime. Inspect the owning source and public
signature before treating a configuration key as a capability.

## Command line interface

Use the shared GEO-INFER-API adapter after installing that package and the
module dependencies:

```bash
uv run --frozen geo-infer catalog INTRA --query config --limit 20
uv run --frozen geo-infer describe geo_infer_intra.utils.config:validate_config
uv run --frozen geo-infer call geo_infer_intra.utils.config:get_config_value \
  --args '[{"general":{"log_level":"INFO"}},"general.log_level"]'
```

The final command returns the value inside the shared JSON result envelope.
There is no `geo-infer-intra ontology`, database migration, user administration,
or ontology-validation CLI. MCP uses the same public targets through `geo_call`;
see [script, CLI, MCP, and skills](../integration/agent_interfaces.md).

## Choosing external semantic tools

Select tools for a declared task and qualify them separately:

- An RDF serializer/store represents and retrieves a graph.
- An OWL reasoner evaluates consequences under a supported logic/profile.
- A SHACL implementation checks supplied graph data against shapes.
- An ontology editor supports human authoring and review.
- A GeoSPARQL implementation supports its advertised geospatial query features.

These are different responsibilities. A successful serializer import does not
qualify a reasoner, endpoint, or editor. Record exact versions, inputs, supported
features, configuration, and executed results before claiming interoperability.
The standards are defined in [RDF 1.1](https://www.w3.org/TR/rdf11-concepts/),
[OWL 2](https://www.w3.org/TR/owl2-primer/),
[SHACL](https://www.w3.org/TR/shacl/), and
[GeoSPARQL 1.1](https://docs.ogc.org/is/22-047r1/22-047r1.html).

## Validation and contribution

Use [modeling](ontology_modeling.md) to define terms,
[custom constraints](custom_rules.md) to separate acceptance from inference,
and [integration](integration.md) to map those terms to real package fields.
Keep future runtime proposals in tracked issues rather than presenting them as
already installed tools. A new implementation needs an owning module, declared
dependencies, public contracts, executable examples, and meaningful tests.

```bash
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module INTRA
uv run --frozen python GEO-INFER-TEST/validate_documentation.py --strict
```
