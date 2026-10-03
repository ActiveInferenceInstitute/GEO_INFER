# Geospatial Applications of Active Inference

GEO-INFER-ACT treats H3 geospatial Active Inference as a first-class contract:
real `h3>=4.5.0,<5` cells, `inferactively-pymdp==1.0.3` categorical belief and
policy inference, normalized per-cell beliefs, finite free-energy and
negative-EFE diagnostics, GIS-ready outputs, and visualizations that are
referenced from each run manifest.

## Geospatial Active Inference Architecture

```mermaid
flowchart TB
    subgraph "Spatial Index"
        SPACE["GEO-INFER-SPACE H3 backend"]
        H3["H3 4.5 cell IDs and boundaries"]
    end
    subgraph "ACT Core"
        GM["GenerativeModel H3 beliefs"]
        PYMDP["pymdp 1.0.3 adapter"]
        AIM["ActiveInferenceModel grid inference"]
        SA["SpatialActiveInferenceAgent"]
        MA["MultiAgentModel H3 lattice"]
    end
    subgraph "Outputs"
        DATA["CSV, JSON, GeoJSON"]
        VIZ["PNG and HTML visualizations"]
        MANIFEST["manifest.json validation"]
    end
    SPACE --> H3 --> GM --> PYMDP
    PYMDP --> AIM
    H3 --> SA
    GM --> AIM --> DATA
    SA --> DATA
    MA --> DATA
    DATA --> MANIFEST
    VIZ --> MANIFEST
```

## H3 Perception-Action Sequence

```mermaid
sequenceDiagram
    participant Runner as geo-infer-act-run
    participant Adapter as H3Adapter
    participant Model as GenerativeModel
    participant Agent as ActiveInferenceModel
    participant Analyzer as ActiveInferenceAnalyzer
    Runner->>Adapter: create or validate H3 4.5 cells
    Runner->>Model: update_h3_beliefs(observations)
    Model-->>Runner: H3BeliefUpdateResult
    Runner->>Agent: infer_over_h3_grid(observations)
    Agent-->>Runner: H3GridInferenceResult with pymdp metadata
    Runner->>Analyzer: record beliefs, actions, FE, EFE
    Analyzer-->>Runner: full_history.json and analysis files
```

## H3 Result and Schema Contracts

```mermaid
classDiagram
    class H3SpatialConsistency {
        global_coherence
        neighbor_correlations
        cell_count
        edge_count
    }
    class H3BeliefUpdateResult {
        h3_beliefs
        average
        aggregate_free_energy
    }
    class H3GridInferenceResult {
        cell_results
        aggregate_free_energy
        metadata
    }
    class RunManifest {
        schema_version
        generated_files
        metrics
        validation
    }
    H3BeliefUpdateResult --> H3SpatialConsistency
    H3GridInferenceResult --> H3SpatialConsistency
    H3GridInferenceResult --> RunManifest
```

## Spatial Belief Propagation and Neighbor Diffusion

```mermaid
flowchart LR
    O["Cell observation"] --> L["Local Bayesian update"]
    L --> P["Posterior belief q(s)"]
    P --> N["Neighbor belief average"]
    N --> D["Precision-weighted diffusion"]
    D --> C["Spatial coherence metric"]
    C --> FE["Spatial free-energy diagnostic"]
    FE --> A["Spatial EFE action selection"]
```

## Runner Output Manifest Pipeline

```mermaid
flowchart LR
    C["RunConfig YAML or CLI flags"] --> R["Scenario runner"]
    R --> H["H3 diagnostics"]
    R --> G["h3_cells.geojson"]
    R --> S["step_metrics.csv"]
    R --> P["PNG and HTML maps"]
    H --> M["manifest.json"]
    G --> M
    S --> M
    P --> M
    M --> V["validator checks files, sizes, finite metrics"]
```

## Multi-Agent H3 Lattice Coordination

```mermaid
flowchart TB
    subgraph "H3 Lattice"
        C1["Cell agent 1"]
        C2["Cell agent 2"]
        C3["Cell agent 3"]
    end
    OBS["Environmental observations"] --> C1
    OBS --> C2
    OBS --> C3
    C1 <-->|belief sharing| C2
    C2 <-->|belief sharing| C3
    C1 --> COH["Coordination coherence"]
    C2 --> COH
    C3 --> COH
    COH --> OUT["lattice history and diagnostics"]
```

## Validation Flow

```mermaid
flowchart TD
    A["Static method inventory"] --> B["Docstring and H3 v4 checks"]
    B --> C["H3 scenario smoke run"]
    B --> D["Spatial scenario smoke run"]
    C --> E["Normalize beliefs and finite metrics"]
    D --> E
    E --> F["Validate JSON, CSV, GeoJSON, PNG, HTML"]
    F --> G["Validate Mermaid docs and local links"]
    G --> H["Geospatial contract OK"]
```

## Method Validation Gates

Every public geospatial method follows the same gate sequence before returning
data to callers: validate cell identity, normalize probability vectors, compute
finite diagnostics, and expose either the default dictionary result or the
typed result object.

```mermaid
flowchart LR
    INPUT["H3 observations"] --> V["Validate H3 v4 cells"]
    V --> S["Check cells belong to model or agent lattice"]
    S --> N["Normalize belief and observation vectors"]
    N --> FE["Compute VFE and EFE diagnostics"]
    FE --> C["Compute H3SpatialConsistency"]
    C --> D{"return_result"}
    D -->|false| DICT["Dictionary contract"]
    D -->|true| TYPED["Typed dataclass contract"]
```

## Visualization and Schema Traceability

Geospatial visualizations are not standalone side effects. Each generated map or
chart is referenced by `manifest.json`, and the manifest records the schema
version, scenario, metrics, and validation status for the run.

```mermaid
flowchart TB
    OBS["Real H3 observations"] --> RUN["H3 or spatial runner"]
    RUN --> HIST["data/full_history.json"]
    RUN --> MET["data/step_metrics.csv"]
    RUN --> CELLS["data/h3_cells.csv and GeoJSON"]
    RUN --> DIAG["data/h3_diagnostics.json"]
    HIST --> PLOTS["FE, EFE, entropy, coherence plots"]
    CELLS --> MAPS["Static PNG and interactive HTML H3 maps"]
    DIAG --> SUMMARY["analysis/run_summary.json"]
    PLOTS --> MAN["manifest.json"]
    MAPS --> MAN
    SUMMARY --> MAN
    MAN --> CHECK["schema and artifact validation"]
```

## Figure Artifact Metadata Pipeline

Every Active Inference figure is written through the package runner artifact
writer. The runner embeds a compact metadata record in the visualization file,
writes a JSON metadata sidecar, writes the exact plotted data to a CSV or JSON
sidecar, and then enriches `manifest.json` with the figure digest and sidecar
links.

```mermaid
flowchart LR
    RUN["Scenario runner"] --> FIG["Figure artifact writer"]
    CFG["RunConfig and CLI provenance"] --> FIG
    DATA["Source data files"] --> FIG
    METRICS["Plotted metric rows"] --> FIG
    FIG --> EMBED["Embedded PNG or HTML metadata"]
    FIG --> META["figure.metadata.json"]
    FIG --> SIDE["figure.data.csv or figure.data.json"]
    EMBED --> MAN["manifest.generated_files"]
    META --> MAN
    SIDE --> MAN
```

## Figure Sidecar Traceability

The sidecar contract is intentionally redundant so downstream tools can audit
figures without rerunning the scenario. PNG artifacts carry
`geo_infer_act_metadata` in the image metadata. HTML maps carry a
`<script type="application/json" id="geo-infer-act-figure-metadata">` block.
Both embedded records match the JSON sidecar for schema version, scenario, and
figure identity.

```mermaid
flowchart TB
    VIZ["visualizations/name.png or .html"] --> EMBED["Embedded metadata"]
    VIZ --> META["visualizations/name.metadata.json"]
    VIZ --> FDATA["visualizations/name.data.csv or .data.json"]
    META --> RUNCFG["Run config: seed, timesteps, H3 resolution"]
    META --> PROV["Package version, scenario, generated timestamp"]
    META --> SOURCE["Data source file list"]
    META --> METRIC["Plotted metrics and alt text"]
    FDATA --> VALUES["Finite plotted numeric values"]
    SOURCE --> RAW["data/full_history, step_metrics, H3 cells, diagnostics"]
```

## Manifest Visualization Validation

`manifest.generated_files` keeps the previous `path` and `size_bytes` fields and
adds stable artifact metadata for every generated file. Visualization entries
also require `figure_metadata_path`, `figure_data_path`, `width_px` and
`height_px` when available, `data_sources`, `plotted_metrics`, `description`,
and `alt_text`.

```mermaid
flowchart TD
    MAN["manifest.json"] --> FILES["generated_files[]"]
    FILES --> BASIC["path, size_bytes, artifact_type, mime_type, sha256"]
    FILES --> TYPE{"artifact_type == visualization"}
    TYPE -->|yes| REQUIRED["metadata sidecar, data sidecar, sources, metrics, alt text"]
    REQUIRED --> EXISTS["Check referenced files exist and are non-empty"]
    EXISTS --> HASH["Verify figure SHA-256"]
    HASH --> EMBED["Check embedded PNG or HTML metadata"]
    EMBED --> FINITE["Check finite plotted data values"]
    FINITE --> PASS["Geospatial contract passes"]
```

## Canonical APIs

| API | Purpose | Return contract |
| --- | --- | --- |
| `GenerativeModel.enable_h3_spatial(...)` | Build an H3-indexed spatial model | Populates valid H3 cells and neighbor graph |
| `GenerativeModel.update_h3_beliefs(..., return_result=True)` | Update per-cell beliefs | `H3BeliefUpdateResult` |
| `ActiveInferenceModel.apply_to_h3(..., return_result=True)` | Apply the active model to H3 observations | `H3BeliefUpdateResult` |
| `ActiveInferenceModel.infer_over_h3_grid(..., return_result=True)` | Score each H3 cell with one inference step | `H3GridInferenceResult` |
| `ActiveInferenceModel.trace_over_h3_grid(...)` | Build flat H3 research diagnostics | `SpatialInferenceTrace` |
| `ActiveInferenceModel.trace_over_nested_h3_grid(...)` | Build nested H3 research diagnostics with parent aggregates | `SpatialInferenceTrace` |
| `SpatialActiveInferenceAgent.step(..., return_result=True)` | Run spatial perception and action | `H3GridInferenceResult` |
| `SpatialActiveInferenceAgent.trace_step(...)` | Build trace diagnostics from a spatial-agent step | `SpatialInferenceTrace` |
| `MultiAgentModel.simulate_h3_lattice(...)` | Run distributed H3-cell agents | Per-timestep cell diagnostics |
| `geo_infer_act.runners.run_scenario(...)` | Run configured H3 or spatial workflows | `ScenarioRunResult` with `manifest.json` |

## Method Contracts

| Method | Required input | Validation and normalization | Diagnostics produced |
| --- | --- | --- | --- |
| `GenerativeModel.enable_h3_spatial` | GeoJSON-like boundary and H3 resolution | Requires non-empty real H3 cell set and builds a first-order H3 neighbor graph | Cell count, H3 resolution, graph edge count |
| `GenerativeModel.update_h3_beliefs` | Mapping of H3 cell to observation vector | Rejects invalid or out-of-model cells; normalizes posterior beliefs | `aggregate_free_energy`, global coherence, neighbor correlation |
| `ActiveInferenceModel.apply_to_h3` | H3 observations plus spatial generative model | Delegates to the canonical generative-model H3 update contract | `H3BeliefUpdateResult` in typed mode |
| `ActiveInferenceModel.infer_over_h3_grid` | H3 observation grid | Validates every H3 cell; restores original agent state after scoring | Per-cell `ActiveInferenceStepResult`, aggregate FE, selected policies |
| `ActiveInferenceModel.trace_over_h3_grid` | H3 observation grid and optional previous beliefs | Reuses a supplied grid result or runs the typed grid scorer | Per-cell, per-edge, and per-level `SpatialInferenceTrace` diagnostics |
| `ActiveInferenceModel.trace_over_nested_h3_grid` | Finest-resolution nested H3 observations | Requires an enabled nested hierarchy and parent aggregate beliefs | Parent/child residuals, cross-level consistency, and nested level diagnostics |
| `SpatialActiveInferenceAgent.spatial_perception` | Observations on the agent H3 lattice | Rejects unknown cells; normalizes beliefs after local and neighbor updates | Spatial free energy and belief history |
| `SpatialActiveInferenceAgent.step` | H3 lattice observations | Runs perception, EFE action selection, and typed consistency aggregation | `H3GridInferenceResult` in typed mode |
| `SpatialActiveInferenceAgent.trace_step` | Spatial-agent observations and optional grid result | Reuses the typed step result to avoid a second pymdp update | `SpatialInferenceTrace` with policy, flux, and coherence diagnostics |
| `MultiAgentModel.simulate_h3_lattice` | Timesteps and observation generator | Normalizes each cell observation and coordinates neighboring agents | Per-timestep beliefs, observations, FE, coordination history |
| `EnvironmentalActiveInferenceEngine.compute_spatial_priors` | Environmental variable and state count | Converts spatial autocorrelation into normalized per-cell priors | H3 cell prior distributions for downstream ACT models |

## Runnable H3 Workflow

```python
from pathlib import Path

from geo_infer_act.runners import RunConfig, run_scenario

result = run_scenario(
    RunConfig(
        scenario="h3",
        output_dir=Path("/tmp/geo-infer-act-h3"),
        seed=42,
        timesteps=4,
        h3_resolution=8,
        h3_ring_size=1,
        visualizations=True,
    )
)

print(result.manifest_path)
print(result.metrics["cell_count"])
```

The same contract applies to `scenario="spatial"`, which runs
`SpatialActiveInferenceAgent` on real H3 cells.

## Diagnostic Semantics

| Diagnostic | Meaning | Expected invariant |
| --- | --- | --- |
| `aggregate_free_energy` | Mean free-energy diagnostic over observed H3 cells | Finite float |
| `expected_free_energy` | Policy score balancing pragmatic and epistemic terms | Finite float |
| `belief_entropy` | Entropy of each normalized belief vector | Finite and nonnegative |
| `policy_entropy` | Entropy of the pymdp action posterior | Finite and nonnegative |
| `posterior_delta` | Absolute-sum change from the previous timestep's belief for the same cell | Finite and nonnegative |
| `belief_flux_divergence` | Neighbor-weighted entropy-gradient proxy for belief flow | Finite float |
| `local_coherence` | Similarity between a cell belief and same-resolution neighbors | Finite value between 0 and 1 for normal H3 paths |
| `cross_level_consistency` | Agreement between child belief and parent aggregate belief | Finite and nonnegative in nested mode |
| `global_coherence` | Cross-cell belief agreement summary | Bounded between 0 and 1 |
| `neighbor_correlations` | Agreement across H3 graph edges | Finite correlation-like value |
| `cell_count` | Count of valid real H3 cells in the run | Matches CSV and GeoJSON feature counts |
| `edge_count` | Undirected H3 neighbor edges in the active lattice | Nonnegative integer |

## Output Contract

Geospatial scenarios write the standard ACT runner outputs plus H3-specific
artifacts:

| Path | Content |
| --- | --- |
| `manifest.json` | Schema version, run config, metrics, generated files, validation status |
| `data/full_history.json` | Analyzer step history with beliefs, observations, actions, FE, EFE |
| `data/step_metrics.csv` | Timestep metrics including FE, EFE, entropy, coherence |
| `data/h3_cells.csv` | Cell centroid, resolution, final FE, EFE, entropy, action |
| `data/h3_cells.geojson` | Polygon FeatureCollection for GIS tools |
| `data/h3_diagnostics.json` | Per-timestep spatial consistency and aggregate FE |
| `data/pymdp_h3_diagnostics.json` | Per-cell pymdp version, posterior, negative EFE, VFE, entropy |
| `data/pymdp_policy_posteriors.csv` | CSV form of pymdp policy posterior and negative-EFE rows |
| `data/spatial_inference_trace.json` | JSON-safe per-cell, per-edge, per-level spatial trace |
| `data/spatial_research_statistics.json` | Run-level summaries, temporal slopes, policy switches, graph statistics, and nested residual summaries |
| `data/h3_cell_diagnostics.csv` | Flattened H3 cell trace diagnostics with lat/lng centroids |
| `data/h3_edge_diagnostics.csv` | Same-resolution H3 edge belief-distance diagnostics |
| `data/h3_hierarchy.csv` | Nested H3 parent-child closure rows |
| `data/nested_h3_diagnostics.json` | Nested per-timestep summaries and consistency diagnostics |
| `data/nested_h3_cell_diagnostics.csv` | Nested cell trace diagnostics including aggregate parent cells |
| `data/nested_h3_parent_child_diagnostics.csv` | Nested parent/child consistency and residual rows |
| `data/nested_h3_level_diagnostics.csv` | Per-resolution entropy, FE, policy entropy, flux, and coherence |
| `analysis/run_summary.json` | Summary metrics for dashboards and reports |
| `visualizations/h3_cell_metric_map.png` | Static H3 cell metric map |
| `visualizations/free_energy_evolution.png` | FE and EFE evolution |
| `visualizations/belief_entropy_coherence.png` | Entropy and coherence trend |
| `visualizations/interactive_h3_map.html` | Interactive cell map |
| `visualizations/pymdp_policy_free_energy.html` | pymdp VFE, negative-EFE, and action-confidence trend |
| `visualizations/h3_belief_flux_map.html` | Interactive belief-flux and posterior-delta map |
| `visualizations/h3_policy_surface.html` | Timestep-by-cell selected-action confidence surface |
| `visualizations/h3_policy_transitions.html` | Stacked selected-action counts by timestep |
| `visualizations/h3_spatial_autocorrelation.html` | H3 adjacency, edge-distance, and flux-balance diagnostics |
| `visualizations/h3_entropy_free_energy_phase.html` | Cell-level entropy/free-energy phase-space view |
| `visualizations/spatial_inference_research_report.html` | Research report linking statistics, trace data, and visualizations |
| `visualizations/nested_h3_level_map.html` | Nested per-resolution summary table |
| `visualizations/nested_h3_hierarchy_map.html` | Real-H3 parent/child boundary map with residuals |
| `visualizations/nested_h3_parent_child_residuals.html` | Nested parent-child cross-level residual chart |
| `visualizations/*.metadata.json` | Figure schema version, package version, scenario, run config, data sources, plotted metrics, alt text, digest, and dimensions |
| `visualizations/*.data.csv` / `*.data.json` | Exact finite rows or payload used to render the figure |

Each `generated_files` entry includes `artifact_type`, `mime_type`, and
`sha256`. Visualization entries additionally include the metadata sidecar path,
plotted-data sidecar path, data source paths, plotted metric names, a
description, and alt text. Static PNGs embed the same ACT metadata payload in
the image metadata; HTML maps embed it as structured JSON in the document head.

## Research Profile And Gallery

Research-profile runs are opt-in through
`RunConfig.parameters["research_profile"] = True` or the shared runner flag
`--research-profile`. This profile keeps real H3 cells and
`inferactively-pymdp==1.0.3`, but uses deterministic offline spatial fields plus
action-conditioned transition and preference matrices so policy posterior,
entropy, coherence, and belief-flux diagnostics can reflect observed variation.
A short environmental sample can legitimately select the same optimal action
at every cell. Research-profile configuration does not promise policy diversity.

Acceptance uses a separate explicit categorical fixture covering known policy
regions. Independent NumPy arithmetic checks frequency-evidence posteriors and
one-step policy scores: expected preference utility plus state-information gain,
then a unit-precision softmax with uniform policy prior. Each gallery cell's
actual score, probability vector, and selected action is checked too; the gallery
continues to use its original environmental inputs.

The pymdp `policy_beliefs` metadata and diagnostic `policy_belief_<state>` columns
retain the exact local posterior used for prospective policy evaluation. Spatial
diffusion and nested blending can change displayed cell beliefs afterward.

Generate the four-run gallery with:

```bash
uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py
```

The gallery writes flat H3, nested H3, flat spatial-agent, and nested
spatial-agent runs under
`GEO-INFER-ACT/examples/output/spatial_active_inference_gallery/` by default.
Use `uv run` for these commands; system Python may contain an older pymdp
distribution and is not the supported ACT/H3 runtime.

## Verification

From the repository root:

```bash
uv run python GEO-INFER-TEST/validate_act_geospatial_contract.py
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run --package geo-infer-act --extra dev geo-infer-act-run \
  --scenario h3 \
  --output-dir /tmp/geo-infer-act-h3 \
  --seed 42 \
  --timesteps 4
```

## Integration With GEO-INFER Modules

| Module | Integration |
| --- | --- |
| `GEO-INFER-SPACE` | H3 indexing, hierarchy, boundary, and neighborhood semantics |
| `GEO-INFER-BAYES` | Probabilistic inference foundations |
| `GEO-INFER-AGENT` | Agent orchestration using ACT contracts |
| `GEO-INFER-SIM` | Scenario simulation around ACT agents |

## Further Reading

- [Active Inference Overview](./active_inference_overview.md)
- [Free Energy Principle](./free_energy_principle.md)
- [Mathematical Framework](./mathematical_framework.md)
- [References](./references.md)

**Last Updated**: 2026-05-19
