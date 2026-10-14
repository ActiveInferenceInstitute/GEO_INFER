# Cascadia: a configured agent/process model matrix

A matrix names exported categorical GNN models and an ordered selection of
agent/process pairs. Every case shares the same initial true state, observation
horizon, process seeds and agent seed. The runner delegates each case to the
[real model-pair runner](cascadia_model_mismatch.md), which conditions beliefs
and evaluates policies with `inferactively-pymdp` 1.0.4. No inference equations
are duplicated in the matrix entry point.

## Declare and run

[model_matrix_design.json](../examples/cascadia/model_matrix_design.json)
selects the reference and uninformative-odor exports in a retained seed study.
It declares both matched cases and both directions of mismatch. From the
GEO_INFER checkout, use an explicitly selected external environment:

```bash
uv run --no-sync python GEO-INFER-ACT/examples/cascadia/run_model_matrix.py \
  --design GEO-INFER-ACT/examples/cascadia/model_matrix_design.json \
  --models-root GEO-INFER-ACT/examples/outputs/STUDY \
  --output GEO-INFER-ACT/examples/outputs/MATRIX_NEW
uv run --no-sync python GEO-INFER-ACT/examples/cascadia/run_model_matrix.py \
  --check GEO-INFER-ACT/examples/outputs/MATRIX_NEW --replay
```

`STUDY` must contain each model path declared in the design. Reproduce each
input with GNN's `docs/other/geo_infer/qualify_export.py` before interpreting
it as an export of that notation. The matrix can also consume other validated
categorical artifacts; the LectureCreate companion is Cascadia-specific.

The design is strict, duplicate-free JSON with schema
`geo-infer-act/model-matrix-design/1`. Models have `name` and a portable
relative JSON `artifact` path. Cases have `name`, `agent` and `process` names.
The `run` object uses the pair runner's `initial_state`, `steps`, `seeds` and
`agent_seed` fields. Names are lowercase identifiers; their order is retained.
All declared models must be selected. Duplicate names, duplicate pairs,
unknown fields, undeclared references, numeric boolean aliases, path traversal
and symlink inputs are refused. The admission limits are eight models,
64 cases, 64 process seeds, 64 KiB of design JSON and a total of 8192
observation steps across cases and seeds.

Before creating output, the runner snapshots every selected input and checks
equal dimensions, ordered states, spatial metadata and physical clocks, plus
the initial-state range. Observation and action index meanings remain an
explicit modelling assumption: their dimensions alone cannot establish
semantic alignment. Cascadia's compiler preserves those ordered alphabets
between the selected cue variants.

## Retained records and failures

The create-only edition contains the design, canonical model snapshots,
`pairs/CASE/` records, a deterministic `comparison.html`, `comparison.svg`
and a closed `matrix_manifest.json`. The manifest binds raw input-file SHA-256,
canonical artifact SHA-256, each child manifest, all output files and the
execution/display owners. Raw file and canonical artifact hashes differ when
JSON formatting differs; neither hash alone proves the original exporter ran.

A process can generate a cue to which the agent assigns zero evidence. That
case fails explicitly. The matrix retains completed predecessors, partial
case records and `failure.json`, and emits no successful matrix manifest.
Choose a new output name after correcting inputs so the failed attempt survives.
Other file, display or source-drift failures also propagate without a success
manifest; they may occur before a case failure record can be written.

## Read the comparisons

The [retained matrix](../examples/cascadia/data/model_matrix_20261014.html),
[SVG](../examples/cascadia/data/model_matrix_20261014.svg) and
[complete archive](../examples/cascadia/data/model_matrix_20261014.zip)
contain the declared four-case, two-seed, 48-step execution. Extract the archive
to a new directory and pass its `cascadia_model_matrix_20261014_v2/` child to
`run_model_matrix.py --check ... --replay`.

The HTML and SVG use the actual retained observation-time posterior and
cue-generating true state. Each selected seed remains a trajectory, rather
than becoming a population estimate. The HTML compares all cases with seed
and metric selectors and a step slider; elapsed time comes from the artifact clock.
An arrival horizon counts observation steps and can right-censor a path.
Mismatch direction matters: changing the process observation law changes
what is sampled; changing the agent observation law changes what is inferred.

The default `--check` checks supplied-manifest integrity, design/model/case
bindings and exact HTML freshness. It does not authenticate the manifest or
execute inference. `--replay` additionally checks current owner bytes and
re-executes every pair, comparing retained trajectories, policy terms, scores
and summaries. SVG bytes are hash-bound; rerendering and visual inspection
are separate checks. Neither mode proves floating-point refinement, calibration,
ecological validity or a native Lean theorem for this matrix.

## Across the four repositories

GNN's `docs/other/geo_infer/model_matrices.md` explains producer custody.
fep_formal's `specs/cascadia-experiments/matrix-proof-obligations.md` separates
agent and process hypotheses. LectureCreate's
`examples/cascadia_geo_act/matrix_companion.py` invokes all three owning tools
through explicit environments, retaining exporter qualifications and exact
proof-impact worksheets beside the comparison. Its matrix companion is a new
teaching edition, separate from the frozen reference video.

```bash
uv run --no-sync pytest GEO-INFER-ACT/tests/integration/test_model_matrix.py -q --no-cov
```
