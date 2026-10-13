# Cascadia experiment workflow

The example now separates the experiment design, model semantics, retained-run
qualification, and display projection. Begin with
[the model and evidence contract](cascadia_river_homing.md); the dated
[seed study](cascadia_seed_study_20261012.md) records the retained results.

## Module responsibilities

| Path | Responsibility |
| --- | --- |
| `models/river_homing.py` | Exact dyadic model compiler |
| `models/river_homing_study.py` | Conditions, bounded design, numerical measures and censoring |
| `models/river_homing_study_evidence.py` | Static agreement of saved run records |
| `runners/closed_loop.py` | Sampled process and public inference replay |
| `runners/study_visualization.py` | Typed finite display projection |
| `runners/study_diagnostics.py` | Recorded posterior scores and uncertainty |
| `examples/cascadia/cascadia_study.py` | Thin seed-grid orchestration |
| `examples/cascadia/validate_study.py` | Read-only manifest/grid/summary entry point |
| `examples/cascadia/study_explorer.py` | Qualified offline HTML entry point |

The model/runner package paths above are relative to `src/geo_infer_act/`.
The example's `study_contract.py` re-exports the package API for existing callers.
The study modules are explicit imports and use ACT's `examples` extra.

For score definitions, clock alignment, portable JSON output and mathematical
scope, read [recorded-run diagnostics](cascadia_diagnostics.md).

For distinct agent and process artifacts, fresh paired execution and replay,
read [model mismatch](cascadia_model_mismatch.md). A matched seed study changes
both process and agent together; it does not establish robustness to mismatch.

## Declare and execute a design

Work from the mounted GitVault checkout. Bind `TMPDIR` and `UV_CACHE_DIR`
to task directories on that volume outside the repository, and set
`UV_LINK_MODE=copy`; subprocess checks inherit those bindings. Repository scans
include example directories, so placing temporary test fixtures there can make
them appear to be production source. Keep earlier attempts and receipts intact.

[study_design.json](../examples/cascadia/study_design.json) is a portable design
for process seeds, assumption conditions, horizon and agent seed. The separate
[model YAML](../examples/cascadia/cascadia_homing.yaml) selects geometry, exact
probabilities, preferences and physical step seconds. Its data paths resolve
relative to the YAML file. A design file does not select a different dataset.

```bash
uv run python GEO-INFER-ACT/examples/cascadia/cascadia_study.py \
  --design GEO-INFER-ACT/examples/cascadia/study_design.json \
  --gnn-repo ../Generalized_Notation_Notation \
  --gnn-python ../Generalized_Notation_Notation/.venv/bin/python \
  --output GEO-INFER-ACT/examples/outputs/cascadia_study_new
```

`--design` excludes `--seeds`, `--conditions`, `--steps` and `--agent-seed`.
Unknown fields, duplicate JSON keys, duplicate seeds, invalid conditions and
an excessive grid fail before output creation. Existing CLI-only calls retain
their defaults. A run directory is create-only. Failed attempts and failed
qualification remain available for diagnosis.

Changing a condition changes the model used by both agent and process. In
particular, removing odor information changes cue frequencies too. These
conditions do not establish an intervention on real fish or isolate information
value under a fixed marginal cue distribution.

## Qualify saved records and explore them

```bash
uv run python GEO-INFER-ACT/examples/cascadia/validate_study.py \
  GEO-INFER-ACT/examples/outputs/cascadia_study_new
uv run python GEO-INFER-ACT/examples/cascadia/study_explorer.py \
  GEO-INFER-ACT/examples/outputs/cascadia_study_new \
  GEO-INFER-ACT/examples/outputs/cascadia_study_new_explorer.html
```

Static qualification checks every recorded hash, the ordered grid and aggregate
summary, plus each run's saved summary, outcome, process/agent seeds, horizon,
observations, state order and artifact identity. It does not recompute inference.
The optional `--geo-infer-root .` additionally checks live owner bytes; a later
owner edit can make that fail while the historical archive remains intact.

The explorer is self-contained and needs no network or JavaScript dependency.
The [retained explorer edition](../examples/cascadia/data/cascadia_study_explorer_20261013_v2.html)
displays the archived seed study with the readable fallback background.
Select a condition and seed, scrub a recorded step, or play successive records.
It shows real H3 cell centers, typed river relations, raw posterior probabilities,
the recorded true state, selected action and cue, and empirical first-home
fractions including censored seeds. A table provides the same arrival/censoring
information accessibly. The manifest digest identifies the displayed edition.
No interpolated belief is represented as an inferred state. All probability
vectors and display carrier indices are checked before rendering.

## Cross-repository teaching companion

GNN's `docs/other/geo_infer/qualify_export.py` reproduces each emitted artifact
with the actual producer. fep_formal's `specs/cascadia-experiments/proof_impact.py`
compares exact tables and lists reopened mathematical obligations. LectureCreate's
`examples/cascadia_geo_act/study_companion.py` invokes those tools in their own
environments and creates the explorer plus a companion receipt.

The existing reference Lean receipt and 15-minute lecture remain separately
bound editions. A worksheet or interactive visual grants no new native proof
status. To teach a variant as verified mathematics, create and compile a named
variant instance with its own witness roster, axiom inspection and mutations.

## Validation

```bash
uv run pytest GEO-INFER-ACT/tests/integration/test_cascadia_study.py \
  GEO-INFER-ACT/tests/integration/test_cascadia_study_custody.py \
  GEO-INFER-ACT/tests/integration/test_cascadia_study_design.py \
  GEO-INFER-ACT/tests/integration/test_cascadia_explorer.py -q --no-cov
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```

The design tests exercise parser conflicts and bounded inputs. Saved-record
tests reject contradictions even after their hashes are updated. Display tests
read the actual retained archive and reject malformed probabilities, carrier
indices, order, and horizon; the entry point must preserve an existing HTML
file and refuse a drifted study before writing.
