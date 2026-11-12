# GNN conformance

`gnn conformance (pinned)` is the 3.12 leg of the `interchange` job in
`.github/workflows/gnn-interchange.yml`, which `ci.yml` also calls as
`paired-interchange`; the 3.11 leg reports as `gnn interchange (3.11)`. The
job reads `.github/gnn-pair.json` (schema `gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`).

After installing both locked runtimes the job runs `geo/.venv/bin/python
geo/scripts/gate.py gnn --gnn-repo $PWD/gnn --gnn-python
$PWD/gnn/.venv/bin/python --output receipts/interchange.json
--expect-geo-revision $GITHUB_SHA`. The arm starts with the shared pin reader
of the GNN–FEP workflow group, executed by the pinned checkout's interpreter:

```bash
python -m gnn.importers check --repo . --require-pointer \
    --workflow .github/workflows/gnn-interchange.yml --job interchange
```

That command reads the pin above, checks this file against it, and parses every
`*.gnn.md` in this repository with the parser installed at the pinned revision.
GEO_INFER keeps no reader of its own: `scripts/gate.py` repeats no pin shape and
no pointer assertion. A reader refusal fails the job before any interchange run
starts.

The arm then runs
`GEO-INFER-TEST/validate_gnn_interchange.py --gnn-repo $PWD/gnn --gnn-python
$PWD/gnn/.venv/bin/python --output receipts/interchange.json`. The script
exports the pinned checkout's
`input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md` through
`gnn.export.geo_infer` at 60-second steps, asserts the artifact's
`source_sha256` against the source bytes, then replays it twice with
`geo_infer_act.core.gnn_contract.run_gnn_inference` and compares each
posterior and predicted prior against direct Bayesian algebra. It then exports
two fixtures from the pinned checkout's `tests/export/`
(`gaussian_rectangular.md`, `factored_example.json`) plus a generated H3
stay/diffuse model, and replays each through its geo_infer_act consumer. The
workflow asserts the receipt's success flag, both exact revisions, both clean
worktrees, complete execution custody, and every retained artifact digest.

The interchange validator itself reads commands and artifacts, not the
specifications: the committed
`GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md` is covered by
the reader's parser pass above, not by the export/replay checks. The job is a
pin, reader and interchange baseline for this repository's GNN surface.
