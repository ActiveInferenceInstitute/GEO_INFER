# GNN conformance

`gnn conformance (pinned)` is the 3.12 leg of the `interchange` job in
`.github/workflows/gnn-interchange.yml`, which `ci.yml` also calls as
`paired-interchange`; the 3.11 leg reports as `gnn interchange (3.11)`. The
job reads `.github/gnn-pair.json` (schema `gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`).

After installing both locked runtimes the job runs
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

It does not scan this repository for `*.gnn.md`, so the committed
`GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md` is outside
what this job reads. The job is a pin and interchange baseline, not a claim
that every model here is covered.
