# Synthetic measurement-quality robustness probe

Public-source local research based on GEO_INFER commit
`8bbe89ce409ffedde198be20cc48ae3d5272794e`. GEO-INFER © 2025 by Daniel Ari
Friedman (@docxology), CC-BY-NC-SA-4.0; the root LICENSE is unchanged.

The probe checks whether a separate observation-generating likelihood can be
used while retaining the planner's assumptions. It establishes an implementation
boundary, not when a monitoring strategy generally helps or fails.

## API and boundary

`monitoring.simulated_report(..., generator_artifact=None)` samples using the
optional generating artifact and returns the planner likelihood for the sampled
outcome. `run_scenario(..., generator_artifacts=None)` uses the override only
after allocation. The actual ACT evaluator/resource allocator and PLACE field
solver are used. The omitted override preserves the checked default behavior.

The compatibility guard allows report-likelihood and provenance differences.
Model identity, axes, outcomes, transitions, policies, preferences, resource
semantics, time and passive readings remain fixed. Generated outcomes must stay
within the planner's support. Standalone Scenario does not carry generator
identity: retain both identities, or use this harness's bundle. The ordinary
historical experiment/receipt workflow is unchanged.

## Synthetic design and interpretation

The fixture adapts the pinned `test_california_monitoring.py` numerical
landscape: three H3 cells, three synthetic indicators, nominal 2016–2018 slots,
two rounds, seeds 0 and 1. Fifty-eight repeated synthetic county entries satisfy
the schema and represent no geographic coverage. Source record counts are zero;
the all-zero source-receipt digest is a deliberate sentinel. No observed data
was acquired.

Existing thresholds derive binary labels from synthetic values. Priors use
2016–2017 labels; target 2018 labels remain hidden from allocation. Recorded
output distinguishes these derived labels, simulated readings and inferred
probabilities.

| Condition | Generator coarse accuracy / coverage | Generator detailed accuracy / coverage |
|---|---|---|
| matched | 0.8 / 0.8 | 0.97 / 0.95 |
| overoptimistic planner | 0.55 / 0.5 | 0.65 / 0.7 |
| pessimistic planner | 0.9 / 0.9 | 0.995 / 0.99 |

The planner retains the matched assumptions in every condition. The two
mismatches change accuracy and coverage jointly; they isolate neither component
and establish no benefit threshold. Mismatched artifacts are locally constructed
and validated, explicitly not independently exported by GNN.

The existing active, quota_active and rotating strategies use the same six-unit
per-round budget ceiling. Quota floors are one unit per indicator. Hard action
costs `(0, 1, 3)` differ from log preference utilities `(0, -0.03, -0.09)`;
neither is measured monetary, energy or field cost. There is no cost mismatch.
Counterseed keys pair the same seed/indicator/site/round/mode across conditions.

The create-only output has 18 tiny scenarios. `results.json` records both
configurations, model/source identities, loaded PLACE/ACT source hashes and
package versions, actions, readings, posteriors, scores and modeled expenditure.
`timings.json` separately records wall time around each scenario including
evaluator construction; host/order effects make this unsuitable as a benchmark.
`files.sha256.json` inventories the outputs. The execution binding is deliberately
narrower than a full California workflow receipt.

## Reproduce locally

Use the repository's required uv 0.13.6 and Python selection, then synchronize
the PLACE dependencies and its California, development and integration extras.
The integration extra supplies the shared repository pytest plugin. This is a
locked module-scoped environment; it does not install every workspace extra.
Run from the repository root:

```sh
uv sync --locked --package geo-infer-place --extra california --extra dev --extra integrations --no-default-groups
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uv run --frozen --no-sync python -m pytest -p pytest_asyncio.plugin GEO-INFER-PLACE/tests/unit/test_california_robustness.py GEO-INFER-PLACE/tests/unit/test_california_inference.py GEO-INFER-PLACE/tests/unit/test_california_monitoring.py -o addopts= -W error --strict-config --strict-markers --import-mode=importlib -q
uv run --frozen --no-sync python -m geo_infer_place.locations.california.robustness --fixture research/measurement-quality/fixture.json --models GEO-INFER-PLACE/locations/california/models --output research/measurement-quality/smoke-reproduction
```

Use a new output path for each CLI run. On 10 November 2026, the three test files
passed all 67 cases on Python 3.12.12 in this locked environment, without
source-path overrides, deselection or inference substitutes. This includes the
historical experiment/receipt, reporting and integrity checks excluded from the
initial minimal-environment run. The harness is exercised by its regression
test; the separate CLI command above was not rerun during submission preparation.

The initial 7 November review used Python 3.11.13 with inherited system packages
and five local wheels. It passed 58 selected tests and excluded nine broader
workflow/reporting cases. Its original fixture outputs and independent
posterior-reconstruction evidence remain historical evidence, distinct from
the new test execution. `baseline-scenarios.json` is the retained original-source
reference; the tests compare both the omitted and explicit matched override
with it. No full workspace, release, Docker or performance suite was run.

No sensor calibration, ecological observation, field experiment, financial
benefit or upstream acceptance follows from this fixture. A subsequent
accuracy-only comparison, with coverage and costs fixed, remains a proposal to
design before running it.
