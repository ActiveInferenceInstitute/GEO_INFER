# Research-grade inference contracts

This page records the executable contracts shared by the current ACT, BAYES,
and RISK implementations. It is intentionally implementation-facing: module
source, tests, and strict validators remain authoritative if this page drifts.

## Active Inference

- `ActiveInferenceModel.act()` accepts scalar or sequence-valued control
  configuration and validates action availability before selecting a policy.
- Categorical H3 inference uses the real `inferactively-pymdp==1.0.3` adapter
  when its matrix contract is present. Local fallback is opt-in through
  `allow_local_pymdp_fallback`; an optional-backend failure is not silently
  reported as a backend result.
- Read-only H3 grid inference snapshots and restores the policy selector RNG,
  so diagnostics do not change a later stochastic decision.
- H3 observations, transition matrices, and belief vectors are shape-checked
  before Bayesian updates. Nested H3 results retain typed cell, level, and
  parent-child diagnostics.
- Optional RxInfer/Bayeux integrations either return posterior data produced by
  the external backend or use the explicitly labelled deterministic/local
  sampler. They do not emit fabricated backend-success payloads.

## Bayesian inference

- ABC, SMC, MCMC, HMC, variational inference, and spatio-temporal GP sampling
  use instance-local NumPy generators. A caller's global RNG state is not part
  of the sampler contract.
- MCMC and HMC flatten scalar and array-valued parameters internally and restore
  the declared parameter shapes in returned draws. HMC's NUTS path performs
  recursive tree construction with a no-U-turn check; it is not an alias for a
  fixed-step sampler.
- Variational inference accepts vector mean-field parameters and preserves
  `initial_var_params` during updates. Full-rank covariance currently has an
  explicit scalar-parameter boundary.
- Diagnostics report effective sample size, Monte Carlo standard error, R-hat,
  and Geweke statistics. R-hat is `NaN` for one chain because between-chain
  convergence is undefined; it is not replaced with a misleading scalar.
- `PosteriorAnalysis.credible_interval()` requires `0 < alpha < 1` and rejects
  missing, empty, or non-finite draws.
- `SpatioTemporalGP.predict()` accepts only an `(n, 3)` matrix of `x`, `y`, and
  time; split spatial/temporal arguments are rejected. Posterior
  predictions fit a concrete spatial GP for each draw without mutating the
  fitted model.

## Risk modeling

- `RiskParameters` validates confidence, iteration, uncertainty, and seed
  settings. Component outputs must be aligned, finite, non-negative, and within
  their declared probability bounds.
- Point-estimate uncertainty bounds come from declared component bounds. When
  no component uncertainty is declared, lower and upper bounds equal the point
  estimate and the result identifies that source explicitly; arbitrary fixed
  percentage bands are not used.
- Monte Carlo runs use a local seeded generator and expose configured percentile
  keys alongside the established result names.
- Hazard event timestamps are reproducible for seeded runs, and return-period
  intensities require a fitted, calibrated model. Gumbel return levels derive
  location and scale from fitted mean and standard deviation.
- `EnhancedRiskEngine` validates configuration before creating output resources,
  checks available integration APIs, and supports `close()` plus context-manager
  lifecycle for its executor and file handler.
- RiskEngine calibration rejects unsupported maximum-likelihood and underspecified
  Bayesian requests explicitly; only validated cross-validation inputs produce
  calibration results.

## Verification

From the repository root, run the focused gates first:

```bash
uv run pytest GEO-INFER-ACT/tests/unit GEO-INFER-ACT/tests/integration -q
uv run pytest GEO-INFER-BAYES/tests/unit GEO-INFER-BAYES/tests/integration -q
uv run pytest GEO-INFER-RISK/tests/unit -q
uv run --with 'ruff>=0.15.6,<0.16' ruff check .
python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
```

Then run the repository contract matrix from `AGENTS.md`, including strict
source, documentation, test, model, reproducibility, skills, unit,
integration, performance, and H3-migration checks. Generated runtime files
belong in `.geo-infer-test-results/` or a temporary directory and must not be
published as source artifacts.
