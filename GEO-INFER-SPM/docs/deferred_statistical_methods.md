# Deferred Statistical Methods Register

This register documents every statistical method in GEO-INFER-SPM that ships
today as a documented approximation rather than a full estimator. Each entry
lists the module site (source of the inline pointer comment), what currently
ships, and what the full estimator requires. Implemented upgrades are listed
at the end for provenance.

Inline comments in `src/geo_infer_spm/` point here with the pattern
`Deferred: see docs/deferred_statistical_methods.md ("<entry title>")` instead
of carrying unexplained deferral text.

---

## Registered (approximation ships; full estimator deferred)

### Bayes factors via marginal likelihoods
- **Site:** `core/bayesian.py` `BayesianSPM._compute_bayes_factors`
- **Ships today:** Bayes factors approximated as `exp((BIC_min - BIC_i)/2)`;
  BIC computed from stored diagnostics or from residuals when absent.
- **Full estimator requires:** per-model marginal likelihood `p(y|M)` via
  nested sampling, bridge sampling, or a Laplace approximation over the
  posterior samples.

### DIC effective-parameters
- **Sites:** `core/bayesian.py` `BayesianSPM._compute_dic`;
  `core/advanced/model_validation.py` `ModelValidator._compute_dic`
- **Ships today:** `DIC = D_bar + 2*k` with `p_D` fixed at the regressor
  count and deviance taken from stored log-likelihood.
- **Full estimator requires:** posterior-sample-based `p_D = var(deviance)`.
  Mean deviance from MCMC draws (requires `posterior_samples`).

### Empirical-Bayes MAP posterior covariance
- **Site:** `core/bayesian.py` `BayesianSPM._fit_empirical_bayes_glm`
- **Ships today:** MAP estimation by BFGS on a Gaussian-prior negative log
  posterior; covariance as `pinv(X'X + I)` (unit-prior Laplace approximation).
- **Full estimator requires:** covariance from the Hessian of the actual
  optimized log posterior (including data-driven prior hyperparameters) and
  hyperparameter uncertainty (integrated Laplace / EB with error estimation).

### Spatial hierarchical model (CAR / Gaussian process)
- **Site:** `core/bayesian.py` `BayesianSPM.spatial_hierarchical_model`
- **Ships today:** Gaussian-basis augmentation of the design matrix fit by
  the empirical-Bayes GLM; basis centers selected from sample coordinates.
- **Full estimator requires:** conditional autoregressive (CAR) prior or a
  Gaussian-process covariance fit jointly with the fixed effects (full MCMC
  or Laplace-GP inference), including hyperparameter posteriors.

### MCMC effective sample size
- **Site:** `core/bayesian.py` `BayesianSPM._compute_ess`
- **Ships today:** ESS reported as `n_draws * n_chains` (upper bound, no
  autocorrelation adjustment).
- **Failure mode:** if posterior chain/draw dimensions are unavailable,
  `_compute_ess` logs a warning and reports `NaN` (missing-value marker),
  mirroring `_compute_r_hat` — never a fabricated baseline constant.
- **Full estimator requires:** ESS from the autocorrelation function of the
  chains with Geyer initial-monotone-sequence truncation, using split chains.

### Mean-field variational inference
- **Site:** `core/bayesian.py` `BayesianSPM.variational_inference`
- **Ships today:** conjugate mean-field CAVI fixed-point updates for a
  Gaussian GLM with Gamma noise prior; returns MAP-style means and a
  Gaussian covariance approximation.
- **Full estimator requires:** ELBO tracking with convergence criteria,
  full-rank or normalizing-flow posteriors, and variance-parameter
  uncertainty propagated into credible intervals.

### Huber-White robust covariance
- **Site:** `core/glm.py` `GeneralLinearModel._fit_robust`
- **Ships today:** IRLS with Huber-type weights; sandwich covariance
  `B^-1 M B^-1` with bread `(X'WX)^-1` and meat `X'WX` (weighted information).
- **Full estimator requires:** meat built from per-observation score
  contributions (`X' diag(w_i^2 e_i^2) X`), cluster-robust variant, and
  finite-sample df corrections.

### Theil-Sen significance test
- **Site:** `core/temporal_analysis.py` `TemporalAnalyzer._theil_sen_trend`
- **Ships today:** significance from a normal z-approximation using the
  standard deviation of all pairwise slopes.
- **Full estimator requires:** McKean-Schrader confidence interval for the
  median slope, or an exact/permutation test on Kendall's tau.

### Full REML / ML variance components
- **Sites:** `core/advanced/mixed_effects.py` `MixedEffectsSPM._fit_reml`,
  `MixedEffectsSPM._fit_ml`
- **Ships today:** joint optimization of beta with two scalar variance
  components over `V = tau2 * Z Z' + sigma2 * I`; the REML restriction term
  `-0.5 * log|X' V^-1 X|` is omitted.
- **Full estimator requires:** profiled GLS beta at each variance step, the
  REML restriction term, per-group variance components matching
  `random_groups`, and derivative-based optimization with convergence checks.

### Smoothing spline / GAM
- **Sites:** `core/advanced/nonparametric.py` `NonparametricSPM._fit_spline`,
  `NonparametricSPM._fit_gam`
- **Ships today:** spline method implemented as a moving-average smoother;
  GAM as per-predictor local smoothing summed without backfitting.
- **Full estimator requires:** penalized regression spline (B-spline basis
  with a second-difference roughness penalty) with GCV/CV smoothing-parameter
  selection and proper influence weights; additive backfitting with
  identifiability constraints for the GAM.

### Out-of-sample prediction (nonparametric)
- **Site:** `core/advanced/nonparametric.py` `NonparametricSPM.predict`
- **Ships today:** returns the fitted in-sample `y_hat` regardless of the
  coordinates in `new_data`.
- **Full estimator requires:** evaluation of the fitted smooth at the new
  coordinates (kernel/basis evaluation or kriging-style interpolation).

### Exact-ML spatial error model (SEM)
- **Site:** `core/advanced/spatial_regression.py` `SpatialRegression._fit_sem`
- **Ships today:** lambda profiled by optimizing a filtered-residual
  log-likelihood; beta from unfiltered OLS; the `log|I - lambda W|` Jacobian
  is not included.
- **Full estimator requires:** concentrated exact-ML likelihood with the
  `log|I - lambda W|` Jacobian and beta from GLS on the spatially filtered
  response `B y`.

### Formula parser
- **Site:** `utils/helpers.py` `_parse_formula`
- **Ships today:** `~` / `+` / `*` grammar over covariates, intercept
  suppression via `0`. Unknown interaction operands, empty terms and empty
  designs are rejected; factor values must belong to nonempty unique levels.
- **Full estimator requires:** patsy-style grammar: `-` term removal, `:`
  pure interactions, `I()` transforms, polynomial/spline term functions, and
  factor-level encoding inside formulas.

### Power analysis with spatial autocorrelation
- **Site:** `utils/helpers.py` `compute_power_analysis`
- **Ships today:** Monte-Carlo one-sample t-test power under i.i.d.
  `N(effect, 1)` draws.
- **Full estimator requires:** effective sample size correction for spatial
  autocorrelation, two-sample and F-test variants, and analytic
  noncentral-t power as a cross-check.

### Gridded spatial filtering
- **Site:** `utils/preprocessing.py` `spatial_filter` (multi-column branch)
- **Ships today:** 1-D point data filtered with Gaussian/median/mean neighbor
  weighting; multi-column gridded data passes through unfiltered.
- **Full estimator requires:** reconstruction of the regular grid from
  coordinates (rasterization with cell-size inference), `scipy.ndimage`
  filtering, and remapping of filtered values to the original points.

### CV prediction layer
- **Site:** `core/advanced/model_validation.py`
  `ModelValidator._predict_on_subset`
- **Ships today:** linear prediction `X_test @ beta` with a training-mean
  fallback.
- **Full estimator requires:** model-type-aware prediction (e.g. spatial
  models using their covariance/weights structure, nonparametric smoothers
  evaluated at test coordinates).

### Bootstrap optimism
- **Site:** `core/advanced/model_validation.py`
  `ModelValidator._compute_bootstrap_score`
- **Ships today:** optimism scored as the residual variance of the bootstrap
  refit.
- **Full estimator requires:** Efron's optimism correction (apparent error
  minus out-of-bag test error) or .632 / .632+ estimates.

### Studentized Breusch-Pagan test
- **Site:** `core/advanced/model_validation.py`
  `ModelValidator._breusch_pagan_test`
- **Ships today:** auxiliary-regression LM/F form on squared residuals with
  two regressors (intercept + fitted values).
- **Full estimator requires:** the studentized (Koenker) variant with
  studentized residuals and chi-square reference distribution.

### Spatial regularization default weights
- **Site:** `core/glm.py` `GeneralLinearModel._fit_spatial`
- **Ships today:** when no spatial weights matrix is supplied, falls back to
  ridge-style regularization `lambda * I`.
- **Full estimator requires:** distance-decay or kernel-based spatial weights
  with data-driven bandwidth/lambda selection.

### Cook's distance (diagnostic plots)
- **Sites:** `visualization/maps.py` diagnostic map panels,
  `visualization/interactive.py` diagnostics figure
- **Ships today:** Cook's distance computed from a dense hat-matrix
  approximation without leverage grouping.
- **Full estimator requires:** cached leverage terms, threshold visualization
  against `4/n` and `4/(n-p)` reference lines, and grouped (spatial-block)
  influence variants.

### Mixed-effects prediction (random effects + new data)
- **Site:** `core/advanced/mixed_effects.py` `MixedEffectsSPM.predict`
- **Ships today:** fixed-effects prediction reusing the training design
  matrix; random effects are not included and new data is not accepted.
- **Full estimator requires:** BLUP random-effect prediction for new data,
  construction of the design matrix for unseen rows, and prediction
  uncertainty.

### Temporal B-spline basis functions
- **Site:** `core/temporal_analysis.py` `TemporalAnalyzer.temporal_basis_functions`
- **Ships today:** Gaussian bumps centered on uniform knots as a B-spline
  stand-in for the ``bspline`` basis type.
- **Full estimator requires:** a true B-spline basis (e.g. `scipy.interpolate.BSpline`
  design matrix) with knot placement control.

### Choropleth rendering (polygon data)
- **Site:** `visualization/interactive.py` (`map_type='choropleth'`)
- **Ships today:** falls back to a scattergeo map with a warning.
- **Full estimator requires:** polygon/geometry ingestion (GeoJSON or
  geopandas boundaries) mapped to region identifiers.

---

## Implemented during this ledger closeout (was deferred, now real)

### Split Gelman-Rubin R-hat
- **Site:** `core/bayesian.py` `BayesianSPM._compute_r_hat`
- Implemented as the standard split-R-hat (Gelman et al., BDA3): half-chain
  splitting, between/within variance, `sqrt(((n-1)/n*W + B/n) / W)`, one value
  per posterior parameter (max across components for vector parameters).
  Module-level entry point: `geo_infer_spm.core.bayesian.gelman_rubin_r_hat`.
- Tests: `tests/unit/test_bayesian.py` (`TestSplitRHat`) — converged chains
  score near 1.0, chains with separated means score far above 1, degenerate
  constant chains handled.

### Spatial block cross-validation (convenience path)
- **Sites:** `core/advanced/model_validation.py` `validate_spm_model`
  (`method='cross_validate'`), `ModelValidator._spatial_cv`
- Implemented: the convenience function now refits the module GLM on each
  training block via the existing fold/block primitives (K-means coordinate
  blocks + `_subset_data` / `_subset_design_matrix` / `_predict_on_subset`)
  and scores the held-out block.
- Tests: `tests/unit/test_advanced_models.py` (`TestModelValidator`) —
  `test_convenience_cross_validation_spatial_blocks`.
