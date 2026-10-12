## GEO-INFER-BAYES — Bayesian Inference and Uncertainty Quantification

**Purpose.** GEO-INFER-BAYES is the framework's probabilistic core: a comprehensive Bayesian inference framework with probabilistic modeling, uncertainty quantification, and computational methods for geospatial data. Its README frames the module as the source of principled uncertainty estimates for downstream geospatial decisions, and its examples and diagnostic figures illustrate sampling and posterior workflows; retained command receipts establish which workflows actually executed.

**Public API.** The `geo_infer_bayes` package exports Gaussian-process machinery (`GaussianProcess`, `SpatialCovariance`, and the spatial variants `SpatialGP`/`SparseSpatialGP`), inference engines (`BayesianInference`, `VariationalInference`, and `MCMC` re-exported as `MCMCSampler`), and posterior tooling via `PosteriorAnalysis`. The civic-intelligence integration exports `HazardCategoricalPrior`, `build_hazard_categorical_prior`, `build_hazard_prior_table`, and `load_crescent_city_intel`, converting structured hazard observations into categorical priors.

The tests exercise Gaussian processes, MCMC, variational inference, and prior construction. Analytical fixtures and sampler diagnostics are distinct evidence surfaces; a small passing fixture does not establish field calibration.

**Theme role.** Inference and learning: BAYES is the uncertainty engine of the stack, feeding posterior beliefs and priors to the active inference core (ACT) and the domain sciences.
