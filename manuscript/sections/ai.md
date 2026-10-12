## GEO-INFER-AI — Machine Learning for Geospatial Analysis

**Purpose.** GEO-INFER-AI supplies advanced machine learning and AI capabilities designed specifically for geospatial analysis and decision-making. Its README frames the module as the ML workhorse of the framework, complementing the probabilistic modules with discriminative models, feature engineering, evaluation, and experiment tracking.

**Public API.** The `geo_infer_ai` package exports a training and evaluation stack: `ModelTrainer` with `TrainingConfig`, `ModelExplainer` for explainability, and `GeospatialModelEvaluator`. Concrete models include `ImageClassifier` and the predictive family `SpatialPredictor` and `SpatialPrediction`. Geospatial specialization appears in `GeospatialFeatureEngineer` (preprocessing), the interpolation pair `IDWInterpolator` and `OrdinaryKriging`, and multi-scale structure via `H3SpatialGraph`, `LevelSpatialGraph`, and `MultiScaleHierarchicalAnalyzer` with `analyze_multi_scale_patterns`. `MLflowPipeline` provides experiment tracking, and `EnvironmentalActiveInferenceEngine`/`EnvironmentalState` bridge to the framework's active inference theme.

The tests exercise training, prediction, preprocessing, value-based evaluation, and multi-scale analysis. The shared repository runner selects this module with `--module AI`.

**Theme role.** Inference and learning: AI is the data-driven learning counterpart to the probabilistic ACT and BAYES modules, feeding learned predictions into the domain sciences and agent bands.
