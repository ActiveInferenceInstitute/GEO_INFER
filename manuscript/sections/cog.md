## GEO-INFER-COG — Spatial Cognition and Human-Centered Tools

**Purpose.** GEO-INFER-COG provides human-centered geospatial tools that model perception, reasoning, and spatial cognition for intuitive interfaces. Its README defines the module as the cognitive-science vertical: computationally modeling how humans perceive and reason about space so that framework interfaces match human spatial thinking.

**Public API.** The `geo_infer_cog` package exports a full cognitive stack. Core: `CognitiveProcessingEngine`, `SpatialPerceptionModel`, `SpatialReasoningEngine`, and `SpatialMemoryModel`. Supporting: `SpatialLanguageProcessor` (its own `spatial_language/` subpackage), `HumanCenteredVisualizer` (visualization/), and `SpatialDecisionSupport` (decision/). Models: `CognitiveMap`, `SpatialKnowledgeGraph`, `UserCognitiveProfile`, and `ProfileManager`. Utilities include `validate_spatial_data`, `validate_cognitive_model`, `load_cognitive_profile`, and `save_cognitive_model` — across the package's subdirectories (`core/`, `models/`, `spatial_language/`, `visualization/`, `decision/`, `utils/`, `api/`).

The tests exercise perception, reasoning, memory, and language behavior. The shared repository runner selects this module with `--module COG`.

**Theme role.** Domain sciences (cognitive science) with an interface vocation: COG connects the inference band's belief machinery to human factors, informing the APP layer's design and the agent band's human-like spatial reasoning.
