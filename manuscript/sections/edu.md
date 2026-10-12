## GEO-INFER-EDU — Educational Technology for Geospatial Systems

GEO-INFER-EDU delivers educational technology for geospatial systems: curriculum design, interactive exercises, and learning analytics (module README). Within the theme table it belongs to Data, API & Applications, serving as the framework's pedagogical surface — the layer through which new users learn the other modules rather than a computational domain module.

The `geo_infer_edu` package exports public classes from `src/geo_infer_edu/__init__.py`: `CurriculumDesigner` for structuring learning paths, `ExerciseGenerator` for producing interactive exercises, `ProgressTracker` for learning analytics, `PersonalizedLearning` for adaptive learner experiences, and `ProfessionalDevelopment` for practitioner training tracks. Dependency declarations live in the package's `pyproject.toml`; the repository root supplies the shared environment and lock.

The tests exercise curriculum, exercise generation, personalization, progress tracking, and professional development. Examples and documentation expose these same interfaces for teaching; classroom effectiveness requires separate evidence.
