# GEO-INFER-EDU/src/geo_infer_edu/core

Core workspace within `GEO-INFER-EDU`.

## Contents

- `__init__.py`
- `curriculum.py`
- `exercises.py`
- `personalization.py`
- `professional.py`
- `progress.py`

## Public Interface

- `curriculum.py:EducationLevel` (class)
- `curriculum.py:PedagogicalApproach` (class)
- `curriculum.py:LearningObjective` (class)
- `curriculum.py:CurriculumModule` (class)
- `curriculum.py:Curriculum` (class)
- `curriculum.py:CurriculumDesigner` (class)
- `exercises.py:ExerciseType` (class)
- `exercises.py:DifficultyLevel` (class)
- `exercises.py:Exercise` (class)
- `exercises.py:Assessment` (class)
- `exercises.py:ExerciseGenerator` (class)
- `personalization.py:LearnerProfile` (class)
- `personalization.py:LearningResource` (class)
- `personalization.py:LearningPathway` (class)
- `personalization.py:compute_skill_gap_pathway` (function)
- `personalization.py:PersonalizedLearning` (class)
- `professional.py:ProfessionalProfile` (class)
- `professional.py:ContinuingEducationActivity` (class)
- `professional.py:CertificationPathway` (class)
- `professional.py:ProfessionalDevelopment` (class)

## Module Metadata

- Module: `GEO-INFER-EDU`
- Package: `geo_infer_edu`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-edu`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EDU`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EDU
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
