---
name: geo-infer-edu
description: Educational technology for geospatial learning. Use when creating spatial analysis curricula, interactive GIS exercises, learning progression models, competency assessment for geographic concepts, or step-by-step spatial tutorials.
prerequisites:
  required: []
  recommended:
    - geo-infer-space
    - geo-infer-data
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-EDU

## Instructions

### Core Capabilities

- **Curriculum design**: Standards-aligned curriculum and module generation (`CurriculumDesigner`)
- **Exercises**: Template-based interactive exercises with starter code and test cases (`ExerciseGenerator`)
- **Progress tracking**: Competency tracking, gap identification, analytics, and privacy-aware exports (`ProgressTracker`)
- **Personalized learning**: Adaptive pathways and resource recommendations (`PersonalizedLearning`)
- **Professional development**: Certification pathways and continuing-education tracking (`ProfessionalDevelopment`)

### Key Imports

```python
from geo_infer_edu import (
    CurriculumDesigner,
    ExerciseGenerator,
    ProgressTracker,
    PersonalizedLearning,
    ProfessionalDevelopment,
)
from geo_infer_edu.core.progress import CompetencyLevel
```

## Examples

```python
from geo_infer_edu import ExerciseGenerator, ProgressTracker

# Generate exercises for a set of concepts
generator = ExerciseGenerator()
exercises = generator.create(
    concepts=["spatial_autocorrelation", "buffer_analysis"],
    format="code",           # 'interactive_map', 'code', 'quiz', 'practical'
    difficulty="progressive",
    include_hints=True,
)

# Track learner progress from activity logs and assessment scores
tracker = ProgressTracker(privacy_compliance="ferpa")
progress = tracker.track_progress(
    learner_id="student_042",
    activity_log=[
        {"id": "act_1", "type": "exercise", "topic": "spatial_autocorrelation",
         "score": 0.82, "duration_minutes": 45},
    ],
    assessments=[{"competency": "spatial_analysis", "score": 0.82}],
)
export = tracker.export_progress("student_042")  # JSON string, FERPA-pseudonymized
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog EDU` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Competency levels are ordered (`NOT_STARTED` < `EMERGING` < `DEVELOPING` < `PROFICIENT` < `EXEMPLARY`); assessments never silently downgrade an achieved level.
- `CurriculumDesigner` accepts only supported standards (`bok`, `gistbok`, `ngss`); unknown standards raise `ValueError`.
- `ProgressTracker` exports embed `record.level.value` strings, so exports are always JSON-serializable.

### Integrations

- Optional integration with SPACE and DATA for spatial exercise content (not imported at runtime)
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EDU`