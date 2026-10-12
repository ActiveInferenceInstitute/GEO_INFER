---
name: geo-infer-examples
description: Working examples and module orchestration patterns. Use when looking for usage examples, cross-module orchestration patterns, end-to-end workflows, or reference implementations of GEO-INFER capabilities.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-EXAMPLES

## Instructions

### Core Capabilities

- **Module orchestrators**: Cross-module workflow examples
- **Domain examples**: Per-domain analysis workflows
- **Active Inference**: End-to-end inference pipeline examples
- **Integration patterns**: How modules connect and compose

### Key Directories

```text
GEO-INFER-EXAMPLES/examples/
├── module_orchestrators/     # Per-module orchestration demos
├── getting_started/          # Beginner walkthroughs
├── agriculture_integration/  # AG + SPACE + RISK workflows
├── climate_integration/      # CLIMATE + SPACE workflows
├── health_integration/       # HEALTH + SPACE workflows
└── iot_radiation_monitoring/ # IOT + DATA + SPACE workflow
```

### Orchestrator Example

```python
from geo_infer_examples.core.module_orchestrator import ModuleOrchestrator

orchestrator = ModuleOrchestrator(monitoring_enabled=False)
workflow_ids = orchestrator.list_workflows()
module_health = orchestrator.get_module_health()
```

## Examples

```python
from geo_infer_examples.core.module_orchestrator import ModuleOrchestrator

# Inspect configured cross-module workflows before execution
orchestrator = ModuleOrchestrator(monitoring_enabled=False)
for workflow_id in orchestrator.list_workflows():
    workflow = orchestrator.get_workflow_definition(workflow_id)
    print(workflow_id, workflow.execution_strategy)
```

```python
# Domain workflow: Agricultural risk assessment
from geo_infer_ag.models.soil_health import SoilHealthModel
from geo_infer_risk.core.risk_engine import EnhancedRiskEngine
from geo_infer_space.backends.h3 import H3Backend

farm_polygon = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]],
}

# 1. Tessellate the farm boundary into H3 cells (real SPACE API)
cells = H3Backend().polygon_to_cells(farm_polygon, resolution=9)

# 2. Assess soil health per cell. SoilHealthModel.predict expects a dict with
#    pandas "field_data" and "soil_data" frames (organic_matter, ph,
#    bulk_density columns) and returns per-indicator scores.
soil = SoilHealthModel()
# predictions = soil.predict({"field_data": field_df, "soil_data": soil_df})

# 3. Compute risk with the real RISK engine
risk = EnhancedRiskEngine()
# risk_results = risk.run_enhanced_analysis(...)
```

```python
# Per-module orchestrator scripts
# (examples/module_orchestrators/<MODULE>/scripts/run_orchestrator.py)
# hand one real operation to the packaged engine; no sys.path setup needed.
import sys

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict:
    from geo_infer_math import haversine_distance  # module's public API

    return {"km": haversine_distance(0.0, 0.0, 0.0, 1.0)}


sys.exit(run_module_orchestrator("MATH", _operation))
# Exit codes: 0 ok, 1 operation raised, 2 missing dependency (JSON names the
# verified geo-infer-<module>[extra] from MODULE_EXTRAS).
```

```python
# Import surface of this package
from geo_infer_examples import ModuleOrchestrator, ExecutionStrategy, ModuleStatus
from geo_infer_examples.core import ModuleOrchestrator as CoreOrchestrator
from geo_infer_examples.models import WorkflowDefinition, GEO_INFER_MODULES
```

## Guidelines

- `module_orchestrator.py` convergence check uses numeric relative-change
- Track new tutorial requests in the root task ledger before adding examples
- Verified runnable examples live under `examples/*/scripts/`
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

### Integrations

- **INTRA** → Documentation hub links to examples
- **All modules** → Examples demonstrate cross-module workflows
- **TEST** → Example code testable as integration tests
