---
name: geo-infer-norms
description: Normative inference and compliance tracking for geospatial governance. Use when evaluating spatial policy compliance, tracking governance metrics, computing normative content influence (Jaccard similarity), or managing multi-criteria regulatory frameworks.
prerequisites:
  required:
    - geo-infer-data
  recommended:
    - geo-infer-api
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-NORMS

## Instructions

### Core Capabilities

- **Compliance tracking**: Threshold, range, and boolean evaluation with weighted scoring
- **Normative inference**: Content influence measurement via Jaccard similarity
- **Policy evaluation**: Multi-criteria governance assessment with configurable metrics
- **Regulatory frameworks**: Spatial regulation management, jurisdiction tracking

### Key Imports

```python
from geo_infer_norms.core.compliance_tracking import ComplianceTracker
from geo_infer_norms.core.normative_inference import NormativeInference
from geo_infer_norms.models.compliance_status import ComplianceMetric
from geo_infer_norms.models.legal_entity import LegalEntity
from geo_infer_norms.models.regulation import Regulation
```

## Examples

```python
import datetime
from geo_infer_norms.core.compliance_tracking import ComplianceTracker
from geo_infer_norms.models.compliance_status import ComplianceMetric
from geo_infer_norms.models.legal_entity import LegalEntity
from geo_infer_norms.models.regulation import Regulation

metric = ComplianceMetric.create(
    name="air_quality",
    description="PM2.5 threshold",
    regulation_id="reg-air",
    evaluation_type="threshold",
    primary_field="pm25",
    threshold_value=35,
    comparison="less_than",
)
tracker = ComplianceTracker("environmental", compliance_metrics=[metric])
entity = LegalEntity("facility-1", "Facility 1", "facility")
regulation = Regulation(
    "reg-air",
    "Air Quality",
    "PM2.5 limit",
    "environmental",
    "County",
    datetime.date(2026, 1, 1),
)
status = tracker.evaluate_compliance(entity, regulation, {"pm25": 28})
print(f"Overall compliance: {status.compliance_level:.1%}")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog NORMS` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Content influence uses Jaccard similarity (real implementation)
- Compliance evaluation handles threshold/range/boolean types correctly

### API Layer

The `geo_infer_norms.api` package exposes five importable FastAPI routers
(compliance, legal, normative, policy, zoning). The module ships no app
assembly; mount the routers into your own FastAPI application:

```python
from fastapi import FastAPI

from geo_infer_norms.api import ComplianceAPI, LegalAPI, NormativeAPI, PolicyAPI, ZoningAPI

app = FastAPI()
for api_cls in (ComplianceAPI, LegalAPI, NormativeAPI, PolicyAPI, ZoningAPI):
    app.include_router(api_cls().router)
```

### Integrations

- None: this module has no runtime imports of other GEO-INFER modules.
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS`
