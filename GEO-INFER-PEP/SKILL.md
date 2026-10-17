---
name: geo-infer-pep
description: People, Engagement, Performance operations for GEO-INFER. Use when building HR/CRM/Talent data pipelines (CSV import, cleaning, enrichment, reports, plots), the PEPEngine workflow orchestration, or the FastAPI PEP service.
prerequisites:
  required: []
  recommended: []
difficulty: intermediate
estimated_time: 45min
examples_dir: ./examples/
---

# GEO-INFER-PEP

## Instructions

GEO-INFER-PEP implements People Operations (HR), Customer Relationship (CRM),
and Talent Acquisition pipelines: CSV importers, data transformers (clean/enrich),
report generators, matplotlib/seaborn visualizations, a workflow engine, and
FastAPI routers. It is a standalone module with no geo_infer_* dependencies.

### Core Capabilities

- **HR pipeline**: `CSVHRImporter` → `clean_employee_data` / `enrich_employee_data` → headcount/diversity reports → matplotlib plots
- **CRM pipeline**: `CSVCRMImporter` → `clean_customer_data` / `enrich_customer_data` → segmentation/conversion reports → status/source plots
- **Talent pipeline**: `CSVTalentImporter` → `clean_candidate_data` / `enrich_candidate_data` → pipeline/time-to-hire reports → pipeline plots
- **Workflow orchestration**: `PEPEngine` (dashboards, health checks, lifecycle) and `PEPOrchestrator` (multi-step workflow tracking)
- **FastAPI service**: routers under `/pep/{hr,crm,talent}` with CSV upload, CRUD, search, reports, and dashboard endpoints

### Key Imports

```python
from geo_infer_pep import (
    Candidate,
    Customer,
    Employee,
    PEPDataManager,
    PEPEngine,
    api_router,
    clear_all_data,
    generate_comprehensive_crm_dashboard,
    generate_comprehensive_hr_dashboard,
    generate_comprehensive_talent_dashboard,
    generate_quarterly_people_report,
    get_all_candidates,
    get_all_customers,
    get_all_employees,
    import_crm_data_from_csv,
    import_hr_data_from_csv,
    import_talent_data_from_csv,
    process_employee_onboarding_workflow,
)
```

All names above resolve against the installed package (verified against
`geo_infer_pep.__all__`). The FastAPI routers require `fastapi`
(a declared dependency); everything else imports without it.

### Data Store Semantics

- `geo_infer_pep.core.data_store.pep_data_manager` is the process-wide shared
  in-memory store. The `methods` layer and the FastAPI endpoints both mutate
  it, so data uploaded via the API is visible to `get_all_employees()` and
  vice versa.
- `clear_all_data()` (and `PEPEngine.shutdown()` on the default manager) are
  **destructive**: they delete every employee, customer, candidate, and
  requisition record.
- Pass a fresh `PEPDataManager()` to `PEPEngine(data_manager=...)` for an
  isolated store (tests).
- `POST /pep/hr/upload/csv` and the CRM/talent upload endpoints **append** to
  the shared store; re-uploading the same CSV creates duplicate records.

## Examples

```python
from geo_infer_pep import (
    PEPEngine,
    import_hr_data_from_csv,
    generate_comprehensive_hr_dashboard,
    get_all_employees,
)

engine = PEPEngine()
engine.initialize()

result = engine.import_hr_data("employees.csv")
assert result["success"] is True

employees = get_all_employees()
dashboard = generate_comprehensive_hr_dashboard()
assert "total_employees" in dashboard
```

```python
from geo_infer_pep import (
    Candidate,
    CandidateStatus,
    Offer,
    process_employee_onboarding_workflow,
)
from datetime import datetime

# A candidate must exist in the shared store with an accepted offer
from geo_infer_pep import pep_data_manager
candidate = Candidate(
    candidate_id="cand_001",
    first_name="New",
    last_name="Hire",
    email="new.hire@example.com",
    applied_at=__import__("datetime").datetime.now(),
    status=CandidateStatus.OFFER_ACCEPTED,
    offer=Offer(
        offer_id="offer_001",
        offered_at=__import__("datetime").datetime.now().date(),
        accepted_at=__import__("datetime").datetime.now().date(),
    ),
)
pep_data_manager.candidates.append(candidate)

completed = process_employee_onboarding_workflow({"candidate_id": "cand_001"})
assert completed is True
```

```python
# Run the FastAPI service
from fastapi import FastAPI
from geo_infer_pep import api_router

app = FastAPI()
app.include_router(api_router)  # routes under /pep/*
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog PEP` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

### Method Contracts

- `import_hr_data_from_csv(path)` / `import_crm_data_from_csv(path)` return the
  list of processed records and store them in the shared store; import/transform
  failures are logged and re-raised.
- `import_talent_data_from_csv(candidates_csv, requisitions_csv)` returns
  `{"candidates": <stored count>, "requisitions": <imported count>,
  "processed_successfully": True}`.
- `process_employee_onboarding_workflow(data)` raises `ValueError` when
  `candidate_id` is missing, returns `False` when the candidate is absent or
  not in `OFFER_ACCEPTED` state, and returns `True` after creating and storing
  the employee. `benefits_client` / `learning_client` callables in `data` are
  invoked with the new employee.
- `PEPDataManager.get_employees(filters=...)` validates filter keys against
  `Employee` model fields (plus the `status`/`gender`/`department` aliases) and
  raises `ValueError` for unknown keys; the same validation applies to
  customers, candidates, and requisitions.
- Dashboards return `{"message": "No ... data available"}` when the
  corresponding store is empty, otherwise a metrics dict with a live
  `generated_at` timestamp.
- Library code logs via `logging.getLogger(__name__)`; no module mutates
  process-wide state (warnings, handlers) at import.

### Integrations

- PEP is standalone; no other GEO-INFER modules are imported.
- Mount `api_router` in any FastAPI app to expose the PEP service.

## Verification

```bash
uv run pytest GEO-INFER-PEP/tests -q
uv run python GEO-INFER-PEP/examples/basic_hr_example.py
uv run python GEO-INFER-PEP/examples/basic_crm_example.py
uv run python GEO-INFER-PEP/examples/onboarding_workflow_example.py
```
