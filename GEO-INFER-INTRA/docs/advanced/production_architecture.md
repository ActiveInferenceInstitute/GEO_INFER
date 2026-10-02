# Production Architecture

GEO-INFER packages provide module behavior. A deployment must explicitly choose
its request surface, storage, authentication, authorization, resource budgets,
observability, and recovery procedures. Python composition can start inside one
service; separate services when measured scaling or ownership needs justify them.

## An application-owned HTTP boundary

This small FastAPI application wraps the actual TIME validator. It runs entirely
inside a local test client and rejects ambiguous timestamps at the request boundary.
It is an application example, not a package-provided gateway or deployed service.

```python
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from geo_infer_time import normalize_timestamp

app = FastAPI()
@app.get("/utc")
def utc(timestamp: str):
    try:
        instant = normalize_timestamp(timestamp)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"timestamp": instant.isoformat()}

with TestClient(app) as client:
    assert client.get("/utc", params={"timestamp": "2024-01-01T00:00:00Z"}).json() == {"timestamp": "2024-01-01T00:00:00+00:00"}
    assert client.get("/utc", params={"timestamp": "2024-01-01"}).status_code == 422
```

## Storage, operations, and recovery

Keep source observations and model outputs distinguishable. Record coordinate
reference system, timestamp identity, units, lineage, model configuration, and
schema version alongside each stored result. A relational spatial database, tiled
raster storage, and a cache serve different access patterns; their credentials,
transactions, retention, and consistency remain deployment responsibilities.

Use module loggers, explicit request/process deadlines, and bounded buffers. Log
failure identity without personal records or credentials. Measure health against
actual dependencies and expose readiness separately from process liveness. Test
backup restoration and rollback against a retained artifact before stating a
recovery objective; this repository does not establish an availability SLA.

For changes in matrix axes or timestamp rules, validate migrations on representative
data before cutover. A local TestClient pass is HTTP contract evidence only;
hosted CI, installed-wheel behavior, and live deployment acceptance are separate.
See [scaling](scaling_guide.md), [performance](performance_optimization.md), and
[installation](../installation.md).
