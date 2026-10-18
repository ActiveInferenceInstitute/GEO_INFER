---
name: geo-infer-ops
description: Operations, workflow orchestration, monitoring, and observability for geospatial infrastructure. Use when composing task dependency graphs, retrieving native workflow results, cancelling async work, configuring metrics and alerts, or managing spatial services.
prerequisites:
  required: []
  recommended:
    - geo-infer-api
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-OPS

## Instructions

### Core Capabilities

- **Monitoring**: System health metrics, spatial operation performance
- **Logging**: Configurable structured logging via ``configure_logging``/``get_logger`` (see ``src/geo_infer_ops/core/logging.py``)
- **Deployment**: Configuration management for spatial services
- **Observability**: Prometheus metrics for cross-module operations
- **Orchestration**: Validated dependency graphs, bounded async scheduling,
  cancellation cleanup and native task results through `Orchestrator`

### Key Imports

```python
from geo_infer_ops.core.monitoring import record_request, record_error, start_metrics_server
from geo_infer_ops.core.deployment import DeploymentManager
from geo_infer_ops.utils.shared_logging import configure_logging
from geo_infer_ops.core.logging import get_logger
from geo_infer_ops.core.orchestrator import Orchestrator
```

## Examples

```python
from geo_infer_ops.core.monitoring import record_request, start_metrics_server

with start_metrics_server(port=9090) as metrics_port:
    record_request("space", "/h3/tessellate", 200, 0.12)
    record_error("space", "TimeoutError")
    print(f"Metrics listening on {metrics_port}")
```

```python
from geo_infer_ops.core.deployment import DeploymentManager
from geo_infer_ops.utils.shared_logging import configure_logging
from geo_infer_ops.core.logging import get_logger

configure_logging(log_level="INFO", json_format=True)
logger = get_logger(__name__)

deployer = DeploymentManager(namespace="geo-prod")
if deployer.build_docker_image(tag="geo-infer-api:local"):
    logger.info("docker_build_ready")
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog OPS` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Uses structured logging (JSON format)
- Prometheus-compatible metrics export
- `start_metrics_server()` yields the selected port and closes server/thread handles on exit
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module OPS`

### Integrations

- **API** → Endpoint monitoring and rate limiting
- **AGENT** → Agent telemetry collection and dashboards
- **IOT** → Sensor health monitoring and alerts
- **LOG** → Logistics operation monitoring
- **SEC** → Security event aggregation

Metrics servers require `prometheus-client>=0.20.0`; the context manager reports
the bound port (including a kernel-selected port 0), shuts down and closes the
server and joins its worker thread on normal or exceptional exit.

### Compose workflow results

Create an `Orchestrator`, add uniquely identified callables, then await
`execute_workflow()`. Missing dependencies and cycles fail before any task runs.
Dependencies start as soon as their upstream tasks complete. Functions execute
inline; returned awaitables run on the current event loop. Blocking synchronous
functions block that loop, and cancellation requires cooperative asynchronous
work. Do not assume thread offloading or a hard execution deadline.

Set `max_retries=0` for operations that cannot safely repeat; the existing
default is three retries. `get_task_result(task_id)` returns the original native
value after completion, while status/history summaries retain string results.
Cancelling the workflow cancels and awaits its owned workers. Concurrent
execute/add/reset operations are rejected until cleanup completes.
