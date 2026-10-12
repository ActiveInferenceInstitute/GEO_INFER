---
name: geo-infer-ops
description: Operations, monitoring, and observability for geospatial infrastructure. Use when setting up monitoring dashboards, configuring alerts, tracking system health, or managing deployment of spatial services.
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

### Key Imports

```python
from geo_infer_ops.core.monitoring import record_request, record_error, start_metrics_server
from geo_infer_ops.core.deployment import DeploymentManager
from geo_infer_ops.utils.shared_logging import configure_logging
from geo_infer_ops.core.logging import get_logger
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
