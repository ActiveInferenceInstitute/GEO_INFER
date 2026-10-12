# GEO-INFER-OPS Deployment Guide

Instructions for running GEO-INFER-OPS locally, in Docker, and on Kubernetes.

## Development Deployment

The module is a member of the root uv workspace. From the repository root:

```bash
# Synchronize the workspace package
uv sync --package geo-infer-ops

# Run the FastAPI application
uv run python -m geo_infer_ops.app
```

## Docker Deployment

```bash
# Build the Docker image (from GEO-INFER-OPS/)
docker build -t geo-infer-ops:latest .

# Run the container
docker run -p 8000:8000 -p 9090:9090 -v "$(pwd)/config:/app/config" geo-infer-ops:latest
```

## Docker Compose Deployment

For a local environment with monitoring:

```bash
docker-compose up -d       # Start all services
docker-compose logs -f     # View logs
docker-compose down        # Stop all services
```

## Kubernetes Deployment

```bash
# Apply the Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check the deployment status
kubectl get pods -n geo-infer

# Port forward to access the API locally
kubectl port-forward svc/geo-infer-ops 8000:8000 -n geo-infer
```

## Environment Variables

The application entrypoint (`geo_infer_ops.app`) reads its YAML configuration
through `geo_infer_ops.utils.config.load_config`, which applies
`GEO_INFER_OPS_<SECTION>_<KEY>` overrides.

| Variable | Description | Default |
|----------|-------------|---------|
| `GEO_INFER_OPS_CONFIG` | Path to configuration file | `config/local.yaml` |
| `GEO_INFER_OPS_LOGGING_LEVEL` | Log level | `INFO` |
| `GEO_INFER_OPS_SERVICE_PORT` | Service port | `8000` |
| `GEO_INFER_OPS_MONITORING_ENABLED` | Enable Prometheus metrics | `true` |

## Health Checks

The service exposes `/health`, which returns HTTP 200 when the service is healthy.

## Monitoring

Prometheus metrics are served at `/metrics` and can be visualized in Grafana.
The docker-compose setup includes Prometheus and Grafana services.
