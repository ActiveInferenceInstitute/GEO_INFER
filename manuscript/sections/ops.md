## GEO-INFER-OPS — Operations and Orchestration

GEO-INFER-OPS provides system orchestration, monitoring, infrastructure management, and deployment automation for the GEO-INFER ecosystem (per its `README.md`). The module carries operational weight beyond code: a `Dockerfile`, `docker-compose.yml`, `deployment/` and `monitoring/` directories, plus `config/` and `logs/`. Its package `geo_infer_ops` contains `app.py`, `config/`, `core/`, `deployment/`, `health/`, and `utils/`.

The public interface, verified from `__init__.py`, exports `Orchestrator` and `Task` with `TaskStatus` for workflow coordination; `DeploymentManager` for release and infrastructure lifecycle; `HealthChecker`, `HealthStatus`, and `HealthCheck` for service condition probes; and `setup_monitoring`, `load_config`, `get_config`, and `setup_testing` for configuration, observability, and test bootstrap.

The tests exercise orchestration, lifecycle cleanup, metrics serving, and health checks. Server cleanup is checked by reopening the port; local lifecycle evidence does not establish a production deployment.

Under the root README's Module Themes, OPS belongs to Agents & AI Orchestration together with AGENT, AG, AI, ANT, and COMMS. Its role there is the runtime substrate: those modules coordinate agentic and AI workloads, and OPS is where their processes are scheduled, deployed, monitored, and health-checked. Within the broader architecture it is the module that keeps every other module runnable, giving the manuscript's reproducibility story an operational anchor in code.
