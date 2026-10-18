# Advanced Workflows

Advanced work starts with the numerical and data contract of an actual module,
then measures the workload before changing its execution strategy. This section
collects implementation guidance; it does not certify a production deployment,
GPU backend, or fixed performance gain.

## Choose a task

- [Performance optimization](performance_optimization.md): profile real operations,
  preserve numerical invariants, and bound allocations.
- [Memory management](memory_management.md): account for working sets, copies,
  and output cardinality.
- [Scaling](scaling_guide.md): evaluate chunking and distributed execution for
  the actual workload; the page's size thresholds are planning heuristics.
- [Custom models](custom_models.md): define states, observations, controls, units,
  and evaluation before extending a model.
- [Production architecture](production_architecture.md): select and qualify
  the components required by an application.

## Measure and preserve evidence

Use [backend profiling](../integration/backend_profiling.md) for actual tracing
or sampling receipts and [agent interfaces](../integration/agent_interfaces.md)
for composable script, CLI, and MCP execution. Profiling identifies costs in a
specific run; it does not prove a throughput speedup. Verify outputs against a
correctness oracle and report repeated, comparable measurements when making a
performance claim.

Distributed processing, cloud services, external databases, and GPU execution
require their own dependencies, deployment configuration, and actual acceptance
evidence. A concept diagram or optional import does not qualify those systems.
The [testing guide](../developer_guide/testing_guide.md) explains the repository
gates; domain validity and hardware qualification remain separate concerns.

## Source and research references

Use the [module catalog](../modules/index.md) and
[API reference policy](../api/reference.md) to find current interfaces. The
repository's [manuscript bibliography](../../../manuscript/references.bib)
records scholarly references; [research artifacts](../integration/research_artifacts.md)
explains the separate source-bound publication workflow.

Report reproducible problems or proposals through the
[GitHub issue tracker](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues)
and follow [contribution guidance](../developer_guide/contributing.md).
This page does not advertise an unverified community forum, Discord invite, or
support mailbox.
