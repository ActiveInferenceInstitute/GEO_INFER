## GEO-INFER-EXAMPLES — Cross-Module Integration Demonstrations

GEO-INFER-EXAMPLES is the comprehensive collection of working examples and tutorials demonstrating cross-module integration patterns and real-world applications (module README). It belongs to the Infrastructure & Validation theme: like TEST and INTRA, it is not a domain module but a verification and demonstration surface for the rest of the fleet.

The `geo_infer_examples` package provides a shared orchestration surface. From `src/geo_infer_examples/__init__.py` it exports `ModuleOrchestrator`, `WorkflowDefinition`, `ExecutionStrategy`, and `ModuleStatus`, plus `core` and `models` submodules and `__version__`. These names reveal the module's real function: it does not merely show snippets, it provides an orchestration vocabulary — workflows that sequence multiple GEO-INFER modules with explicit execution strategies and status tracking — and then exercises that vocabulary on real cross-module scenarios. Example scripts exercise this vocabulary; inspect their retained execution receipts before using generated outputs as integration evidence.

The unit and integration tests exercise orchestrator behavior and composed workflows. Each executed integration scenario establishes its recorded boundary behavior rather than universal composability across all module combinations.
