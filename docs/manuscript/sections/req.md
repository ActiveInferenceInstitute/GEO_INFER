## GEO-INFER-REQ — Requirements and Traceability

GEO-INFER-REQ provides requirements analysis, dependency resolution, traceability, and validation for geospatial projects (per its `README.md`). Its package `geo_infer_req` contains `core/` and `__init__.py` for its requirements-engineering interfaces.

The public interface, verified from `__init__.py`, exports components for requirements, dependencies, traceability and validation. Requirements model: `Requirement`, `RequirementType`, `RequirementStatus`, `PriorityLevel`, and `RequirementsAnalyzer` for capturing and classifying what a geospatial project must do. Dependency structure: `DependencyGraph` and `CompletenessReport` for resolving inter-requirement relationships and spotting gaps. Traceability: `TraceabilityManager`, `TraceLink`, `ArtifactType`, `TraceMatrixEntry`, `CoverageReport`, and `ImpactReport` for binding requirements to implementation and test artifacts and for assessing change impact. Validation: `RequirementValidator`, `RequirementSpec`, `ValidationIssue`, `ValidationSeverity`, `ConflictType`, `ConflictDetectionResult`, `ConsistencyReport`, and `FeasibilityAssessment` for checking specs for conflicts, consistency, and feasibility.

The tests exercise analysis, dependency resolution, trace matrices, and validation verdicts on constructed requirement sets.

Under the root README's Module Themes, REQ belongs to Governance, Risk & Domain alongside METAGOV, NORMS, ORG, PEP, SEC, and RISK. Its role there is the contract layer: it is the machinery by which the framework's claims about modules and tests are written down as requirements, traced to artifacts, and checked — the same discipline the manuscript applies to the repository as a whole.
