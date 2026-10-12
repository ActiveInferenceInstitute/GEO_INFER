"""
Requirement traceability for GEO-INFER-REQ.

Provides trace matrix construction, coverage analysis,
and impact analysis for tracking requirements through implementation.
"""

from dataclasses import dataclass
from enum import Enum


class ArtifactType(Enum):
    """Types of artifacts that requirements can trace to."""

    SOURCE_CODE = "source_code"
    TEST_CASE = "test_case"
    DESIGN_DOCUMENT = "design_document"
    USER_STORY = "user_story"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"
    CONFIGURATION = "configuration"


@dataclass
class TraceLink:
    """A link between a requirement and an implementing artifact."""

    req_id: str
    artifact_id: str
    artifact_type: ArtifactType
    description: str | None = None
    verified: bool = False
    bidirectional: bool = True


@dataclass
class TraceMatrixEntry:
    """An entry in the traceability matrix."""

    req_id: str
    linked_artifacts: dict[str, list[str]]  # artifact_type -> list of artifact_ids
    forward_coverage: float  # % of req traced to artifacts
    backward_coverage: float  # % of artifacts traced to reqs
    verification_status: str


@dataclass
class CoverageReport:
    """Coverage analysis report."""

    total_requirements: int
    traced_requirements: int
    untraced_requirements: list[str]
    coverage_ratio: float
    coverage_by_type: dict[str, float]
    artifact_counts: dict[str, int]
    bidirectional_links: int
    unidirectional_links: int


@dataclass
class ImpactReport:
    """Impact analysis report for requirement changes."""

    changed_requirement: str
    directly_affected_artifacts: list[str]
    indirectly_affected_requirements: list[str]
    indirectly_affected_artifacts: list[str]
    impact_severity: float
    affected_count: int


class TraceabilityManager:
    """
    Manages requirement traceability across the project lifecycle.

    Builds and maintains trace matrices, performs coverage analysis,
    and conducts impact analysis for requirement changes.
    """

    def __init__(self) -> None:
        self._links: list[TraceLink] = []
        self._req_ids: set[str] = set()
        self._artifact_ids: set[str] = set()
        # Dependency map from RequirementsAnalyzer context
        self._req_dependencies: dict[str, list[str]] = {}

    def register_requirement(
        self, req_id: str, dependencies: list[str] | None = None
    ) -> None:
        """
        Register a requirement for traceability tracking.

        Args:
            req_id: The requirement identifier.
            dependencies: Optional list of dependency requirement IDs.
        """
        self._req_ids.add(req_id)
        if dependencies:
            self._req_dependencies[req_id] = dependencies

    def register_requirements(self, req_ids: list[str]) -> None:
        """
        Register multiple requirements.

        Args:
            req_ids: List of requirement identifiers.
        """
        for rid in req_ids:
            self.register_requirement(rid)

    def add_trace_link(self, link: TraceLink) -> None:
        """
        Add a trace link between a requirement and an artifact.

        Args:
            link: The trace link to add.
        """
        self._req_ids.add(link.req_id)
        self._artifact_ids.add(link.artifact_id)
        self._links.append(link)

    def add_trace_links(self, links: list[TraceLink]) -> None:
        """
        Add multiple trace links.

        Args:
            links: List of trace links.
        """
        for link in links:
            self.add_trace_link(link)

    def build_trace_matrix(self) -> dict[str, TraceMatrixEntry]:
        """
        Build the complete traceability matrix.

        Returns:
            Mapping of req_id to TraceMatrixEntry.
        """
        matrix: dict[str, TraceMatrixEntry] = {}

        for req_id in self._req_ids:
            req_links = [link for link in self._links if link.req_id == req_id]

            linked_artifacts: dict[str, list[str]] = {}
            verified_count = 0
            for link in req_links:
                art_type = link.artifact_type.value
                if art_type not in linked_artifacts:
                    linked_artifacts[art_type] = []
                linked_artifacts[art_type].append(link.artifact_id)
                if link.verified:
                    verified_count += 1

            total_links = len(req_links)
            forward_coverage = 1.0 if total_links > 0 else 0.0

            # Backward: what fraction of linked artifacts trace back
            bidirectional_count = sum(1 for link in req_links if link.bidirectional)
            backward_coverage = (
                bidirectional_count / total_links if total_links > 0 else 0.0
            )

            if total_links == 0:
                status = "untraced"
            elif verified_count == total_links:
                status = "fully_verified"
            elif verified_count > 0:
                status = "partially_verified"
            else:
                status = "unverified"

            matrix[req_id] = TraceMatrixEntry(
                req_id=req_id,
                linked_artifacts=linked_artifacts,
                forward_coverage=round(forward_coverage, 4),
                backward_coverage=round(backward_coverage, 4),
                verification_status=status,
            )

        return matrix

    def analyze_coverage(self) -> CoverageReport:
        """
        Analyze trace coverage across all requirements.

        Returns:
            CoverageReport with coverage metrics.
        """
        # Determine which requirements have at least one trace link
        traced_reqs: set[str] = set()
        for link in self._links:
            traced_reqs.add(link.req_id)

        untraced = sorted(self._req_ids - traced_reqs)
        total = len(self._req_ids)
        coverage_ratio = len(traced_reqs) / total if total > 0 else 0.0

        # Coverage by artifact type
        type_reqs: dict[str, set[str]] = {}
        artifact_counts: dict[str, int] = {}
        bidirectional = 0
        unidirectional = 0

        for link in self._links:
            art_type = link.artifact_type.value
            if art_type not in type_reqs:
                type_reqs[art_type] = set()
            type_reqs[art_type].add(link.req_id)
            artifact_counts[art_type] = artifact_counts.get(art_type, 0) + 1

            if link.bidirectional:
                bidirectional += 1
            else:
                unidirectional += 1

        coverage_by_type = {}
        for art_type, req_set in type_reqs.items():
            coverage_by_type[art_type] = (
                round(len(req_set) / total, 4) if total > 0 else 0.0
            )

        return CoverageReport(
            total_requirements=total,
            traced_requirements=len(traced_reqs),
            untraced_requirements=untraced,
            coverage_ratio=round(coverage_ratio, 4),
            coverage_by_type=coverage_by_type,
            artifact_counts=artifact_counts,
            bidirectional_links=bidirectional,
            unidirectional_links=unidirectional,
        )

    def analyze_impact(self, changed_req_id: str) -> ImpactReport:
        """
        Analyze the impact of changing a specific requirement.

        Identifies directly affected artifacts and indirectly affected
        requirements and their artifacts through the dependency graph.

        Args:
            changed_req_id: The requirement being changed.

        Returns:
            ImpactReport with affected elements.

        Raises:
            KeyError: If the requirement is not registered.
        """
        if changed_req_id not in self._req_ids:
            raise KeyError(f"Requirement {changed_req_id} not registered")

        # Direct artifacts
        direct_artifacts = [
            link.artifact_id for link in self._links if link.req_id == changed_req_id
        ]

        # Indirect requirements (those that depend on changed req)
        indirect_reqs: list[str] = []
        visited: set[str] = {changed_req_id}
        queue = [changed_req_id]

        # Build reverse dependency map
        reverse_deps: dict[str, list[str]] = {}
        for rid, deps in self._req_dependencies.items():
            for dep in deps:
                if dep not in reverse_deps:
                    reverse_deps[dep] = []
                reverse_deps[dep].append(rid)

        while queue:
            current = queue.pop(0)
            for dependent in reverse_deps.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    indirect_reqs.append(dependent)
                    queue.append(dependent)

        # Indirect artifacts
        indirect_artifacts: list[str] = []
        for rid in indirect_reqs:
            for link in self._links:
                if link.req_id == rid and link.artifact_id not in direct_artifacts:
                    indirect_artifacts.append(link.artifact_id)

        total_affected = (
            len(direct_artifacts) + len(indirect_reqs) + len(indirect_artifacts)
        )
        total_trackable = len(self._artifact_ids) + len(self._req_ids)
        severity = total_affected / total_trackable if total_trackable > 0 else 0.0

        return ImpactReport(
            changed_requirement=changed_req_id,
            directly_affected_artifacts=direct_artifacts,
            indirectly_affected_requirements=indirect_reqs,
            indirectly_affected_artifacts=indirect_artifacts,
            impact_severity=round(severity, 4),
            affected_count=total_affected,
        )

    def get_unverified_links(self) -> list[TraceLink]:
        """
        Get all trace links that have not been verified.

        Returns:
            List of unverified trace links.
        """
        return [link for link in self._links if not link.verified]

    def verify_link(self, req_id: str, artifact_id: str) -> bool:
        """
        Mark a trace link as verified.

        Args:
            req_id: The requirement ID.
            artifact_id: The artifact ID.

        Returns:
            True if the link was found and verified, False otherwise.
        """
        for link in self._links:
            if link.req_id == req_id and link.artifact_id == artifact_id:
                link.verified = True
                return True
        return False
