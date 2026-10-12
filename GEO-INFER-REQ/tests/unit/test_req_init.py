"""Tests for GEO-INFER-REQ module initialization and imports."""

from pathlib import Path
import tomllib


class TestReqImports:
    def test_import_module(self):
        import geo_infer_req

        project = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]
        assert geo_infer_req.__version__ == project["version"]

    def test_import_requirements(self):
        from geo_infer_req import (
            RequirementsAnalyzer,
            Requirement,
            RequirementType,
            PriorityLevel,
        )

        assert all(
            isinstance(exported, type)
            for exported in (
                Requirement,
                RequirementType,
                PriorityLevel,
            )
        )

        assert RequirementsAnalyzer is not None
        analyzer = RequirementsAnalyzer()
        assert analyzer is not None

    def test_import_traceability(self):
        from geo_infer_req import (
            TraceabilityManager,
            TraceLink,
            ArtifactType,
        )

        assert all(isinstance(exported, type) for exported in (ArtifactType,))

        assert TraceabilityManager is not None
        assert TraceLink is not None

    def test_import_validation(self):
        from geo_infer_req import (
            RequirementValidator,
            RequirementSpec,
            ValidationSeverity,
        )

        assert all(isinstance(exported, type) for exported in (ValidationSeverity,))

        assert RequirementValidator is not None
        assert RequirementSpec is not None

    def test_core_imports(self):
        from geo_infer_req.core import (
            RequirementsAnalyzer,
            TraceabilityManager,
            RequirementValidator,
        )

        assert RequirementsAnalyzer is not None
        assert TraceabilityManager is not None
        assert RequirementValidator is not None
