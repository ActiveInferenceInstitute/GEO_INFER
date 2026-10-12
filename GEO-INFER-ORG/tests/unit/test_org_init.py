"""Tests for GEO-INFER-ORG module initialization and imports."""

from pathlib import Path
import tomllib


class TestOrgImports:
    def test_import_module(self):
        import geo_infer_org

        project = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]
        assert geo_infer_org.__version__ == project["version"]

    def test_import_organization(self):
        from geo_infer_org import (
            OrganizationModel,
            OrgUnit,
            Role,
            OrgStructureType,
            RoleLevel,
        )

        assert all(
            isinstance(exported, type)
            for exported in (
                OrgUnit,
                Role,
                OrgStructureType,
                RoleLevel,
            )
        )

        assert OrganizationModel is not None
        model = OrganizationModel()
        assert model is not None

    def test_import_governance(self):
        from geo_infer_org import (
            VotingEngine,
            ConsensusModel,
            VotingMethod,
            Vote,
            Proposal,
        )

        assert all(
            isinstance(exported, type)
            for exported in (
                VotingMethod,
                Vote,
                Proposal,
            )
        )

        assert VotingEngine is not None
        assert ConsensusModel is not None

    def test_import_collaboration(self):
        from geo_infer_org import (
            CollaborationNetwork,
            TeamFormation,
            TeamMember,
            CollaborationEdge,
        )

        assert all(
            isinstance(exported, type)
            for exported in (
                TeamMember,
                CollaborationEdge,
            )
        )

        assert CollaborationNetwork is not None
        assert TeamFormation is not None

    def test_core_imports(self):
        from geo_infer_org.core import (
            OrganizationModel,
            VotingEngine,
            CollaborationNetwork,
        )

        assert OrganizationModel is not None
        assert VotingEngine is not None
        assert CollaborationNetwork is not None
