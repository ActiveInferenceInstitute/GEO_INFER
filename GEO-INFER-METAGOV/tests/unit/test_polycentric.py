"""Unit tests for polycentric governance system."""

import pytest

from geo_infer_metagov.core.polycentric import (
    PolycentricGovernanceSystem,
)


class TestPolycentricGovernanceSystem:
    """Test suite for PolycentricGovernanceSystem."""

    @pytest.fixture
    def system(self):
        """Create a test system instance."""
        return PolycentricGovernanceSystem(
            governance_model="polycentric",
            coordination_mechanism="network_based",
            redundancy_level="adaptive",
        )

    def test_system_initialization(self, system):
        """Test system initializes correctly."""
        assert system.governance_model == "polycentric"
        assert system.coordination_mechanism == "network_based"
        assert system.redundancy_level == "adaptive"
        assert len(system.polycentric_designs) == 0

    def test_design_polycentric_structure(self, system):
        """Test polycentric structure design."""
        design = system.design_polycentric_structure(
            governing_bodies=[
                {
                    "id": "body_1",
                    "name": "Authority 1",
                    "level": "local",
                    "domains": ["water"],
                },
                {
                    "id": "body_2",
                    "name": "Authority 2",
                    "level": "regional",
                    "domains": ["water", "land"],
                },
            ],
            jurisdictional_overlaps={"body_1": ["body_2"]},
            spatial_scales=["local", "regional"],
            functional_domains=["water", "land"],
            feedback_mechanisms={"monitoring": "continuous"},
        )

        assert design is not None
        assert design.design_id is not None
        assert len(design.governing_bodies) == 2
        assert "redundancy_assessment" in design.__dict__

    def test_analyze_authority_relationships(self, system):
        """Test authority relationship analysis."""
        analysis = system.analyze_authority_relationships(
            authorities=[
                {
                    "id": "a1",
                    "name": "Authority 1",
                    "domains": ["water"],
                    "capacity": 0.8,
                },
                {
                    "id": "a2",
                    "name": "Authority 2",
                    "domains": ["water", "land"],
                    "capacity": 0.7,
                },
                {
                    "id": "a3",
                    "name": "Authority 3",
                    "domains": ["land"],
                    "capacity": 0.9,
                },
            ],
            relationships=["coordination", "cooperation"],
            effectiveness_measures=["efficiency", "equity"],
        )

        assert "authority_count" in analysis
        assert analysis["authority_count"] == 3
        assert "coordination_index" in analysis
        assert 0 <= analysis["coordination_index"] <= 1.0
        assert "network_density" in analysis
        assert "overlap_analysis" in analysis
        assert "redundancy_metrics" in analysis
        assert "coordination_failure_risk" in analysis

    def test_redundancy_assessment(self, system):
        """Test redundancy assessment."""
        design = system.design_polycentric_structure(
            governing_bodies=[
                {"id": "b1", "domains": ["water"]},
                {"id": "b2", "domains": ["water", "land"]},
                {"id": "b3", "domains": ["land"]},
            ],
            jurisdictional_overlaps={"b1": ["b2"], "b2": ["b3"]},
            spatial_scales=["local", "regional"],
            functional_domains=["water", "land"],
            feedback_mechanisms={},
        )

        redundancy = design.redundancy_assessment
        assert "redundancy_ratio" in redundancy
        assert "resilience_level" in redundancy
        assert "efficiency_impact" in redundancy
        assert 0 <= redundancy["redundancy_ratio"] <= 1.0

    def test_authority_overlap_analysis(self, system):
        """Test authority overlap analysis."""
        analysis = system.analyze_authority_relationships(
            authorities=[
                {"id": "a1", "jurisdiction": ["region1"], "domains": ["water"]},
                {
                    "id": "a2",
                    "jurisdiction": ["region1", "region2"],
                    "domains": ["water", "land"],
                },
            ],
            relationships=["coordination"],
            effectiveness_measures=["efficiency"],
        )

        assert "overlap_analysis" in analysis
        overlap = analysis["overlap_analysis"]
        assert "jurisdictional_overlaps" in overlap
        assert "functional_overlaps" in overlap
        assert "overlap_matrix" in overlap

    def test_coordination_failure_risk(self, system):
        """Test coordination failure risk assessment."""
        analysis = system.analyze_authority_relationships(
            authorities=[
                {"id": "a1", "domains": ["water"], "capacity": 0.3},  # Low capacity
                {"id": "a2", "domains": ["water"], "capacity": 0.4},  # Low capacity
            ],
            relationships=["competition"],  # Competition without coordination
            effectiveness_measures=["efficiency"],
        )

        assert "coordination_failure_risk" in analysis
        assert 0 <= analysis["coordination_failure_risk"] <= 1.0
        assert "resilience_score" in analysis
        assert 0 <= analysis["resilience_score"] <= 1.0

    def test_multiple_designs(self, system):
        """Test creating multiple polycentric designs."""
        design1 = system.design_polycentric_structure(
            governing_bodies=[{"id": "b1"}],
            jurisdictional_overlaps={},
            spatial_scales=["local"],
            functional_domains=["water"],
            feedback_mechanisms={},
        )

        design2 = system.design_polycentric_structure(
            governing_bodies=[{"id": "b2"}],
            jurisdictional_overlaps={},
            spatial_scales=["regional"],
            functional_domains=["land"],
            feedback_mechanisms={},
        )

        assert design1.design_id != design2.design_id
        assert len(system.polycentric_designs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
