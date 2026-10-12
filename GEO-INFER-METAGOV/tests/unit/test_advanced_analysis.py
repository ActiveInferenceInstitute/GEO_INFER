"""Unit tests for advanced governance analysis."""

import pytest

from geo_infer_metagov.core.advanced_analysis import (
    AdvancedGovernanceAnalyzer,
    ConflictAnalysis,
    ConflictType,
)


class TestAdvancedGovernanceAnalyzer:
    """Test suite for AdvancedGovernanceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer instance."""
        return AdvancedGovernanceAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None

    def test_analyze_power_dynamics(self, analyzer):
        """Test power dynamics analysis."""
        stakeholders = [
            {"name": "Government", "decision_power": 0.8, "interests": ["regulation"]},
            {"name": "Community", "decision_power": 0.3, "interests": ["livelihood"]},
            {"name": "Business", "decision_power": 0.6, "interests": ["profit"]},
        ]

        analysis = analyzer.analyze_power_dynamics(stakeholders)

        assert "power_distribution" in analysis
        assert "herfindahl_index" in analysis
        assert "power_gap" in analysis
        assert "concentration_level" in analysis
        assert "balance_assessment" in analysis
        assert 0 <= analysis["herfindahl_index"] <= 1.0
        assert analysis["balance_assessment"] in [
            "highly_unbalanced",
            "moderately_unbalanced",
            "reasonably_balanced",
            "well_distributed",
        ]

    def test_identify_conflicts(self, analyzer):
        """Test conflict identification."""
        stakeholders = [
            {
                "name": "Group A",
                "interests": ["water"],
                "decision_domains": ["water_management"],
            },
            {
                "name": "Group B",
                "interests": ["land"],
                "decision_domains": ["water_management"],
            },
            {
                "name": "Group C",
                "interests": ["water"],
                "decision_domains": ["land_management"],
            },
        ]

        conflicts = analyzer.identify_conflicts(
            stakeholders=stakeholders,
            decision_domains=["water_management", "land_management"],
        )

        assert isinstance(conflicts, list)
        assert len(conflicts) > 0
        for conflict in conflicts:
            assert isinstance(conflict, ConflictAnalysis)
            assert conflict.conflict_type in ConflictType
            assert 0 <= conflict.severity <= 1.0
            assert len(conflict.stakeholders_involved) > 0

    def test_suggest_governance_improvements(self, analyzer):
        """Test governance improvement suggestions."""
        current_structure = {
            "entities": [{"id": "e1"}, {"id": "e2"}],
            "stakeholders": [{"id": "s1"}, {"id": "s2"}],
        }

        performance_metrics = {
            "efficiency": 0.6,  # Low
            "equity": 0.5,  # Low
            "participation": 0.7,
            "transparency": 0.75,
        }

        suggestions = analyzer.suggest_governance_improvements(
            current_structure=current_structure, performance_metrics=performance_metrics
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert "category" in suggestion
            assert "priority" in suggestion
            assert "suggestion" in suggestion
            assert "estimated_impact" in suggestion
            assert "implementation_effort" in suggestion
            assert "impact_effort_ratio" in suggestion

    def test_scenario_analysis(self, analyzer):
        """Test scenario analysis."""
        current_structure = {
            "entities": [{"id": "e1"}],
            "coordination_mechanisms": ["vertical_alignment"],
        }

        scenarios = [
            {
                "name": "scenario1",
                "description": "Increased stakeholder participation",
                "modifications": {"stakeholder_engagement": "high"},
            },
            {
                "name": "scenario2",
                "description": "Reduced resources",
                "modifications": {"budget": "low"},
            },
        ]

        results = analyzer.scenario_analysis(
            current_structure=current_structure, scenarios=scenarios
        )

        assert "base_case" in results
        assert "scenarios" in results
        assert len(results["scenarios"]) == 2
        for scenario_result in results["scenarios"]:
            assert "name" in scenario_result
            assert "evaluation" in scenario_result
            assert "improvement" in scenario_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
