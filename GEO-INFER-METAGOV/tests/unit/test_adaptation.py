"""Unit tests for adaptive governance system."""

import pytest

from geo_infer_metagov.core.adaptation import (
    AdaptiveGovernanceSystem,
)


class TestAdaptiveGovernanceSystem:
    """Test suite for AdaptiveGovernanceSystem."""

    @pytest.fixture
    def system(self):
        """Create a test system instance."""
        return AdaptiveGovernanceSystem(
            learning_approach="adaptive_management",
            timeframe="multi_year_cycles",
            feedback_mechanisms="real_time",
        )

    def test_system_initialization(self, system):
        """Test system initializes correctly."""
        assert system.learning_approach == "adaptive_management"
        assert system.timeframe == "multi_year_cycles"
        assert system.feedback_mechanisms == "real_time"
        assert len(system.adaptive_cycles) == 0

    def test_establish_adaptive_cycle(self, system):
        """Test establishing adaptive management cycle."""
        cycle = system.establish_adaptive_cycle(
            governance_domain="natural_resource_management",
            decision_frequency="annual_review",
            learning_mechanisms=["monitoring", "evaluation", "adjustment"],
            stakeholder_participation="continuous",
        )

        assert cycle is not None
        assert cycle.cycle_id is not None
        assert cycle.governance_domain == "natural_resource_management"
        assert cycle.monitoring_plan is not None
        assert cycle.evaluation_schedule is not None

    def test_monitor_performance(self, system):
        """Test performance monitoring."""
        results = system.monitor_performance(
            governance_indicators=["effectiveness", "equity", "sustainability"],
            data_sources=["administrative", "stakeholder_feedback"],
            evaluation_periods="annual",
        )

        assert "indicators" in results
        assert "performance_scores" in results
        assert "overall_performance" in results
        assert "performance_trends" in results
        assert "data_quality" in results
        assert "performance_gaps" in results
        assert len(results["performance_scores"]) == 3
        assert 0 <= results["overall_performance"] <= 1.0

    def test_adapt_governance(self, system):
        """Test governance adaptation."""
        performance_results = {
            "performance_scores": {
                "effectiveness": 0.6,
                "equity": 0.5,
                "sustainability": 0.7,
            },
            "overall_performance": 0.6,
            "performance_gaps": {"effectiveness": 0.4, "equity": 0.5},
        }

        learning_outcomes = {
            "lessons": [
                "improve_transparency",
                "increase_participation",
                "successful_monitoring",
            ]
        }

        adaptation_pathways = [
            {
                "name": "pathway1",
                "expected_impact": 0.8,
                "feasibility": 0.7,
                "cost": 0.4,
                "target_domains": ["effectiveness"],
            },
            {
                "name": "pathway2",
                "expected_impact": 0.6,
                "feasibility": 0.9,
                "cost": 0.3,
                "target_domains": ["equity"],
            },
        ]

        adapted = system.adapt_governance(
            performance_results=performance_results,
            learning_outcomes=learning_outcomes,
            scenario_changes=[],
            adaptation_pathways=adaptation_pathways,
        )

        assert "adaptations_made" in adapted
        assert "pathways_selected" in adapted
        assert "implementation_timeline" in adapted
        assert "stakeholder_support" in adapted
        assert "adaptation_quality" in adapted
        assert adapted["adaptations_made"] > 0
        assert 0 <= adapted["stakeholder_support"] <= 1.0
        assert 0 <= adapted["adaptation_quality"] <= 1.0

    def test_adapt_governance_no_pathways(self, system):
        """Test adaptation with no pathways."""
        adapted = system.adapt_governance(
            performance_results={"overall_performance": 0.5},
            learning_outcomes={},
            scenario_changes=[],
            adaptation_pathways=[],
        )

        assert adapted["adaptations_made"] == 0
        assert len(adapted["pathways_selected"]) == 0

    def test_performance_trends(self, system):
        """Test performance trend detection."""
        results = system.monitor_performance(
            governance_indicators=["effectiveness", "efficiency"],
            data_sources=["administrative"],
            evaluation_periods="quarterly",
        )

        assert "performance_trends" in results
        for indicator in results["indicators"]:
            assert indicator in results["performance_trends"]
            trend = results["performance_trends"][indicator]
            assert trend in ["improving", "stable", "declining"]

    def test_data_quality_assessment(self, system):
        """Test data quality assessment."""
        results = system.monitor_performance(
            governance_indicators=["transparency"],
            data_sources=["administrative", "stakeholder_feedback"],
            evaluation_periods="monthly",
        )

        assert "data_quality" in results
        for indicator in results["indicators"]:
            assert indicator in results["data_quality"]
            quality = results["data_quality"][indicator]
            assert "completeness" in quality
            assert "reliability" in quality
            assert "timeliness" in quality
            assert 0 <= quality["completeness"] <= 1.0

    def test_monitor_performance_determinism_pin(self, system):
        """Two identical calls produce identical dicts (determinism pin).

        Mirrors the GEO-INFER-MATH MCMC determinism-pin precedent: the
        indicator-hash seeded derivation makes simulated scores
        reproducible. The wall-clock ``monitoring_timestamp`` is
        excluded; every other field, including the ``simulated`` flag,
        is pinned.
        """
        results_a = system.monitor_performance(
            governance_indicators=["effectiveness", "equity", "transparency"],
            data_sources=["administrative", "stakeholder_feedback"],
            evaluation_periods="annual",
        )
        results_b = system.monitor_performance(
            governance_indicators=["effectiveness", "equity", "transparency"],
            data_sources=["administrative", "stakeholder_feedback"],
            evaluation_periods="annual",
        )

        assert results_a.pop("monitoring_timestamp") is not None
        assert results_b.pop("monitoring_timestamp") is not None
        assert results_a == results_b
        assert results_a["simulated"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
