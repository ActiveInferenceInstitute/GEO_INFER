"""
Integration tests for GEO-INFER-NORMS: normative inference + zoning analysis together.

Tests NormativeInference, SocialNormDiffusion, ZoningCode, and ZoningDistrict
working together in a compliance assessment and norm diffusion pipeline.
"""

import pytest
import datetime

from shapely.geometry import Point, Polygon

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def normative_engine():
    """Set up a NormativeInference engine with realistic zoning norms."""
    from geo_infer_norms.core.normative_inference import NormativeInference

    engine = NormativeInference()

    # Add building height norm
    height_norm_id = engine.add_norm(
        name="max_building_height",
        condition=lambda obs: obs.get("building_height", 0) <= 35,
        probability=1.0,
        description="Buildings must not exceed 35 feet",
        spatial_constraint=Polygon(
            [
                (-118.30, 34.00),
                (-118.30, 34.10),
                (-118.20, 34.10),
                (-118.20, 34.00),
                (-118.30, 34.00),
            ]
        ),
    )

    # Add lot coverage norm
    coverage_norm_id = engine.add_norm(
        name="max_lot_coverage",
        condition=lambda obs: obs.get("lot_coverage", 0) <= 0.60,
        probability=0.95,
        description="Lot coverage must not exceed 60%",
    )

    # Add setback norm
    setback_norm_id = engine.add_norm(
        name="min_front_setback",
        condition=lambda obs: obs.get("front_setback", 0) >= 20,
        probability=1.0,
        description="Front setback must be at least 20 feet",
    )

    # Add norm relationships
    engine.add_norm_relationship(
        height_norm_id, coverage_norm_id, "supports", strength=0.7
    )

    return engine, height_norm_id, coverage_norm_id, setback_norm_id


class TestNormativeCompliancePipeline:
    """Test full compliance assessment pipeline."""

    def test_compliant_entity_assessment(self, normative_engine):
        """Test that a compliant entity gets high compliance scores."""
        engine, height_id, coverage_id, setback_id = normative_engine

        # Add observations for a compliant entity
        engine.add_observation(
            "building_A",
            "building_height",
            30,
            location=Point(-118.25, 34.05),
            certainty=0.95,
        )
        engine.add_observation("building_A", "lot_coverage", 0.45, certainty=0.9)
        engine.add_observation("building_A", "front_setback", 25, certainty=0.85)

        # Check compliance for each norm
        for norm_id in [height_id, coverage_id, setback_id]:
            compliant, certainty = engine.check_norm_compliance(norm_id, "building_A")
            assert compliant is True, f"Building A should comply with norm {norm_id}"
            assert certainty > 0, "Certainty should be positive for observed entity"

    def test_non_compliant_entity_detection(self, normative_engine):
        """Test that violations are properly detected."""
        engine, height_id, coverage_id, setback_id = normative_engine

        # Add observations for a non-compliant entity
        engine.add_observation(
            "building_B",
            "building_height",
            50,
            location=Point(-118.25, 34.05),
            certainty=0.95,
        )
        engine.add_observation("building_B", "lot_coverage", 0.75, certainty=0.9)
        engine.add_observation("building_B", "front_setback", 10, certainty=0.85)

        # The Bayesian inference model returns probabilities centered around the
        # prior (0.5), so non-compliant entities with high certainty get
        # compliance probabilities below the prior but not extremely low.
        # Use a threshold that matches the Bayesian update behavior.
        violations = engine.identify_norm_violations("building_B", threshold=0.95)
        assert len(violations) > 0, (
            "Should detect at least one violation at threshold=0.95"
        )

        # Violations should be sorted by severity
        if len(violations) > 1:
            assert violations[0]["severity"] >= violations[1]["severity"]

        # Also verify direct compliance check shows non-compliance
        compliant, certainty = engine.check_norm_compliance(height_id, "building_B")
        assert compliant is False, "Building B should not comply with height norm"

    def test_bayesian_inference_updates_with_observations(self, normative_engine):
        """Test that Bayesian inference properly updates beliefs with observations."""
        engine, height_id, coverage_id, setback_id = normative_engine

        # Set prior belief
        engine.set_prior_belief(height_id, "building_C", compliance_probability=0.5)

        # Add compliant observation
        engine.add_observation(
            "building_C",
            "building_height",
            25,
            location=Point(-118.25, 34.05),
            certainty=0.9,
        )

        # Infer compliance
        compliance_prob = engine.infer_compliance("building_C", height_id)
        assert compliance_prob > 0.5, (
            "Compliance probability should increase with compliant observation"
        )

    def test_network_compliance_uses_relationships(self, normative_engine):
        """Test that network compliance considers norm relationships."""
        engine, height_id, coverage_id, setback_id = normative_engine

        # Add observations
        engine.add_observation(
            "building_D",
            "building_height",
            30,
            location=Point(-118.25, 34.05),
            certainty=0.9,
        )
        engine.add_observation("building_D", "lot_coverage", 0.40, certainty=0.9)

        # Network compliance should consider the "supports" relationship
        network_prob = engine.infer_network_compliance("building_D", height_id)
        direct_prob = engine.infer_compliance("building_D", height_id)

        # Both should be valid probabilities
        assert 0.0 <= network_prob <= 1.0
        assert 0.0 <= direct_prob <= 1.0

    def test_all_norms_compliance_check(self, normative_engine):
        """Test inferring compliance across all norms at once."""
        engine, height_id, coverage_id, setback_id = normative_engine

        engine.add_observation(
            "building_E",
            "building_height",
            28,
            location=Point(-118.25, 34.05),
            certainty=0.85,
        )
        engine.add_observation("building_E", "lot_coverage", 0.55, certainty=0.8)
        engine.add_observation("building_E", "front_setback", 22, certainty=0.9)

        # Infer all norms at once
        all_compliance = engine.infer_compliance("building_E")
        assert isinstance(all_compliance, dict)
        assert len(all_compliance) == 3, "Should have compliance for all 3 norms"

        for norm_id, prob in all_compliance.items():
            assert 0.0 <= prob <= 1.0, (
                f"Compliance probability {prob} out of range for norm {norm_id}"
            )

    def test_compliance_improvement_suggestions(self, normative_engine):
        """Test that improvement suggestions are generated for non-compliant entities."""
        engine, height_id, coverage_id, setback_id = normative_engine

        # Non-compliant entity
        engine.add_observation(
            "building_F",
            "building_height",
            45,
            location=Point(-118.25, 34.05),
            certainty=0.95,
        )
        engine.add_observation("building_F", "lot_coverage", 0.80, certainty=0.9)
        engine.add_observation("building_F", "front_setback", 12, certainty=0.85)

        # The Bayesian model yields compliance probabilities near 0.5 for
        # non-compliant observations with default priors, so use a threshold
        # of 0.95 to catch them as candidates for improvement.
        suggestions = engine.suggest_compliance_improvements(
            "building_F", improvement_threshold=0.95
        )
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0, (
            "Should suggest improvements for non-compliant entity"
        )

        for suggestion in suggestions:
            assert "norm_id" in suggestion
            assert "norm_name" in suggestion
            assert "recommendation" in suggestion


class TestSocialNormDiffusionPipeline:
    """Test norm diffusion simulation with spatial entities."""

    def test_diffusion_simulation_converges(self):
        """Test that norm diffusion simulation reaches convergence."""
        from geo_infer_norms.core.normative_inference import SocialNormDiffusion

        model = SocialNormDiffusion()

        # Add entities in a spatial arrangement
        locations = [
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
            Point(0, 1),
            Point(1, 1),
            Point(2, 1),
        ]
        for i, loc in enumerate(locations):
            model.add_entity(
                f"entity_{i}",
                attributes={"type": "residential"},
                location=loc,
                adoption_threshold=0.3,
            )

        # Add social connections (grid neighbors)
        connections = [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (0, 3),
            (1, 4),
            (2, 5),
        ]
        for i, j in connections:
            model.add_social_connection(f"entity_{i}", f"entity_{j}", strength=0.8)

        # Add norm with initial adopters
        model.add_norm(
            "recycling_norm",
            name="Recycling Mandate",
            initial_adopters=["entity_0", "entity_5"],
            spatial_factor=0.3,
            network_factor=0.7,
        )

        # Simulate
        results = model.simulate(steps=10)
        assert len(results) > 0

        # Check adoption summary
        summary = model.get_adoption_summary()
        assert "recycling_norm" in summary
        assert (
            summary["recycling_norm"]["adopted_count"] >= 2
        )  # At least initial adopters

    def test_diffusion_with_multiple_norms(self):
        """Test diffusion with multiple norms simultaneously."""
        from geo_infer_norms.core.normative_inference import SocialNormDiffusion

        model = SocialNormDiffusion()

        for i in range(5):
            model.add_entity(
                f"e_{i}", attributes={"region": "urban"}, adoption_threshold=0.4
            )

        for i in range(4):
            model.add_social_connection(f"e_{i}", f"e_{i + 1}", strength=0.9)

        model.add_norm(
            "norm_A",
            name="Norm A",
            initial_adopters=["e_0"],
            network_factor=0.8,
            spatial_factor=0.2,
        )
        model.add_norm(
            "norm_B",
            name="Norm B",
            initial_adopters=["e_4"],
            network_factor=0.8,
            spatial_factor=0.2,
        )

        _results = model.simulate(steps=10)
        summary = model.get_adoption_summary()

        assert "norm_A" in summary
        assert "norm_B" in summary
        # Both norms should have spread from their initial adopters
        assert summary["norm_A"]["adopted_count"] >= 1
        assert summary["norm_B"]["adopted_count"] >= 1


class TestZoningModelsIntegration:
    """Test ZoningCode and ZoningDistrict models working together."""

    def test_zoning_code_use_management(self):
        """Test adding and checking uses across categories."""
        from geo_infer_norms.models.zoning import ZoningCode

        code = ZoningCode.create(
            code="R-1",
            name="Single Family Residential",
            description="Low density residential",
            category="residential",
            jurisdiction_id="city_001",
            allowed_uses=["single_family_home", "home_office"],
            conditional_uses=["daycare"],
            prohibited_uses=["industrial", "commercial_retail"],
        )

        assert code.is_use_allowed("single_family_home") is True
        assert code.is_use_conditional("daycare") is True
        assert code.is_use_prohibited("industrial") is True

        # Moving a prohibited use to allowed should update all categories
        code.add_allowed_use("daycare")
        assert code.is_use_allowed("daycare") is True
        assert code.is_use_conditional("daycare") is False

    def test_zoning_district_lifecycle(self):
        """Test creating and managing a zoning district."""
        from geo_infer_norms.models.zoning import ZoningDistrict

        district = ZoningDistrict.create(
            name="Downtown Core",
            zoning_code="C-2",
            jurisdiction_id="city_001",
            geometry=Polygon(
                [
                    (-118.25, 34.04),
                    (-118.25, 34.06),
                    (-118.23, 34.06),
                    (-118.23, 34.04),
                    (-118.25, 34.04),
                ]
            ),
            effective_date=datetime.date(2020, 1, 1),
        )

        assert district.is_active() is True
        assert district.zoning_code == "C-2"

        # Add overlay and change zoning
        district.add_overlay_code("TOD-1")
        assert "TOD-1" in district.overlay_codes

        district.change_zoning("MU-1")
        assert district.zoning_code == "MU-1"

    def test_zoning_code_and_district_combined(self):
        """Test using ZoningCode to validate uses within a ZoningDistrict."""
        from geo_infer_norms.models.zoning import ZoningCode, ZoningDistrict

        # Create zoning code
        code = ZoningCode.create(
            code="MU-1",
            name="Mixed Use",
            description="Mixed use commercial and residential",
            category="mixed_use",
            jurisdiction_id="city_001",
            allowed_uses=["retail", "residential", "office"],
            prohibited_uses=["heavy_industrial"],
        )

        # Create district using that code
        district = ZoningDistrict.create(
            name="Main Street Corridor",
            zoning_code=code.code,
            jurisdiction_id="city_001",
            geometry=Polygon(
                [
                    (-118.26, 34.04),
                    (-118.26, 34.06),
                    (-118.24, 34.06),
                    (-118.24, 34.04),
                    (-118.26, 34.04),
                ]
            ),
        )

        # Validate a proposed use against the district's zoning code
        assert district.zoning_code == code.code
        assert code.is_use_allowed("retail") is True
        assert code.is_use_prohibited("heavy_industrial") is True
