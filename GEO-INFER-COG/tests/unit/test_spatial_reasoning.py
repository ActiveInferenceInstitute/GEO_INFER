"""
Unit tests for spatial reasoning: SpatialRelation, ReasoningStep, SpatialReasoningEngine.
"""

from geo_infer_cog.core.spatial_reasoning import (
    SpatialRelation,
    ReasoningStep,
    SpatialReasoningEngine,
)
import logging
import time


from geo_infer_cog.core import spatial_reasoning as spatial_reasoning_module


class TestSpatialRelation:
    """Test SpatialRelation data class."""

    def test_defaults(self) -> None:
        rel = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        assert rel.confidence == 1.0
        assert rel.reasoning_path == []

    def test_is_consistent_with_same_pair(self) -> None:
        rel1 = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        rel2 = SpatialRelation(
            source_region="B", target_region="A", relation_type="disconnected"
        )
        assert rel1.is_consistent_with(rel2) is True

    def test_inconsistent_relations(self) -> None:
        rel1 = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        rel2 = SpatialRelation(
            source_region="B", target_region="A", relation_type="equal"
        )
        assert rel1.is_consistent_with(rel2) is False

    def test_no_conflict_unrelated_pair(self) -> None:
        rel1 = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        rel2 = SpatialRelation(
            source_region="C", target_region="D", relation_type="equal"
        )
        assert rel1.is_consistent_with(rel2) is True


class TestReasoningStep:
    """Test ReasoningStep class."""

    def test_step_fields(self) -> None:
        step = ReasoningStep(
            step_id="s1",
            operation="compose",
            input_premises=["p1", "p2"],
            conclusion="c1",
            confidence=0.9,
            explanation="Test explanation",
        )
        assert step.step_id == "s1"
        assert step.operation == "compose"
        assert len(step.input_premises) == 2
        assert step.confidence == 0.9


class TestSpatialReasoningEngine:
    """Test SpatialReasoningEngine class."""

    def test_init_defaults(self) -> None:
        engine = SpatialReasoningEngine()
        assert engine.reasoning_type == "qualitative_spatial"
        assert engine.uncertainty_method == "probabilistic"

    def test_knowledge_base_initialized(self) -> None:
        engine = SpatialReasoningEngine()
        kb = engine.spatial_knowledge_base
        assert "topological_relations" in kb
        assert "directional_relations" in kb
        assert "distance_relations" in kb
        assert "disconnected" in kb["topological_relations"]
        assert "north" in kb["directional_relations"]

    def test_determine_topological_equal(self) -> None:
        engine = SpatialReasoningEngine()
        bbox = (0.0, 0.0, 1.0, 1.0)
        result = engine._determine_topological_relation(bbox, bbox)
        assert result == "equal"

    def test_determine_topological_disconnected(self) -> None:
        engine = SpatialReasoningEngine()
        bbox1 = (0.0, 0.0, 1.0, 1.0)
        bbox2 = (5.0, 5.0, 6.0, 6.0)
        result = engine._determine_topological_relation(bbox1, bbox2)
        assert result == "disconnected"

    def test_determine_topological_overlapping(self) -> None:
        engine = SpatialReasoningEngine()
        bbox1 = (0.0, 0.0, 2.0, 2.0)
        bbox2 = (1.0, 1.0, 3.0, 3.0)
        result = engine._determine_topological_relation(bbox1, bbox2)
        assert result == "partially_overlapping"

    def test_determine_topological_nontangential_proper_part(self) -> None:
        engine = SpatialReasoningEngine()
        inner = (0.5, 0.5, 1.5, 1.5)
        outer = (0.0, 0.0, 2.0, 2.0)
        result = engine._determine_topological_relation(inner, outer)
        assert result == "non_tangential_proper_part"

    def test_compose_relations_chain(self) -> None:
        engine = SpatialReasoningEngine()
        rel1 = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        rel2 = SpatialRelation(
            source_region="B", target_region="C", relation_type="disconnected"
        )
        composed = engine._compose_relations(rel1, rel2)
        assert composed is not None
        assert composed.source_region == "A"
        assert composed.target_region == "C"
        assert composed.relation_type == "disconnected"

    def test_compose_relations_no_chain_returns_none(self) -> None:
        engine = SpatialReasoningEngine()
        rel1 = SpatialRelation(
            source_region="A", target_region="B", relation_type="disconnected"
        )
        rel2 = SpatialRelation(
            source_region="C", target_region="D", relation_type="equal"
        )
        assert engine._compose_relations(rel1, rel2) is None

    def test_get_alternative_relations(self) -> None:
        engine = SpatialReasoningEngine()
        alts = engine._get_alternative_relations("disconnected")
        assert "externally_connected" in alts

    def test_get_status(self) -> None:
        engine = SpatialReasoningEngine()
        status = engine.get_status()
        assert status["engine_type"] == "spatial_reasoning"
        assert status["reasoning_type"] == "qualitative_spatial"
        assert status["status"] == "active"

    def test_calculate_overall_confidence_empty(self) -> None:
        engine = SpatialReasoningEngine()
        assert engine._calculate_overall_confidence([]) == 0.0

    def test_validate_reasoning_chain_no_issues(self) -> None:
        engine = SpatialReasoningEngine()
        conclusions = [
            SpatialRelation(
                source_region="A",
                target_region="B",
                relation_type="disconnected",
                confidence=0.9,
            ),
        ]
        result = engine._validate_reasoning_chain(conclusions, "chain_1")
        assert result["valid"] is True
        assert abs(result["confidence"] - 0.9) < 1e-6

    def test_region_id_deterministic_across_processes(self) -> None:
        """Regression: region ids must be stable, not salted per process."""
        engine = SpatialReasoningEngine()
        geom = {"type": "Point", "coordinates": [1.0, 2.0]}
        first = engine._region_id(geom)
        second = SpatialReasoningEngine()._region_id(geom)
        assert first == second
        assert first.startswith("region_")
        # Different geometry content yields a different region id.
        assert first != engine._region_id({"type": "Point", "coordinates": [2.0, 1.0]})

    def test_inferred_relation_uses_deterministic_regions(self) -> None:
        engine = SpatialReasoningEngine()
        geom1 = {"type": "Point", "coordinates": [0.0, 0.0]}
        geom2 = {"type": "Point", "coordinates": [1.0, 1.0]}
        relation = engine._infer_spatial_relation(geom1, geom2)
        assert relation is not None
        assert relation.source_region == SpatialReasoningEngine()._region_id(geom1)
        assert relation.target_region == SpatialReasoningEngine()._region_id(geom2)


class TestConstraintSolverBounded:
    """Regression tests for GS-126: constraint search must stay bounded."""

    @staticmethod
    def _chain_premises() -> list:
        return [
            SpatialRelation("A", "B", "disconnected"),
            SpatialRelation("B", "C", "disconnected"),
            SpatialRelation("C", "D", "disconnected"),
            SpatialRelation("D", "E", "disconnected"),
        ]

    def test_five_region_constraint_reasoning_completes_bounded(self) -> None:
        engine = SpatialReasoningEngine(reasoning_type="constraint_based")
        premises = self._chain_premises()

        start = time.perf_counter()
        conclusions = engine._constraint_based_reasoning(premises, "chain_test")
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0
        # Conclusions are deduplicated novel relations for the six
        # premise-free pairs; none may restate a premise pair.
        premise_pairs = {(p.source_region, p.target_region) for p in premises}
        assert {
            (c.source_region, c.target_region, c.relation_type) for c in conclusions
        } == {
            ("A", "C", relation)
            for relation in engine.spatial_knowledge_base["topological_relations"]
        }
        for conclusion in conclusions:
            assert (conclusion.source_region, conclusion.target_region) not in (
                premise_pairs
            )
            assert conclusion.reasoning_path == ["constraint_satisfaction"]

    def test_solver_pins_premise_relations(self) -> None:
        engine = SpatialReasoningEngine(reasoning_type="constraint_based")
        variables, domains, constraints = engine._setup_spatial_constraints(
            self._chain_premises()
        )
        solutions = engine._solve_spatial_constraints(variables, domains, constraints)

        assert len(solutions) == 6**6, "all satisfying assignments must be retained"
        for solution in solutions:
            assert solution["A_B"] == "disconnected"
            assert solution["B_C"] == "disconnected"
            assert solution["C_D"] == "disconnected"
            assert solution["D_E"] == "disconnected"

    def test_disconnected_components_are_not_enumerated(self) -> None:
        engine = SpatialReasoningEngine(reasoning_type="constraint_based")
        premises = [
            SpatialRelation("A", "B", "disconnected"),
            SpatialRelation("C", "D", "disconnected"),
        ]
        _, domains, _ = engine._setup_spatial_constraints(premises)

        # Two components: no cross-component pair may get a domain variable.
        assert sorted(domains) == ["A_B", "C_D"]

    def test_constraint_search_degrades_with_warning(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            spatial_reasoning_module,
            "_MAX_EVALUATED_CONSTRAINT_ASSIGNMENTS",
            10,
        )
        engine = SpatialReasoningEngine(reasoning_type="constraint_based")
        premises = self._chain_premises()

        with caplog.at_level(
            logging.WARNING, logger="geo_infer_cog.core.spatial_reasoning"
        ):
            solutions = engine._solve_spatial_constraints(
                *engine._setup_spatial_constraints(premises)
            )

        assert any("degraded" in record.message for record in caplog.records)
        # Cap triggers early: only a few solutions are collected, far fewer
        # than the full 6**6 Cartesian product of the six free pairs.
        assert len(solutions) < 6**6

    def test_conflicting_and_out_of_domain_fixed_relations_are_unsatisfiable(self):
        engine = SpatialReasoningEngine(reasoning_type="constraint_based")
        for second in ("externally_connected", "unknown"):
            constraints = [
                {"type": "fixed_relation", "variable": "A_B", "value": value}
                for value in ("disconnected", second)
            ]
            assert (
                engine._solve_spatial_constraints(
                    ["A", "B"],
                    {"A_B": ["disconnected", "externally_connected"]},
                    constraints,
                )
                == []
            )
