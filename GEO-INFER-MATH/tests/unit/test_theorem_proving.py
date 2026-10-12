"""
Tests for GEO-INFER-MATH theorem proving module.

Tests cover: TheoremProver, ProofResult, TheoremDatabase, ProofVerifier,
and proof strategy classes.
"""

import sys
import types

import pytest


from geo_infer_math.core.theorem_proving.prover import (
    ProofResult,
    ProofStatus,
    TheoremProver,
    create_prover,
)
from geo_infer_math.core.theorem_proving.spatial_theorems import (
    SpatialTheorem,
    GeometricTheorem,
    StatisticalTheorem,
    TheoremDatabase,
    TheoremType,
)
from geo_infer_math.core.theorem_proving.proof_verification import (
    ProofVerifier,
)
from geo_infer_math.core.theorem_proving.proof_strategies import (
    GeometricProofStrategy,
    StatisticalProofStrategy,
    ProofStrategySelector,
)


class TestProofResult:
    """Tests for ProofResult dataclass."""

    def test_proof_result_creation(self):
        result = ProofResult(
            status=ProofStatus.PROVEN,
            theorem="x + y = y + x",
            proof="By commutativity of addition.",
            backend="sympy",
        )
        assert result.status == ProofStatus.PROVEN
        assert result.theorem == "x + y = y + x"
        assert result.proof is not None

    def test_proof_result_unknown_status(self):
        result = ProofResult(
            status=ProofStatus.UNKNOWN,
            theorem="P = NP",
            error_message="Cannot determine",
            backend="numpy",
        )
        assert result.status == ProofStatus.UNKNOWN
        assert result.error_message is not None

    def test_proof_status_enum_values(self):
        assert ProofStatus.PROVEN.value == "proven"
        assert ProofStatus.DISPROVEN.value == "disproven"
        assert ProofStatus.UNKNOWN.value == "unknown"
        assert ProofStatus.TIMEOUT.value == "timeout"
        assert ProofStatus.ERROR.value == "error"


class TestTheoremProver:
    """Tests for TheoremProver."""

    def test_create_prover(self):
        prover = create_prover(backend="numpy")
        assert prover is not None

    def test_prover_with_numpy_backend(self):
        prover = create_prover(backend="numpy")
        result = prover.prove("2 + 2 == 4")
        assert isinstance(result, ProofResult)
        assert result.status in [ProofStatus.PROVEN, ProofStatus.UNKNOWN]

    def test_prover_backend_attribute(self):
        prover = create_prover(backend="numpy")
        assert hasattr(prover, "backend")
        assert prover.backend == "numpy"

    def test_explicit_z3_request_without_z3_raises(self, monkeypatch):
        """An explicit backend="z3" request with z3-solver absent raises
        a ValueError naming the theorem-proving extra; it never silently
        swaps to numpy."""
        monkeypatch.setitem(sys.modules, "z3", None)
        with pytest.raises(ValueError, match="theorem-proving"):
            TheoremProver(backend="z3")

    def test_default_backend_selection_still_downgrades(self, monkeypatch):
        """The default backend selection still silently falls back to
        numpy when z3-solver is absent (TheoremProver() and the
        create_prover factory)."""
        monkeypatch.setitem(sys.modules, "z3", None)
        for factory in (TheoremProver, create_prover):
            prover = factory()
            assert prover.backend == "numpy"
            assert prover._prover is None

    def test_explicit_z3_request_with_z3_present_selects_z3(self, monkeypatch):
        """An explicit z3 request with z3-solver importable selects z3."""
        monkeypatch.setitem(sys.modules, "z3", types.ModuleType("z3"))
        prover = TheoremProver(backend="z3")
        assert prover.backend == "z3"
        assert prover._prover is not None


class TestTheoremDatabase:
    """Tests for TheoremDatabase."""

    def test_database_creation_with_standard_theorems(self):
        db = TheoremDatabase()
        # Standard theorems are loaded automatically
        triangle = db.get_theorem("Triangle Inequality")
        assert triangle is not None

    def test_add_and_retrieve_theorem(self):
        db = TheoremDatabase()
        theorem = GeometricTheorem(
            name="test_custom_theorem",
            statement="For all points p, distance(p, p) = 0",
        )
        db.add_theorem(theorem)
        retrieved = db.get_theorem("test_custom_theorem")
        assert retrieved is not None
        assert retrieved.name == "test_custom_theorem"

    def test_get_nonexistent_theorem(self):
        db = TheoremDatabase()
        result = db.get_theorem("nonexistent_theorem_xyz")
        assert result is None

    def test_search_theorems_returns_standard(self):
        db = TheoremDatabase()
        theorems = db.search_theorems()
        assert len(theorems) >= 3  # At least the standard theorems
        names = [t.name for t in theorems]
        assert "Pythagorean Theorem" in names

    def test_search_theorems_by_type(self):
        db = TheoremDatabase()
        geometric = db.search_theorems(theorem_type=TheoremType.GEOMETRIC)
        assert len(geometric) >= 2
        for t in geometric:
            assert t.theorem_type == TheoremType.GEOMETRIC

    def test_search_theorems_by_keyword(self):
        db = TheoremDatabase()
        results = db.search_theorems(keyword="triangle")
        assert len(results) >= 1


class TestSpatialTheorem:
    """Tests for SpatialTheorem and its subclasses."""

    def test_spatial_theorem_creation(self):
        theorem = SpatialTheorem(
            name="test",
            statement="test statement",
            theorem_type=TheoremType.GEOMETRIC,
        )
        assert theorem.name == "test"
        assert theorem.statement == "test statement"
        assert theorem.theorem_type == TheoremType.GEOMETRIC

    def test_geometric_theorem_auto_type(self):
        theorem = GeometricTheorem(
            name="pythagorean",
            statement="a^2 + b^2 = c^2",
        )
        assert theorem.theorem_type == TheoremType.GEOMETRIC

    def test_statistical_theorem_auto_type(self):
        theorem = StatisticalTheorem(
            name="clt",
            statement="averages converge to normal",
        )
        assert theorem.theorem_type == TheoremType.STATISTICAL

    def test_theorem_default_lists(self):
        theorem = SpatialTheorem(
            name="t",
            statement="s",
            theorem_type=TheoremType.ALGEBRAIC,
        )
        assert theorem.assumptions == []
        assert theorem.corollaries == []
        assert theorem.applications == []


class TestProofVerifier:
    """Tests for proof verification."""

    def test_verifier_creation(self):
        verifier = ProofVerifier(backend="numpy")
        assert verifier is not None
        assert verifier.backend == "numpy"


class TestProofStrategies:
    """Tests for proof strategy classes."""

    def test_geometric_strategy_can_apply_geometric(self):
        strategy = GeometricProofStrategy()
        assert strategy.can_apply("triangle inequality holds") is True

    def test_geometric_strategy_cannot_apply_statistical(self):
        strategy = GeometricProofStrategy()
        assert strategy.can_apply("mean converges") is False

    def test_statistical_strategy_can_apply(self):
        strategy = StatisticalProofStrategy()
        assert strategy.can_apply("expectation is linear") is True

    def test_statistical_strategy_cannot_apply_geometric(self):
        strategy = StatisticalProofStrategy()
        assert strategy.can_apply("triangle side lengths") is False

    def test_strategy_selector_selects_geometric(self):
        selector = ProofStrategySelector()
        strategy = selector.select_strategy("triangle inequality")
        assert isinstance(strategy, GeometricProofStrategy)

    def test_strategy_selector_selects_statistical(self):
        selector = ProofStrategySelector()
        strategy = selector.select_strategy("expectation of random variable")
        assert isinstance(strategy, StatisticalProofStrategy)

    def test_triangle_inequality_verified_empirical_never_proven(self):
        """Numeric confirmation reports VERIFIED_EMPIRICAL, never PROVEN."""
        strategy = GeometricProofStrategy()
        result = strategy._try_triangle_inequality(
            "triangle inequality holds in the plane", []
        )
        assert result.status == ProofStatus.VERIFIED_EMPIRICAL
        assert result.status != ProofStatus.PROVEN
        assert result.proof is not None

    def test_expectation_linearity_verified_empirical(self):
        """Linearity of expectation is confirmed numerically, not claimed."""
        strategy = StatisticalProofStrategy()
        result = strategy._try_expectation_properties(
            "expectation is a linear operator", []
        )
        assert result.status == ProofStatus.VERIFIED_EMPIRICAL
        assert result.status != ProofStatus.PROVEN

    def test_verified_empirical_enum_member(self):
        assert ProofStatus.VERIFIED_EMPIRICAL.value == "verified_empirical"
