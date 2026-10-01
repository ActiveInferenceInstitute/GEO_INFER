"""
Automated Proof Strategies

This module provides automated proof strategies for common
spatial mathematics proof patterns.
"""

from typing import Any
import logging
from abc import ABC, abstractmethod

import numpy as np

from geo_infer_math.utils.rng import resolve_rng

from geo_infer_math.core.theorem_proving.prover import (
    TheoremProver,
    ProofResult,
    ProofStatus,
)

logger = logging.getLogger(__name__)


class ProofStrategy(ABC):
    """
    Abstract base class for proof strategies.

    Proof strategies provide automated approaches to proving
    theorems in spatial mathematics.
    """

    def __init__(self, prover: TheoremProver | None = None):
        """
        Initialize proof strategy.

        Args:
            prover: Optional theorem prover instance
        """
        self.prover = prover or TheoremProver()

    @abstractmethod
    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """
        Attempt to prove a theorem using this strategy.

        Args:
            theorem: Theorem statement
            assumptions: List of assumptions
            **kwargs: Additional parameters

        Returns:
            ProofResult
        """
        raise RuntimeError(
            "ProofStrategy.prove requires a concrete proof strategy implementation"
        )

    def can_apply(self, theorem: str) -> bool:
        """
        Check if this strategy can be applied to a theorem.

        Args:
            theorem: Theorem statement

        Returns:
            True if strategy can be applied
        """
        return True


class GeometricProofStrategy(ProofStrategy):
    """
    Proof strategy for geometric theorems.

    Uses geometric reasoning and properties.
    """

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """Prove geometric theorem."""
        assumptions = assumptions or []

        # Add geometric assumptions
        geometric_assumptions = [
            "Points are in Euclidean space",
            "Distances satisfy triangle inequality",
        ]
        assumptions.extend(geometric_assumptions)

        # Try to prove using prover
        result = self.prover.prove(theorem, assumptions, **kwargs)

        if result.status == ProofStatus.UNKNOWN:
            # Try geometric-specific reasoning
            if "triangle" in theorem.lower() or "distance" in theorem.lower():
                # Try triangle inequality reasoning
                result = self._try_triangle_inequality(theorem, assumptions)

        return result

    def _try_triangle_inequality(
        self, theorem: str, assumptions: list[str]
    ) -> ProofResult:
        """Verify the triangle inequality empirically on sampled point triples.

        Draws random triples of Euclidean points and checks
        d(a, c) <= d(a, b) + d(b, c) for each. A complete numeric
        confirmation reports ProofStatus.VERIFIED_EMPIRICAL (never PROVEN —
        no symbolic proof was performed); a violating triple is reported as
        DISPROVEN with the counterexample.
        """
        if "triangle" not in theorem.lower() or "inequality" not in theorem.lower():
            return ProofResult(
                status=ProofStatus.UNKNOWN, theorem=theorem, backend=self.prover.backend
            )

        rng = resolve_rng(0)
        n_samples = 1000
        points = rng.uniform(0.0, 10.0, size=(n_samples, 3, 2))
        a, b, c = points[:, 0, :], points[:, 1, :], points[:, 2, :]
        d_ab = np.linalg.norm(a - b, axis=1)
        d_bc = np.linalg.norm(b - c, axis=1)
        d_ac = np.linalg.norm(a - c, axis=1)

        violations = d_ac > d_ab + d_bc + 1e-9
        if np.any(violations):
            first = int(np.argmax(violations))
            counterexample = {
                "a": a[first].tolist(),
                "b": b[first].tolist(),
                "c": c[first].tolist(),
            }
            return ProofResult(
                status=ProofStatus.DISPROVEN,
                theorem=theorem,
                counterexample=counterexample,
                backend=self.prover.backend,
            )

        return ProofResult(
            status=ProofStatus.VERIFIED_EMPIRICAL,
            theorem=theorem,
            proof=(
                "Empirical verification: the triangle inequality "
                "d(a,c) <= d(a,b) + d(b,c) held for all "
                f"{n_samples} randomly sampled Euclidean point triples "
                "(numeric check only; no symbolic proof performed)"
            ),
            backend=self.prover.backend,
        )

    def can_apply(self, theorem: str) -> bool:
        """Check if geometric strategy applies."""
        geometric_keywords = [
            "triangle",
            "distance",
            "angle",
            "polygon",
            "circle",
            "line",
            "point",
            "geometric",
        ]
        return any(keyword in theorem.lower() for keyword in geometric_keywords)


class StatisticalProofStrategy(ProofStrategy):
    """
    Proof strategy for statistical theorems.

    Uses statistical reasoning and properties.
    """

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """Prove statistical theorem."""
        assumptions = assumptions or []

        # Add statistical assumptions
        statistical_assumptions = [
            "Samples are independent",
            "Distributions have finite variance",
        ]
        assumptions.extend(statistical_assumptions)

        # Try to prove using prover
        result = self.prover.prove(theorem, assumptions, **kwargs)

        if result.status == ProofStatus.UNKNOWN:
            # Try statistical-specific reasoning
            if "expectation" in theorem.lower() or "variance" in theorem.lower():
                result = self._try_expectation_properties(theorem, assumptions)

        return result

    def _try_expectation_properties(
        self, theorem: str, assumptions: list[str]
    ) -> ProofResult:
        """Verify linearity of expectation empirically on sampled variables.

        Draws large random samples of X and Y and checks
        E[aX + bY] = aE[X] + bE[Y] numerically. A confirmed check reports
        ProofStatus.VERIFIED_EMPIRICAL (numeric check only, never PROVEN).
        """
        if "expectation" not in theorem.lower() or "linear" not in theorem.lower():
            return ProofResult(
                status=ProofStatus.UNKNOWN, theorem=theorem, backend=self.prover.backend
            )

        rng = resolve_rng(0)
        n_samples = 100_000
        x_samples = rng.normal(0.0, 1.0, size=n_samples)
        y_samples = rng.normal(0.0, 2.0, size=n_samples)
        a, b = 1.3, -0.7

        lhs = float(np.mean(a * x_samples + b * y_samples))
        rhs = a * float(np.mean(x_samples)) + b * float(np.mean(y_samples))
        tolerance = 10.0 / np.sqrt(n_samples)

        if abs(lhs - rhs) <= tolerance:
            return ProofResult(
                status=ProofStatus.VERIFIED_EMPIRICAL,
                theorem=theorem,
                proof=(
                    "Empirical verification: E[aX + bY] = aE[X] + bE[Y] held "
                    f"numerically on {n_samples} Gaussian samples "
                    f"(|E[aX+bY] - (aE[X]+bE[Y])| = {abs(lhs - rhs):.2e} "
                    f"<= tolerance {tolerance:.2e}; numeric check only)"
                ),
                backend=self.prover.backend,
            )

        return ProofResult(
            status=ProofStatus.UNKNOWN,
            theorem=theorem,
            proof="Empirical check did not converge within tolerance",
            backend=self.prover.backend,
        )

    def can_apply(self, theorem: str) -> bool:
        """Check if statistical strategy applies."""
        statistical_keywords = [
            "expectation",
            "variance",
            "distribution",
            "probability",
            "statistical",
            "random",
        ]
        return any(keyword in theorem.lower() for keyword in statistical_keywords)


class DirectProofStrategy(ProofStrategy):
    """
    Direct proof strategy.

    Attempts direct proof without special reasoning.
    """

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """Attempt direct proof."""
        return self.prover.prove(theorem, assumptions, **kwargs)


class ContradictionProofStrategy(ProofStrategy):
    """
    Proof by contradiction strategy.

    Attempts to prove by showing negation leads to contradiction.
    """

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """Prove by contradiction."""
        assumptions = assumptions or []

        # Negate theorem
        negated_theorem = f"NOT ({theorem})"

        # Try to find contradiction
        result = self.prover.prove(negated_theorem, assumptions, **kwargs)

        if result.status == ProofStatus.DISPROVEN:
            # Contradiction found, original theorem is proven
            return ProofResult(
                status=ProofStatus.PROVEN,
                theorem=theorem,
                proof=f"Proof by contradiction: {result.counterexample} leads to contradiction",
                backend=self.prover.backend,
            )

        return result


class InductionProofStrategy(ProofStrategy):
    """
    Proof by induction strategy.

    Attempts to prove by mathematical induction.
    """

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """Prove by induction."""
        # Induction requires base case and inductive step
        # Induction requires parsing the theorem into base case and inductive
        # step, which needs a formal-language front-end this backend lacks

        assumptions = assumptions or []

        # Induction requires parsing the theorem string to extract a base case
        # and inductive step; this needs an AST or formal-language front-end

        return ProofResult(
            status=ProofStatus.UNKNOWN,
            theorem=theorem,
            proof="Induction proof strategy requires pattern matching",
            backend=self.prover.backend,
        )


class ProofStrategySelector:
    """
    Selects appropriate proof strategy for a theorem.
    """

    def __init__(self) -> None:
        """Initialize strategy selector."""
        self.strategies = [
            GeometricProofStrategy(),
            StatisticalProofStrategy(),
            ContradictionProofStrategy(),
            InductionProofStrategy(),
            DirectProofStrategy(),
        ]

    def select_strategy(
        self, theorem: str, theorem_type: str | None = None
    ) -> ProofStrategy:
        """
        Select best strategy for a theorem.

        Args:
            theorem: Theorem statement
            theorem_type: Optional theorem type hint

        Returns:
            Selected proof strategy
        """
        # Try strategies in order of specificity
        for strategy in self.strategies:
            if strategy.can_apply(theorem):
                return strategy

        # Default to direct proof
        return DirectProofStrategy()

    def try_all_strategies(
        self, theorem: str, assumptions: list[str] | None = None
    ) -> list[ProofResult]:
        """
        Try all applicable strategies.

        Args:
            theorem: Theorem statement
            assumptions: List of assumptions

        Returns:
            List of proof results from each strategy
        """
        results = []

        for strategy in self.strategies:
            if strategy.can_apply(theorem):
                result = strategy.prove(theorem, assumptions)
                results.append(result)

        return results
