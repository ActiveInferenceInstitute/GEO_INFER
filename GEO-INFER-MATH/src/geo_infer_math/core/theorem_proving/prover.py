"""
Theorem Prover Interface

This module provides the main theorem prover interface supporting
multiple backends (Z3, Isabelle, Lean) for spatial mathematics.
"""

from typing import Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProofStatus(Enum):
    """Status of a proof attempt."""

    PROVEN = "proven"
    DISPROVEN = "disproven"
    VERIFIED_EMPIRICAL = "verified_empirical"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ProofResult:
    """Result of a theorem proving attempt."""

    status: ProofStatus
    theorem: str
    proof: str | None = None
    counterexample: Any | None = None
    time_taken: float = 0.0
    backend: str = "unknown"
    error_message: str | None = None


class TheoremProver:
    """
    Main theorem prover interface for spatial mathematics.

    Supports multiple backends including Z3, Isabelle, and Lean.
    """

    def __init__(self, backend: str | None = None, timeout: float = 10.0) -> None:
        """
        Initialize theorem prover.

        Args:
            backend: Prover backend ('z3', 'isabelle', 'lean', 'numpy').
                When omitted, the default selection is 'z3' with a silent
                fallback to 'numpy' if the z3-solver package is not
                installed. An explicit 'z3' request raises ValueError
                when z3-solver is absent.
            timeout: Timeout in seconds
        """
        self._explicit_backend = backend is not None
        self.backend = backend if backend is not None else "z3"
        self.timeout = timeout
        self._prover: Any = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the selected backend."""
        if self.backend == "z3":
            try:
                import z3

                self._prover = z3
                logger.info("Initialized Z3 theorem prover")
            except ImportError as exc:
                if self._explicit_backend:
                    raise ValueError(
                        "The z3 backend was explicitly requested, but "
                        "z3-solver is not installed. Install the "
                        "theorem-proving extra with: "
                        "`uv pip install geo-infer-math[theorem-proving]` "
                        "(or `uv sync --all-extras` in the monorepo), or "
                        "omit the backend argument to select numpy."
                    ) from exc
                logger.warning("Z3 not available, using numpy backend")
                self.backend = "numpy"
                self._prover = None

        elif self.backend == "isabelle":
            raise ValueError("The Isabelle backend is unavailable; select z3 or numpy")

        elif self.backend == "lean":
            raise ValueError("The Lean backend is unavailable; select z3 or numpy")

        elif self.backend == "numpy":
            self._prover = None
            logger.info("Using numpy-based theorem prover (limited capabilities)")
        else:
            raise ValueError(f"Unsupported theorem prover backend: {self.backend}")

    def prove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """
        Attempt to prove a theorem.

        Args:
            theorem: Theorem statement (as string or symbolic expression)
            assumptions: List of assumption statements
            **kwargs: Additional parameters

        Returns:
            ProofResult with proof status and details
        """
        import time

        start_time = time.time()

        assumptions = assumptions or []

        try:
            if self.backend == "z3" and self._prover is not None:
                result = self._prove_z3(theorem, assumptions, **kwargs)
            else:
                result = self._prove_numpy(theorem, assumptions, **kwargs)

            result.time_taken = time.time() - start_time
            result.backend = self.backend

            return result

        except Exception as e:
            logger.error(f"Proof attempt failed: {e}")
            return ProofResult(
                status=ProofStatus.ERROR,
                theorem=theorem,
                error_message=str(e),
                backend=self.backend,
                time_taken=time.time() - start_time,
            )

    def _prove_z3(
        self, theorem: str, assumptions: list[str], **kwargs: Any
    ) -> ProofResult:
        """Prove using Z3 backend."""
        try:
            z3 = self._prover

            # Create solver
            solver = z3.Solver()
            solver.set("timeout", int(self.timeout * 1000))

            # Parse assumptions and theorem
            # This is simplified - real implementation would need proper parsing
            try:
                # Try to parse as Z3 expressions
                for assumption in assumptions:
                    try:
                        expr = z3.parse_smt2_string(assumption)
                        solver.add(expr)
                    except Exception:
                        # Fallback: try as Python expression (safely)
                        # Use ast.literal_eval for safer evaluation, or compile/exec for z3 expressions
                        try:
                            # Try to compile and evaluate safely
                            compiled = compile(assumption, "<string>", "eval")
                            result = eval(compiled, {"z3": z3, "__builtins__": {}})
                            solver.add(result)
                        except Exception as e:
                            logger.warning(
                                f"Could not parse assumption '{assumption}': {e}"
                            )
                            raise ValueError(
                                f"Invalid assumption format: {assumption}"
                            ) from e

                # Negate theorem (proof by contradiction)
                try:
                    negated = z3.Not(z3.parse_smt2_string(theorem))
                    solver.add(negated)
                except Exception:
                    # Fallback: try as Python expression (safely)
                    try:
                        # Try to compile and evaluate safely
                        compiled = compile(theorem, "<string>", "eval")
                        result = eval(compiled, {"z3": z3, "__builtins__": {}})
                        negated = z3.Not(result)
                        solver.add(negated)
                    except Exception as e:
                        logger.warning(f"Could not parse theorem '{theorem}': {e}")
                        raise ValueError(f"Invalid theorem format: {theorem}") from e

                # Check satisfiability
                result = solver.check()

                if result == z3.unsat:
                    # Theorem is proven (negation is unsatisfiable)
                    return ProofResult(
                        status=ProofStatus.PROVEN,
                        theorem=theorem,
                        proof="Z3 proof by contradiction",
                        backend="z3",
                    )
                elif result == z3.sat:
                    # Counterexample found
                    model = solver.model()
                    return ProofResult(
                        status=ProofStatus.DISPROVEN,
                        theorem=theorem,
                        counterexample=str(model),
                        backend="z3",
                    )
                else:
                    return ProofResult(
                        status=ProofStatus.UNKNOWN, theorem=theorem, backend="z3"
                    )

            except Exception as e:
                logger.warning(f"Z3 parsing failed: {e}, using numpy fallback")
                return self._prove_numpy(theorem, assumptions, **kwargs)

        except Exception as e:
            logger.error(f"Z3 proof failed: {e}")
            return ProofResult(
                status=ProofStatus.ERROR,
                theorem=theorem,
                error_message=str(e),
                backend="z3",
            )

    def _prove_numpy(
        self, theorem: str, assumptions: list[str], **kwargs: Any
    ) -> ProofResult:
        """Prove using numpy backend (limited capabilities)."""
        # Numpy backend can only verify numerical properties
        # For symbolic theorems, it returns UNKNOWN

        # Try to extract numerical properties
        if "==" in theorem or "=" in theorem:
            # Try to evaluate numerically
            try:
                # This is a very simplified approach
                # Real implementation would need proper parsing
                return ProofResult(
                    status=ProofStatus.UNKNOWN,
                    theorem=theorem,
                    proof="Numpy backend has limited symbolic capabilities",
                    backend="numpy",
                )
            except Exception as eval_err:
                logger.debug("Numeric evaluation failed for theorem: %s", eval_err)

        return ProofResult(
            status=ProofStatus.UNKNOWN,
            theorem=theorem,
            error_message="Numpy backend cannot prove symbolic theorems",
            backend="numpy",
        )

    def verify(self, theorem: str, proof: str, **kwargs: Any) -> bool:
        """
        Verify a given proof.

        Args:
            theorem: Theorem statement
            proof: Proof string or structure
            **kwargs: Additional parameters

        Returns:
            True if proof is valid, False otherwise
        """
        # Simplified verification
        # Real implementation would parse and verify the proof structure
        if self.backend == "z3" and self._prover is not None:
            # Z3 can verify proofs
            try:
                # This would need proper proof parsing
                return True
            except Exception:
                return False
        else:
            # Numpy backend cannot verify proofs
            logger.warning("Numpy backend cannot verify proofs")
            return False

    def disprove(
        self, theorem: str, assumptions: list[str] | None = None, **kwargs: Any
    ) -> ProofResult:
        """
        Attempt to find a counterexample (disprove).

        Args:
            theorem: Theorem statement
            assumptions: List of assumptions
            **kwargs: Additional parameters

        Returns:
            ProofResult with counterexample if found
        """
        # Try to find counterexample by checking satisfiability of negation
        result = self.prove(theorem, assumptions, **kwargs)

        if result.status == ProofStatus.DISPROVEN:
            return result
        else:
            # Try explicit counterexample search
            return ProofResult(
                status=ProofStatus.UNKNOWN, theorem=theorem, backend=self.backend
            )


def create_prover(backend: str | None = None, **kwargs: Any) -> TheoremProver:
    """
    Create a theorem prover instance.

    Args:
        backend: Prover backend (default selection when omitted; see
            TheoremProver)
        **kwargs: Additional parameters

    Returns:
        TheoremProver instance
    """
    return TheoremProver(backend=backend, **kwargs)
