"""Finite real matrix, Markov-current and discrete graph Hodge operations.

These numerical constructs provide explicit arrays and remainder witnesses;
they do not assert continuum boundary conditions or formal theorem acceptance.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "skew_part",
    "probability_current",
    "circulation_part",
    "current_divergence",
    "satisfies_detailed_balance",
    "GraphHodgeDecomposition",
    "graph_hodge_decomposition",
]


def _real_array(value: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}-dimensional array")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _tolerance(atol: float) -> float:
    if isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol):
        raise ValueError("atol must be a finite nonnegative number")
    if not isinstance(atol, (int, float, np.integer, np.floating)):
        raise ValueError("atol must be a finite nonnegative number")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    return float(atol)


def _square(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = _real_array(value, name, 2)
    if array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return array


def skew_part(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return (A - A.T)/2 for a finite real square matrix, without mutation."""
    array = _square(matrix, "matrix")
    # Scaling first avoids overflow when subtracting opposite large entries.
    return array * 0.5 - array.T * 0.5


def probability_current(
    kernel: ArrayLike,
    probability: ArrayLike,
    *,
    kind: str = "transition",
    atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Return J[i,j] = pi[i] K[i,j] - pi[j] K[j,i].

    ``transition`` requires nonnegative row-stochastic K; ``rate`` requires
    nonnegative off-diagonals, nonpositive diagonals and zero row sums.
    pi is a normalized nonnegative vector and need not be stationary.
    No input is normalized, clipped or reordered. Tolerance is absolute.
    """
    tolerance = _tolerance(atol)
    array = _square(kernel, "kernel")
    pi = _real_array(probability, "probability", 1)
    if (
        pi.shape != (len(array),)
        or np.any(pi < 0)
        or not np.isclose(pi.sum(), 1, rtol=0, atol=tolerance)
    ):
        raise ValueError(
            "probability must be nonnegative, normalized and match kernel size"
        )
    if kind == "transition":
        valid = np.all(array >= 0) and np.allclose(
            array.sum(axis=1), 1, rtol=0, atol=tolerance
        )
    elif kind == "rate":
        off_diagonal = array.copy()
        np.fill_diagonal(off_diagonal, 0)
        valid = (
            np.all(off_diagonal >= 0)
            and np.all(np.diag(array) <= 0)
            and np.allclose(array.sum(axis=1), 0, rtol=0, atol=tolerance)
        )
    else:
        raise ValueError("kind must be 'transition' or 'rate'")
    if not valid:
        raise ValueError(f"kernel does not satisfy the {kind} contract")
    with np.errstate(over="ignore", invalid="ignore"):
        flow = pi[:, None] * array
        current = flow - flow.T
    if not np.isfinite(current).all():
        raise ValueError("probability current exceeds finite floating-point range")
    return current


def circulation_part(
    kernel: ArrayLike,
    probability: ArrayLike,
    *,
    kind: str = "transition",
    atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Return the skew part of pi-weighted flow, exactly half the current."""
    return probability_current(kernel, probability, kind=kind, atol=atol) * 0.5


def current_divergence(
    current: ArrayLike, *, atol: float = 1e-12
) -> NDArray[np.float64]:
    """Return incoming node divergence div[j] = sum_i J[i,j].

    A stationary Markov current has zero divergence. This function reports
    the actual residual, rather than assuming stationarity from normalization.
    """
    tolerance = _tolerance(atol)
    array = _square(current, "current")
    if not np.allclose(array * 0.5, -array.T * 0.5, rtol=0, atol=tolerance * 0.5):
        raise ValueError("current must be skew-symmetric")
    with np.errstate(over="ignore", invalid="ignore"):
        divergence = array.sum(axis=0)
    if not np.isfinite(divergence).all():
        raise ValueError("divergence exceeds finite floating-point range")
    return divergence


def satisfies_detailed_balance(
    kernel: ArrayLike,
    probability: ArrayLike,
    *,
    kind: str = "transition",
    atol: float = 1e-12,
) -> bool:
    """Whether every pi-weighted pair has zero current within absolute atol."""
    tolerance = _tolerance(atol)
    current = probability_current(kernel, probability, kind=kind, atol=tolerance)
    return bool(np.all(np.abs(current) <= tolerance))


@dataclass(frozen=True)
class GraphHodgeDecomposition:
    """Euclidean projections and explicit harmonic remainder in edge order.

    ``solenoidal + harmonic`` is the divergence-free part. ``solenoidal``
    is the face-boundary projection; without faces it is zero and all cycle
    flow is reported as harmonic. Potentials use the minimum-norm gauge.
    """

    gradient: NDArray[np.float64]
    solenoidal: NDArray[np.float64]
    harmonic: NDArray[np.float64]
    node_potential: NDArray[np.float64]
    face_potential: NDArray[np.float64]


def graph_hodge_decomposition(
    incidence: ArrayLike,
    flow: ArrayLike,
    face_boundary: ArrayLike | None = None,
    *,
    atol: float = 1e-12,
) -> GraphHodgeDecomposition:
    """Split real edge flow into gradient, face curl and harmonic remainder.

    B (nodes x edges) has exactly one -1 source and +1 target per column.
    C (edges x faces), when supplied, must satisfy B C = 0 after each nonzero
    column is scaled to unit maximum magnitude for the tolerance check.
    Projections use
    least squares onto range(B.T) and range(C); the remainder lies in ker(B)
    and ker(C.T), up to numerical error. Disconnected graphs are supported.
    This is an unweighted finite graph/complex contract, not a PDE solver.
    """
    tolerance = _tolerance(atol)
    boundary = _real_array(incidence, "incidence", 2)
    edge_flow = _real_array(flow, "flow", 1)
    nodes, edges = boundary.shape
    if nodes == 0 or edge_flow.shape != (edges,):
        raise ValueError(
            "flow must match incidence edge count and nodes must be nonempty"
        )
    if not (
        np.isin(boundary, [-1, 0, 1]).all()
        and np.all((boundary == -1).sum(axis=0) == 1)
        and np.all((boundary == 1).sum(axis=0) == 1)
    ):
        raise ValueError("each incidence column must have one source and one target")
    faces = (
        np.empty((edges, 0))
        if face_boundary is None
        else _real_array(face_boundary, "face_boundary", 2)
    )
    if faces.shape[0] != edges:
        raise ValueError("face_boundary must match incidence edge count")
    # A tiny invalid column still spans an invalid direction: least squares
    # cancels its magnitude. Validate directions rather than raw magnitudes.
    scales = np.max(np.abs(faces), axis=0, initial=0)
    normalized_faces = np.divide(
        faces, scales, out=np.zeros_like(faces), where=scales != 0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        chain = boundary @ normalized_faces
    if not np.isfinite(chain).all() or not np.allclose(
        chain, 0, rtol=0, atol=tolerance
    ):
        raise ValueError("face_boundary must satisfy incidence @ face_boundary = 0")
    node_potential = np.linalg.lstsq(boundary.T, edge_flow, rcond=None)[0]
    gradient = boundary.T @ node_potential
    face_potential = np.linalg.lstsq(faces, edge_flow - gradient, rcond=None)[0]
    solenoidal = faces @ face_potential
    harmonic = edge_flow - gradient - solenoidal
    if not all(
        np.isfinite(part).all()
        for part in (gradient, solenoidal, harmonic, node_potential, face_potential)
    ):
        raise ValueError("decomposition exceeds finite floating-point range")
    return GraphHodgeDecomposition(
        gradient, solenoidal, harmonic, node_potential, face_potential
    )
