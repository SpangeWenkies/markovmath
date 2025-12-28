from typing import List, TypeVar, Callable, Sequence
import math

Point = tuple[float, ...]
X = TypeVar("X")
Observable = Callable[[X], float]


def _validate_corr_matrix(corr: List[List[float]], tol: float = 1e-12) -> None:
    d = len(corr)
    if d == 0 or any(len(row) != d for row in corr):
        raise ValueError("corr must be a nonempty square matrix (d x d)")
    for i in range(d):
        if abs(corr[i][i] - 1.0) > 1e-9:
            raise ValueError(f"corr diagonal must be 1; corr[{i}][{i}]={corr[i][i]}")
        for j in range(d):
            if abs(corr[i][j] - corr[j][i]) > 1e-9:
                raise ValueError("corr must be symmetric")
            if corr[i][j] < -1.0 - tol or corr[i][j] > 1.0 + tol:
                raise ValueError("corr entries must be in [-1, 1]")


def cov_from_stds_and_corr(stds: Point, corr: List[List[float]]) -> List[List[float]]:
    _validate_corr_matrix(corr)
    d = len(stds)
    if len(corr) != d:
        raise ValueError("stds length and corr dimension must match")
    if any(s < 0 for s in stds):
        raise ValueError("stds must be nonnegative")

    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = corr[i][j] * stds[i] * stds[j]
    return cov


def cholesky_spd(a: List[List[float]], tol: float = 1e-12) -> List[List[float]]:
    """
    Cholesky factorization for symmetric positive definite matrices.
    Returns lower-triangular L such that a = L L^T.
    Raises ValueError if a is not SPD (within tolerance).
    """
    d = len(a)
    if any(len(row) != d for row in a):
        raise ValueError("matrix must be square")

    # symmetry check
    for i in range(d):
        for j in range(i + 1, d):
            if abs(a[i][j] - a[j][i]) > 1e-9:
                raise ValueError("matrix must be symmetric for Cholesky")

    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= tol:
                    raise ValueError(
                        "matrix not SPD (nonpositive pivot); check corr/stds"
                    )
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L

def rd_key(x: tuple[float, ...], ndigits: int = 2) -> tuple[float, ...]:
    """Coarse key for R^d points represented as tuples."""
    return tuple(round(xi, ndigits) for xi in x)


def indicator(A: Callable[[X], bool]) -> Observable[X]:
    """Turn an event A(x)->bool into the indicator observable 1_A(x)."""
    return lambda x: 1.0 if A(x) else 0.0


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    if len(u) != len(v):
        raise ValueError("dimension mismatch in dot product")
    return float(sum(ui * vi for ui, vi in zip(u, v)))


def _as_float_seq(x: object) -> Sequence[float]:
    """Best-effort conversion to a sequence of floats (for R^d utilities).

    Intended for states represented as tuples/lists of floats.
    """
    if isinstance(x, (tuple, list)):
        return [float(v) for v in x]
    raise TypeError("Expected state to be a tuple/list of floats for this helper.")