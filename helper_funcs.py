from typing import List
import math

Point = tuple[float, ...]


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
