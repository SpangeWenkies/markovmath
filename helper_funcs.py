from typing import List, TypeVar, Callable, Sequence, TypeAlias, Optional
import math

Vector: TypeAlias = tuple[float, ...]
MutableVector: TypeAlias = list[float]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]
MutableMatrix: TypeAlias = list[list[float]]

X = TypeVar("X")
Observable = Callable[[X], float]

def to_mutable(mat: Matrix) -> MutableMatrix:
    """Convert an immutable matrix (tuple-of-tuples) to a mutable list-of-lists."""
    return [list(row) for row in mat]

def to_immutable(mat: MutableMatrix) -> Matrix:
    """Convert a mutable list-of-lists matrix to an immutable tuple-of-tuples."""
    return tuple(tuple(row) for row in mat)

def to_mutable_point(p: Vector) -> MutableVector:
    """Convert an immutable vector (tuple) to a mutable list."""
    return list(p)

def to_immutable_point(p: Sequence[float]) -> Vector:
    """Convert a sequence of floats to an immutable vector (tuple)."""
    return tuple(float(v) for v in p)


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


def cov_from_stds_and_corr(stds: Vector, corr: List[List[float]]) -> List[List[float]]:
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


# --- Correlation / covariance repair and PSD handling ---

def cholesky_with_jitter(
    a: List[List[float]],
    *,
    tol: float = 1e-12,
    eta0: float = 1e-12,
    mult: float = 10.0,
    eta_max: float = 1e-3,
) -> List[List[float]]:
    """Try Cholesky on (a + eta I) with increasing eta until it succeeds.

    This is a common numerical regularization ('jitter' / 'nugget').
    It changes the covariance from Σ to Σ + ηI.
    """
    d = len(a)
    eta = eta0
    last_err: Optional[Exception] = None
    while eta <= eta_max:
        # add jitter to diagonal
        aj = [row[:] for row in a]
        for i in range(d):
            aj[i][i] += eta
        try:
            return cholesky_spd(aj, tol=tol)
        except ValueError as e:
            last_err = e
            eta *= mult
    raise ValueError(
        f"Cholesky failed even with jitter up to {eta_max}. Last error: {last_err}"
    )


def jacobi_eigh_sym(
    a: List[List[float]],
    *,
    tol: float = 1e-12,
    max_sweeps: int = 50,
) -> tuple[List[float], List[List[float]]]:
    """Eigen-decomposition of a symmetric matrix via Jacobi rotations.

    Returns (eigenvalues, Q) where Q is orthonormal and
    a ≈ Q diag(eigenvalues) Q^T.

    Pure Python; suitable for small/medium d.
    """
    d = len(a)
    if any(len(row) != d for row in a):
        raise ValueError("matrix must be square")

    # Work on a mutable copy and enforce symmetry by averaging.
    A = [[0.5 * (a[i][j] + a[j][i]) for j in range(d)] for i in range(d)]

    # Q = I
    Q = [[0.0] * d for _ in range(d)]
    for i in range(d):
        Q[i][i] = 1.0

    def max_offdiag() -> tuple[float, int, int]:
        m = 0.0
        p = 0
        q = 0
        for i in range(d):
            for j in range(i + 1, d):
                v = abs(A[i][j])
                if v > m:
                    m = v
                    p, q = i, j
        return m, p, q

    for _ in range(max_sweeps):
        m, p, q = max_offdiag()
        if m < tol:
            break

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]
        if abs(apq) < tol:
            continue

        phi = 0.5 * math.atan2(2.0 * apq, (aqq - app))
        c = math.cos(phi)
        s = math.sin(phi)

        # rotate rows/cols p,q of A
        for k in range(d):
            if k != p and k != q:
                aik = A[k][p]
                akq = A[k][q]
                A[k][p] = A[p][k] = c * aik - s * akq
                A[k][q] = A[q][k] = s * aik + c * akq

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p][q] = A[q][p] = 0.0

        # update eigenvectors (columns p,q)
        for k in range(d):
            qkp = Q[k][p]
            qkq = Q[k][q]
            Q[k][p] = c * qkp - s * qkq
            Q[k][q] = s * qkp + c * qkq

    eigvals = [A[i][i] for i in range(d)]
    return eigvals, Q


def project_psd_frobenius(
    a: List[List[float]],
    *,
    tol: float = 1e-12,
) -> List[List[float]]:
    """Project a symmetric matrix onto the PSD cone (Frobenius norm).

    For symmetric A = QΛQ^T, the projection is Q max(Λ,0) Q^T.
    """
    eigvals, Q = jacobi_eigh_sym(a, tol=tol)
    d = len(a)
    lam = [0.0] * d
    for k, v in enumerate(eigvals):
        # clamp tiny negatives caused by rounding
        if v < 0.0 and abs(v) <= 10 * tol:
            lam[k] = 0.0
        elif v < 0.0:
            lam[k] = 0.0
        else:
            lam[k] = v

    X = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(d):
                s += Q[i][k] * lam[k] * Q[j][k]
            X[i][j] = s
    # enforce symmetry (guards against tiny numerical asymmetries)
    for i in range(d):
        for j in range(i + 1, d):
            v = 0.5 * (X[i][j] + X[j][i])
            X[i][j] = X[j][i] = v
    return X


def psd_factor_from_eigh(
    a: List[List[float]],
    *,
    tol: float = 1e-12,
) -> List[List[float]]:
    """Return B such that a ≈ B B^T for symmetric PSD a.

    Uses eigen-decomposition; supports singular (PSD) matrices.
    Raises if a has a significantly negative eigenvalue.
    """
    eigvals, Q = jacobi_eigh_sym(a, tol=tol)
    d = len(a)
    sqrt_lam = [0.0] * d
    for k, v in enumerate(eigvals):
        if v < -100 * tol:
            raise ValueError(f"matrix not PSD; eigenvalue {v} < 0")
        sqrt_lam[k] = math.sqrt(v) if v > 0.0 else 0.0
    # B = Q diag(sqrt_lam)
    B = [[Q[i][k] * sqrt_lam[k] for k in range(d)] for i in range(d)]
    return B


def repair_corr_quick(corr: Matrix, *, tol: float = 1e-12) -> Matrix:
    """Fast repair: symmetrize -> PSD projection -> renormalize diag to 1.

    Common in practice when corr is 'almost' valid but numerically indefinite.
    Not guaranteed to be the true nearest correlation matrix.
    """
    A = to_mutable(corr)
    d = len(A)

    # symmetrize
    for i in range(d):
        for j in range(i + 1, d):
            v = 0.5 * (A[i][j] + A[j][i])
            A[i][j] = A[j][i] = v

    # project to PSD (nearest PSD matrix in Frobenius norm)
    X = project_psd_frobenius(A, tol=tol)

    # renormalize to correlation: R_ij = X_ij / sqrt(X_ii X_jj), diag=1
    diag = [X[i][i] for i in range(d)]
    R = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            if i == j:
                R[i][j] = 1.0
            else:
                denom = math.sqrt(max(diag[i], 0.0) * max(diag[j], 0.0))
                R[i][j] = (X[i][j] / denom) if denom > 0.0 else 0.0
    return to_immutable(R)


def _frob_norm(a: List[List[float]]) -> float:
    return math.sqrt(sum(v * v for row in a for v in row))


def _mat_sub(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    d = len(a)
    return [[a[i][j] - b[i][j] for j in range(d)] for i in range(d)]


def _mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    d = len(a)
    return [[a[i][j] + b[i][j] for j in range(d)] for i in range(d)]


def _set_unit_diag(a: List[List[float]]) -> List[List[float]]:
    d = len(a)
    out = [row[:] for row in a]
    for i in range(d):
        out[i][i] = 1.0
    return out


def nearest_psd_correlation_higham(
    corr: Matrix,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
    auto_boost_iter: bool = True,
    near_zero_eig: float = 1e-8,
    boosted_max_iter: int = 200,
) -> Matrix:
    """Higham/Dykstra: nearest correlation matrix (PSD + unit diagonal).

    This targets the true projection onto the intersection of:
      - PSD symmetric matrices
      - matrices with diag = 1
    under Frobenius norm (up to algorithmic tolerances).
    
    If we have close to zero eigenvalues and we would like to not Jitter then we run more iterations of the algorithm
    """
    Y = to_mutable(corr)
    d = len(Y)

    # symmetrize initial
    for i in range(d):
        for j in range(i + 1, d):
            v = 0.5 * (Y[i][j] + Y[j][i])
            Y[i][j] = Y[j][i] = v

    if auto_boost_iter and d >= 2:
        eigvals, _ = jacobi_eigh_sym([row[:] for row in Y], tol=tol)    # use a COPY because jacobi_eigh_sym mutates its input.
        # “close to zero” means we are near the PSD boundary, note that convergence can be slower.
        if min(abs(v) for v in eigvals) < near_zero_eig:
            max_iter = max(max_iter, boosted_max_iter)

    Delta = [[0.0] * d for _ in range(d)]

    for _ in range(max_iter):
        R = _mat_sub(Y, Delta)
        X = project_psd_frobenius(R, tol=tol)
        Delta = _mat_sub(X, R)
        Y_next = _set_unit_diag(X)

        diff = _frob_norm(_mat_sub(Y_next, Y))
        base = max(1.0, _frob_norm(Y))
        Y = Y_next
        if diff / base < 1e-10:
            break

    # enforce symmetry once more
    for i in range(d):
        for j in range(i + 1, d):
            v = 0.5 * (Y[i][j] + Y[j][i])
            Y[i][j] = Y[j][i] = v
    return to_immutable(Y)

def rd_key(x: tuple[float, ...], ndigits: int = 2) -> tuple[float, ...]:
    """Coarse key for R^d points represented as tuples. Rounds to a grid"""
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