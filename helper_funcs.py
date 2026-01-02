from typing import List, TypeVar, Callable, Sequence, TypeAlias
import math

PointRd: TypeAlias = tuple[float, ...]
MutablePointRd: TypeAlias = List[float]
X = TypeVar("X")
Observable: TypeAlias = Callable[[X], float]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]  # is this mutable? don't think so
MutableMatrix: TypeAlias = List[List[float]]

def to_mutable(A: Matrix) -> MutableMatrix:
    return [list(row) for row in A]

def to_immutable(A: MutableMatrix) -> Matrix:
    return tuple(tuple(row) for row in A)

def repair_corr_quick(corr: Matrix, tol: float = 1e-12) -> Matrix:
    mut_corr = to_mutable(corr)
    d = len(mut_corr)

    # 1) symmetrize
    A = [[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            A[i][j] = 0.5 * (mut_corr[i][j] + mut_corr[j][i])

    # 2) project to PSD: A = Q Λ Q^T, clamp Λ>=0
    eigvals, Q = jacobi_eigh_sym(A, tol=tol)
    lam = [0.0]*d
    for k, v in enumerate(eigvals):
        # allow tiny negative due to rounding
        if v < 0.0 and abs(v) <= 10*tol:
            lam[k] = 0.0
        elif v < 0.0:
            lam[k] = 0.0  # hard clamp
        else:
            lam[k] = v

    # reconstruct PSD matrix: X = Q diag(lam) Q^T
    X = [[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(d):
                s += Q[i][k] * lam[k] * Q[j][k]
            X[i][j] = s

    # 3) renormalize to correlation (force diag=1):
    # R_ij = X_ij / sqrt(X_ii X_jj)
    diag = [X[i][i] for i in range(d)]
    for i in range(d):
        if diag[i] < tol:
            # if diag is ~0, you have a degenerate dimension; set safe scaling
            diag[i] = 0.0

    R = [[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            if i == j:
                R[i][j] = 1.0
            else:
                denom = math.sqrt(max(diag[i], 0.0) * max(diag[j], 0.0))
                R[i][j] = (X[i][j] / denom) if denom > 0.0 else 0.0

    return to_immutable(R)

def cholesky_spd(a: Matrix, tol: float = 1e-12) -> Matrix: 
    """ 
    Cholesky factorization for symmetric positive definite matrices. 
    Returns lower-triangular L such that a = L L^T. 
    Raises ValueError if a is not symmetric positive definite (within tolerance).
    """ 
    mut_a = to_mutable(a)
    d = len(mut_a) 
    if any(len(row) != d for row in mut_a): 
        raise ValueError("matrix must be square") # symmetry check 
    for i in range(d): 
        for j in range(i + 1, d): 
            if abs(mut_a[i][j] - mut_a[j][i]) > 1e-9: 
                raise ValueError("matrix must be symmetric for Cholesky") 
    L = [[0.0] * d for _ in range(d)] 
    for i in range(d): 
        for j in range(i + 1): 
            s = mut_a[i][j] 
            for k in range(j): 
                s -= L[i][k] * L[j][k] 
            if i == j: 
                if s <= tol: 
                    raise ValueError( "matrix not SPD (symmetric positive definite); check corr/stds" ) 
                L[i][j] = math.sqrt(s) 
            else: 
                L[i][j] = s / L[j][j] 
    return to_immutable(L)

def matvec(A: Matrix, x: List[float]) -> List[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def matvec_lower(L: Matrix, x: List[float]) -> List[float]:
    d = len(L)
    y = [0.0] * d
    for i in range(d):
        s = 0.0
        Li = L[i]
        for j in range(i + 1):
            s += Li[j] * x[j]
        y[i] = s
    return y

def _add_jitter(a: MutableMatrix, eps: float) -> Matrix:
    d = len(a)
    out = [row[:] for row in a]
    for i in range(d):
        out[i][i] += eps
    return out

def cholesky_with_jitter(a: Matrix, *, eps0: float = 1e-12, mult: float = 10.0, eps_max: float = 1e-3) -> Matrix:
    a_mut = to_mutable(a)
    eps = eps0
    last_err = None
    while eps <= eps_max:
        try:
            return cholesky_spd(_add_jitter(a_mut, eps))
        except ValueError as e:
            last_err = e
            eps *= mult
    raise ValueError(f"Cholesky failed even with jitter up to {eps_max}. Last error: {last_err}")

def jacobi_eigh_sym(a: MutableMatrix, tol: float = 1e-12, max_sweeps: int = 50):
    """
    Symmetric eigen-decomposition via Jacobi rotations.
    Returns (eigvals, Q) where Q is orthonormal and a ≈ Q diag(eigvals) Q^T.
    Pure Python; OK for moderate dimensions, for 100+ needs NumPy/LAPACK
    """
    d = len(a)
    A = [row[:] for row in a]

    # Q = I
    Q = [[0.0]*d for _ in range(d)]
    for i in range(d):
        Q[i][i] = 1.0

    def max_offdiag():
        p = q = 0
        m = 0.0
        for i in range(d):
            for j in range(i+1, d):
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

        # Compute rotation angle
        phi = 0.5 * math.atan2(2.0*apq, (aqq - app))
        c = math.cos(phi)
        s = math.sin(phi)

        # Rotate A: update rows/cols p,q
        for k in range(d):
            if k != p and k != q:
                aik = A[k][p]
                akq = A[k][q]
                A[k][p] = A[p][k] = c*aik - s*akq
                A[k][q] = A[q][k] = s*aik + c*akq

        A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
        A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq
        A[p][q] = A[q][p] = 0.0

        # Update Q
        for k in range(d):
            qkp = Q[k][p]
            qkq = Q[k][q]
            Q[k][p] = c*qkp - s*qkq
            Q[k][q] = s*qkp + c*qkq

    eigvals = [A[i][i] for i in range(d)]
    return eigvals, Q

def psd_factor_via_eigh(a: Matrix, tol: float = 1e-12) -> List[List[float]]:
    """
    For symmetric PSD matrix a, return factor B such that a ≈ B B^T.
    Uses eigen-decomp; clamps tiny negative eigenvalues to 0.
    """
    a_mut = to_mutable(a)
    eigvals, Q = jacobi_eigh_sym(a_mut, tol=tol)
    d = len(a_mut)

    lam = [0.0]*d
    for i, v in enumerate(eigvals):
        # clamp negatives due to numerical error
        lam[i] = 0.0 if v < 0.0 and abs(v) <= 10*tol else v
        if lam[i] < 0.0:
            raise ValueError(f"Matrix not PSD: eigenvalue {v} < 0")

    sqrt_lam = [math.sqrt(v) if v > 0.0 else 0.0 for v in lam]

    # B = Q * diag(sqrt_lam)
    B = [[Q[i][k] * sqrt_lam[k] for k in range(d)] for i in range(d)]
    return to_immutable(B)


def rd_key(x: PointRd, ndigits: int = 2) -> PointRd:
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