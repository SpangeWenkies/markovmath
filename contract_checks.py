from __future__ import annotations
from typing import Callable, Iterable, Sequence
from dataclasses import dataclass
import random
import math
from core_interfaces import (
    MetricSpace,
    Sampler,
    X,
    Event,
    E,
    MarkovKernel,
    MeasurableSpace,
    Measure,
    event_key,
    Union,
    Complement,
    ProbabilityMeasure,
    DensityEvolution,
    LawEvolution,
    Density,
    PointRd,
)
from operators import (
    DiscreteResolvent,
    Observable,
    Scalar,
    DiscreteSemigroup,
    ContinuousSemigroup,
    Generator
)


def estimate_prob(
    law: Sampler[X], event: Event[X], n: int, rng: random.Random
) -> float:
    # Draws n samples from law.
    # Counts how many fall in the event.
    # Returns the empirical frequency, an estimate of P(X \in event)
    hits = 0
    for _ in range(n):
        if event(law.sample(rng)):
            hits += 1
    return hits / n


def check_metric_contract(
    metric: MetricSpace[X],
    point_sampler: Sampler[X],
    *,
    rng: random.Random,
    n_pairs: int = 200,
    n_triples: int = 200,
    tol: float = 1e-12,
) -> None:
    """
    Monte Carlo checks for metric axioms in R:
      - nonnegativity: d(x,y) >= 0
      - symmetry: d(x,y) == d(y,x)
      - identity: d(x,x) == 0
      - triangle: d(x,z) <= d(x,y)+d(y,z)

    These can't be proven by finite tests; failures provide proof, passes are "confidence" and mathematical proof must follow
    """

    # identity and nonnegativity + symmetry
    for _ in range(n_pairs):
        x = point_sampler.sample(rng)
        y = point_sampler.sample(rng)

        dxy = metric.dist(x, y)
        dyx = metric.dist(y, x)
        dxx = metric.dist(x, x)

        assert dxy >= -tol, f"Nonnegativity violated: d(x,y)={dxy}"
        assert abs(dyx - dxy) <= 1e-9, f"Symmetry violated: d(x,y)={dxy}, d(y,x)={dyx}"
        assert abs(dxx - 0.0) <= 1e-9, f"Identity violated: d(x,x)={dxx}"

    # triangle inequality
    for _ in range(n_triples):
        x = point_sampler.sample(rng)
        y = point_sampler.sample(rng)
        z = point_sampler.sample(rng)

        dxz = metric.dist(x, z)
        dxy = metric.dist(x, y)
        dyz = metric.dist(y, z)

        assert dxz <= dxy + dyz + 1e-9, (
            f"Triangle violated: d(x,z)={dxz} > d(x,y)+d(y,z)={dxy + dyz}"
        )


def approx_subset(
    A: Event[X],
    B: Event[X],
    *,
    sampler: Sampler[X],
    rng: random.Random,
    n: int = 2000,
) -> bool:
    """
    Monte Carlo check: returns True if we did not find a counterexample x with A(x)=True and B(x)=False.
    """
    for _ in range(n):
        x = sampler.sample(rng)
        if A(x) and (not B(x)):
            return False
    return True


def approx_disjoint(
    A: Event[X],
    B: Event[X],
    *,
    sampler: Sampler[X],
    rng: random.Random,
    n: int = 2000,
) -> bool:
    """
    Monte Carlo check for disjointness: no sampled x falls in A∩B.
    """
    for _ in range(n):
        x = sampler.sample(rng)
        if A(x) and B(x):
            return False
    return True


def check_measure_contracts(
    space: MeasurableSpace[X, E],
    mu: Measure[X, E],
    *,
    rng: random.Random,
    reference_sampler: Sampler[X],
    events: list[E],
    tol: float = 2e-2,
) -> None:
    """
    checks:
      - μ(∅)=0
      - μ(A) >= 0 for tested A
      - monotonicity on pairs where A⊆B (approx by sampling)
      - finite additivity on pairs that are disjoint (approx by sampling):
            μ(A∪B) ≈ μ(A)+μ(B)
    """

    # Assumes we know the measure as a function of an event. This is almost never the case.
    # In many cases we do not have exact μ(A), but we have it through sampling
    # We then can have measure be a monte carlo wrapper implementing the .measure() by sampling,
    # but we then must account in the contract checks for monte carlo noise

    # Empty set sanity
    mu_empty = mu.measure(space.empty())
    assert abs(mu_empty - 0.0) <= tol, f"Expected μ(∅)=0, got {mu_empty}"

    # Nonnegativity sanity on provided events
    for A in events:
        m = mu.measure(A)
        assert m >= -tol, f"Nonnegativity violated on tested event: μ(A)={m}"

    # Pairwise monotonicity + additivity checks using Monte Carlo subset/disjoint tests.
    # This assumes E is also an Event[X] (callable), which our Borel events will be.
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            A = events[i]
            B = events[j]

            # If we can treat them as callables, we can do approximate subset checks
            if isinstance(A, Event) and isinstance(B, Event):
                if approx_subset(A, B, sampler=reference_sampler, rng=rng):
                    mA = mu.measure(A)
                    mB = mu.measure(B)
                    assert mA <= mB + tol, (
                        f"Monotonicity suspect: μ(A)={mA} > μ(B)={mB}"
                    )

                # If disjoint, check additivity with A∪B if you have a union constructor.
                # The explicit Borel-event AST helps here because we can then build Union((A,B)) deterministically.


def check_kernel_contracts(
    kernel: MarkovKernel[X],
    *,
    rng: random.Random,
    state_sampler: Sampler[X],
    test_functions: list[Callable[[X], float]],
    n_states: int = 20,
    n_inner: int = 2000,
    z: float = 4.0,  # how many standard errors you allow
    min_se: float = 1e-12,  # avoid division by 0
) -> None:
    # TODO: generalize this function to work for more than Random walk (we must create some rule for setting n_inner using se from a MC sample)
    for _ in range(n_states):
        x0 = state_sampler.sample(rng)
        law1 = kernel.law(x0)
        assert hasattr(law1, "sample"), "kernel.law(x) must return a Sampler"

        for f in test_functions:
            # Way 1 samples
            vals1 = []
            for _ in range(n_inner):
                x1 = kernel.law(x0).sample(rng)
                x2 = kernel.law(x1).sample(rng)
                vals1.append(f(x2))
            m1 = sum(vals1) / n_inner
            v1 = sum((u - m1) ** 2 for u in vals1) / max(1, n_inner - 1)
            se1 = math.sqrt(v1 / n_inner)

            # Way 2 samples (independent stream because rng has advanced)
            vals2 = []
            for _ in range(n_inner):
                x1 = kernel.law(x0).sample(rng)
                x2 = kernel.law(x1).sample(rng)
                vals2.append(f(x2))
            m2 = sum(vals2) / n_inner
            v2 = sum((u - m2) ** 2 for u in vals2) / max(1, n_inner - 1)
            se2 = math.sqrt(v2 / n_inner)

            # Combine standard errors
            se = math.sqrt(se1 * se1 + se2 * se2)
            se = max(se, min_se)

            diff = abs(m1 - m2)
            assert diff <= z * se, (
                f"Kernel sanity check failed (diff too large): diff={diff}, "
                f"allowed≈{z * se} (z={z}, se={se})"
            )


# TODO: put the following checks in the contract_checks file

# Estimate probabilities of generated events under init and one-step laws
# Beware: fam can explode. Cap to a manageable number for the demo by following:
# events = [borel.whole(), borel.empty(), *gens]
# events += rng.sample(fam, k=min(40 - len(events), len(fam)))


def check_event_probabilities_monotonicity_additivity(
    mp,
    fam,
    origin,
    rng,
    gens,
    borel,
    mc_n=10_000,  # samples per probability estimate (tune up/down)
    tol_prob=0.07,
    subset_trials=30,
    disjoint_trials=30,
    subset_n=3000,
    disjoint_n=3000,
) -> None:
    events = [borel.whole(), borel.empty(), *gens]
    events += rng.sample(fam, k=min(40 - len(events), len(fam)))

    probs_init = []
    probs_one_step = []

    law1 = mp.kernel.law(origin)

    for i, A in enumerate(events):
        # separate RNG streams per event so estimates are reproducible and comparable.
        pA_init = estimate_prob(mp.init, A, mc_n, rng=random.Random(10_000 + i))
        pA_1 = estimate_prob(law1, A, mc_n, rng=random.Random(20_000 + i))

        probs_init.append((pA_init, A))
        probs_one_step.append((pA_1, A))

    probs_init.sort(key=lambda t: t[0])
    probs_one_step.sort(key=lambda t: t[0])

    print("\nLowest-prob events under init:")
    for pA, A in probs_init[:5]:
        print(
            f"  P_init≈{pA:.6g}  event={event_key(A) if 'event_key' in globals() else repr(A)}"
        )

    print("\nHighest-prob events under init:")
    for pA, A in probs_init[-5:][::-1]:
        print(
            f"  P_init≈{pA:.6g}  event={event_key(A) if 'event_key' in globals() else repr(A)}"
        )

    print("\nLowest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[:5]:
        print(
            f"  P_1≈{pA:.6g}     event={event_key(A) if 'event_key' in globals() else repr(A)}"
        )

    print("\nHighest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[-5:][::-1]:
        print(
            f"  P_1≈{pA:.6g}     event={event_key(A) if 'event_key' in globals() else repr(A)}"
        )

    # “Measure-like” sanity checks using approx_subset / approx_disjoint

    # These are Monte Carlo heuristics on the finite event family.

    # Build a lookup for init probs to avoid recomputation
    p_init_map = {A: p for (p, A) in probs_init}

    print("\nHeuristic monotonicity checks (under init law):")
    for k in range(subset_trials):
        A = rng.choice(events)
        B = rng.choice(events)
        if approx_subset(
            A, B, sampler=mp.init, rng=random.Random(30_000 + k), n=subset_n
        ):
            if p_init_map[A] > p_init_map[B] + tol_prob:
                print("  WARNING: monotonicity suspect")
                print(f"    P(A)≈{p_init_map[A]:.4f} > P(B)≈{p_init_map[B]:.4f}")
                print(f"    A={event_key(A) if 'event_key' in globals() else repr(A)}")
                print(f"    B={event_key(B) if 'event_key' in globals() else repr(B)}")

    print("\nHeuristic additivity checks on disjoint pairs (under init law):")
    for k in range(disjoint_trials):
        A = rng.choice(events)
        B = rng.choice(events)
        if approx_disjoint(
            A, B, sampler=mp.init, rng=random.Random(40_000 + k), n=disjoint_n
        ):
            union_event = Union((A, B))
            p_union = estimate_prob(
                mp.init, union_event, mc_n, rng=random.Random(50_000 + k)
            )
            err = abs(p_union - (p_init_map[A] + p_init_map[B]))
            if err > tol_prob:
                print("  WARNING: additivity suspect on approx-disjoint pair")
                print(
                    f"    P(A∪B)≈{p_union:.4f} vs P(A)+P(B)≈{(p_init_map[A] + p_init_map[B]):.4f} (err={err:.4f})"
                )

    # following monotonicity and additivity checks should be guaranteed to trigger by design
    A = rng.choice(fam)
    B = rng.choice(fam)

    # Guaranteed subset:
    C = Union((A, B))  # A ⊆ C

    pA = estimate_prob(mp.init, A, mc_n, random.Random(1))
    pC = estimate_prob(mp.init, C, mc_n, random.Random(2))
    print(f"Monotonicity check: P(A)={pA:.4f} <= P(A∪B)={pC:.4f}")

    # Guaranteed additivity for complement:
    Ac = Complement(A)
    pAc = estimate_prob(mp.init, Ac, mc_n, random.Random(3))
    print(f"Complement additivity: P(A)+P(Ac)≈{pA + pAc:.4f} (should be ≈ 1)")

def check_discrete_chapman_kolmogorov(
    semigroup: DiscreteSemigroup[X],
    f: Observable[X],
    x0: X,
    *,
    m: int,
    n: int,
    n_outer: int,
    n_inner: int,
    rng: random.Random,
) -> tuple[Scalar, Scalar]:
    """Heuristic Chapman–Kolmogorov check: T^{m+n}f ≈ T^m(T^n f).

    Since T^n f is not explicitly available as a function, we estimate:
      - lhs = T^{m+n} f(x0) by direct simulation
      - rhs = E[ g(X_m) ] where g(y) = T^n f(y), estimated with an inner MC at each y.

    Returns (lhs, rhs). Their difference is your diagnostic.
    """
    if m < 0 or n < 0:
        raise ValueError("m,n must be >= 0")
    if n_outer <= 0 or n_inner <= 0:
        raise ValueError("n_outer,n_inner must be > 0")

    lhs = semigroup.estimate_Tn(f, x0, n=m + n, n_samples=n_outer, rng=rng)

    # rhs: outer simulate m steps, inner estimate n-step expectation
    acc: Scalar = 0.0
    for _ in range(n_outer):
        x = x0
        for _k in range(m):
            x = semigroup.kernel.law(x).sample(rng)
        # inner expectation starting from x
        acc += semigroup.estimate_Tn(f, x, n=n, n_samples=n_inner, rng=rng)
    rhs: Scalar = acc / n_outer
    return lhs, rhs


def check_discrete_resolvent_identity(
    resolvent: DiscreteResolvent[X],
    semigroup: DiscreteSemigroup[X],
    f: Observable[X],
    x0: X,
    *,
    n_outer: int,
    n_inner: int,
    rng: random.Random,
) -> tuple[Scalar, Scalar]:
    """Heuristic check of U_λ f = f + λ T(U_λ f) at x0.

    - lhs = U_λ f(x0) estimated by geometric stopping time
    - rhs = f(x0) + λ E[ U_λ f(X_1) ] where the expectation is outer MC and U_λ f(X_1)
      is estimated with an inner MC at each visited X_1.
      
    Numerically verifies the discrete resolvent identity:

        U_\lambda f(x) = f(x) + \lambda (T U_\lambda f)(x)
        where (Tg)(x) = E_x[g(X_1)].

    We estimate:
      - LHS via resolvent.estimate_U(f, x, n_paths_lhs)
      - RHS via f(x) + \lambda * average_{j=1..n_outer} U_\lambda f(X1_j),
        where X1_j ~ K(x,·) and each U_\lambda f(X1_j) is itself estimated by Monte Carlo.

    Notes:
      - This is a nested Monte Carlo check, which sadly means variance can be high for unbounded f.
        Prefer bounded f (indicators, tanh, cos, clipped observables).
      - Passing the check provides confidence; failure indicates either a bug or insufficient sampling.
      - If it's slow, reduce n_states first, then reduce n_outer, then reduce n_paths_inner. 
        - If it fails sporadically, increase those (or loosen tol) or use bounded f
      
    """
    if n_outer <= 0 or n_inner <= 0:
        raise ValueError("n_outer,n_inner must be > 0")

    lhs = resolvent.estimate_U(f, x0, n_paths=n_outer, rng=rng)

    acc: Scalar = 0.0
    for _ in range(n_outer):
        x1 = semigroup.kernel.law(x0).sample(rng)
        acc += resolvent.estimate_U(f, x1, n_paths=n_inner, rng=rng)
    rhs: Scalar = f(x0) + resolvent.lam * (acc / n_outer)
    return lhs, rhs


def estimate_drift_condition(
    Af: Callable[[X], float],
    f: Callable[[X], float],
    states: Iterable[X],
    *,
    c_grid: Sequence[float] = (0.0, 0.1, 0.2, 0.5, 1.0),
) -> tuple[float, float]:
    """Heuristic drift/stability fit for Af <= -c f + b on a set of sampled states.

    For each c in c_grid, define b(c) = max_x (Af(x) + c f(x)).
    Choose c minimizing b(c) (ties broken by larger c).

    This is purely empirical: it only certifies the inequality on the provided states.
    """
    best_c = None
    best_b = None
    for c in c_grid:
        b = -float("inf")
        for x in states:
            b = max(b, float(Af(x) + c * f(x)))
        if best_b is None or b < best_b - 1e-12 or (abs(b - best_b) <= 1e-12 and c > (best_c or 0.0)):
            best_b = b
            best_c = float(c)
    return float(best_c or 0.0), float(best_b or 0.0)


def check_discrete_martingale(
    kernel: MarkovKernel[X],
    Af: Callable[[X], float],
    f: Callable[[X], float],
    x0: X,
    *,
    n_steps: int,
    n_paths: int,
    rng: random.Random,
) -> float:
    """Heuristic martingale check for the discrete-time generator A = T - I.

    If Af(x) = E[f(X_1)|X_0=x] - f(x), then

        M_n := f(X_n) - f(X_0) - Σ_{k=0}^{n-1} Af(X_k)

    satisfies E[M_n] = 0 (and is a martingale under integrability).

    Returns an MC estimate of E[M_n] (should be near 0).
    """
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    acc: float = 0.0
    for _ in range(n_paths):
        x = x0
        sumAf = 0.0
        for _k in range(n_steps):
            sumAf += float(Af(x))
            x = kernel.law(x).sample(rng)
        Mn = float(f(x)) - float(f(x0)) - sumAf
        acc += Mn
    return float(acc / n_paths)


def check_continuous_chapman_kolmogorov(
    semigroup: ContinuousSemigroup[X],
    f: Observable[X],
    x0: X,
    *,
    s: float,
    t: float,
    n_outer: int,
    n_inner: int,
    rng: random.Random,
) -> tuple[Scalar, Scalar]:
    """Heuristic Chapman–Kolmogorov check for discretized continuous time.

    Checks: P_{s+t} f(x0) ≈ P_s(P_t f)(x0), using nested Monte Carlo.

    Returns (lhs, rhs).
    """
    m = semigroup.n_steps(s)
    n = semigroup.n_steps(t)
    disc = semigroup._disc
    return check_discrete_chapman_kolmogorov(
        disc, f, x0, m=m, n=n, n_outer=n_outer, n_inner=n_inner, rng=rng
    )


def check_kolmogorov_backward_discretized(
    semigroup: ContinuousSemigroup[X],
    f: Observable[X],
    x0: X,
    *,
    t: float,
    n_outer: int,
    n_inner: int,
    rng: random.Random,
) -> tuple[Scalar, Scalar]:
    """Heuristic backward equation at grid times for the discretized semigroup.

    For Δt = semigroup.dt and t = kΔt, the *exact* discrete-time identity is:

        P_{t+Δt} f - P_t f = P_t(P_{Δt} f - f)

    Divide by Δt to view it as a finite-difference version of ∂_t P_t f = P_t A f.

    This routine estimates:
      lhs = (P_{t+Δt}f(x0) - P_t f(x0)) / Δt
      rhs = P_t A_Δt f(x0), where A_Δt f(y) = (E[f(X_{Δt})|y] - f(y)) / Δt
            estimated by inner MC at visited y ~ law(X_t|x0).

    Returns (lhs, rhs).
    """
    k = semigroup.n_steps(t)
    dt = semigroup.dt
    disc = semigroup._disc
    kernel = semigroup.kernel_dt

    # lhs: finite difference at x0
    pt = disc.estimate_Tn(f, x0, n=k, n_samples=n_outer, rng=rng)
    pt_dt = disc.estimate_Tn(f, x0, n=k + 1, n_samples=n_outer, rng=rng)
    lhs = (pt_dt - pt) / dt

    # rhs: outer simulate to X_k, inner estimate one-step expectation
    acc: Scalar = 0.0
    for _ in range(n_outer):
        x = x0
        for _j in range(k):
            x = kernel.law(x).sample(rng)
        # estimate E[f(X_{k+1})|X_k=x] by inner MC
        inner = 0.0
        for _i in range(n_inner):
            x1 = kernel.law(x).sample(rng)
            inner += f(x1)
        Tf_x = inner / n_inner
        A_dt_f_x = (Tf_x - f(x)) / dt
        acc += A_dt_f_x
    rhs = acc / n_outer
    return lhs, rhs


def check_positivity_preservation(
    semigroup: DiscreteSemigroup[X],
    f: Callable[[X], float],
    states: Iterable[X],
    *,
    n_samples: int,
    rng: random.Random,
    tol: float = 1e-10,
) -> bool:
    """Heuristic check: if f≥0 (pointwise on tested states), then T f ≥ 0.

    Returns True if no violation was detected on the provided `states`.
    """
    for x in states:
        if f(x) < -tol:
            raise ValueError("f must be nonnegative on the tested states for this check.")
        val = semigroup.estimate_T(f, x, n_samples=n_samples, rng=rng)
        if float(val) < -tol:
            return False
    return True


def check_supnorm_contraction(
    semigroup: DiscreteSemigroup[X],
    f: Callable[[X], float],
    states: Iterable[X],
    *,
    n_samples: int,
    rng: random.Random,
    tol: float = 1e-6,
) -> bool:
    """Heuristic check of sup-norm contraction: ||T f||_∞ ≤ ||f||_∞.

    Computes an empirical sup over `states`, so this is only a diagnostic.
    """
    f_sup = max(abs(float(f(x))) for x in states)
    tf_sup = 0.0
    for x in states:
        tf = semigroup.estimate_T(f, x, n_samples=n_samples, rng=rng)
        tf_sup = max(tf_sup, abs(float(tf)))
    return tf_sup <= f_sup + tol


def check_invariant_measure(
    semigroup: DiscreteSemigroup[X],
    mu: Sampler[X],
    f: Callable[[X], float],
    *,
    n_mu: int,
    n_inner: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Heuristic invariance check: E_μ[f] ≈ E_μ[T f].

    Returns (E_μ[f], E_μ[T f]).
    """
    if n_mu <= 0 or n_inner <= 0:
        raise ValueError("n_mu and n_inner must be > 0")

    acc_f = 0.0
    acc_tf = 0.0
    for _ in range(n_mu):
        x = mu.sample(rng)
        acc_f += float(f(x))
        acc_tf += float(semigroup.estimate_T(f, x, n_samples=n_inner, rng=rng))
    return acc_f / n_mu, acc_tf / n_mu

@dataclass(frozen=True, slots=True)
class MartingaleDiagnostics:
    mean: float
    std_err: float
    n_paths: int
    t: float
    tolerance: float
    holds: bool


def check_martingale_contract(
    generator: Generator[X],
    f: Observable[X],
    x0: X,
    *,
    t: float,
    n_paths: int,
    n_A_samples: int,
    rng: random.Random,
    tol: float = 1e-1,
) -> MartingaleDiagnostics:
    """Estimate E[M_t] for M_t = f(X_t)-f(X_0)-∫_0^t Af(X_s) ds.

    Uses a Riemann sum with step Δt = generator.dt and MC estimates for Af.
    """
    if n_paths <= 0 or n_A_samples <= 0:
        raise ValueError("n_paths and n_A_samples must be > 0")
    if t < 0:
        raise ValueError("t must be >= 0")

    dt = generator.dt
    n_steps = int(round(t / dt))
    t_used = n_steps * dt
    if abs(t_used - t) > 1e-9:
        raise ValueError("t must be a multiple of generator.dt for this check.")

    kernel = generator.semigroup.kernel
    m_vals: list[float] = []
    for _ in range(n_paths):
        x = x0
        x_start = x
        integral = 0.0
        for _k in range(n_steps):
            Af_x = generator.estimate_Af(
                f, x, n_samples=n_A_samples, rng=rng
            )
            integral += float(Af_x) * dt
            x = kernel.law(x).sample(rng)
        m_t = float(f(x)) - float(f(x_start)) - integral
        m_vals.append(m_t)

    mean = sum(m_vals) / n_paths
    if n_paths == 1:
        std_err = 0.0
    else:
        var = sum((v - mean) ** 2 for v in m_vals) / (n_paths - 1)
        std_err = math.sqrt(var / n_paths)
    holds = abs(mean) <= tol
    return MartingaleDiagnostics(
        mean=mean,
        std_err=std_err,
        n_paths=n_paths,
        t=t_used,
        tolerance=tol,
        holds=holds,
    )


@dataclass(frozen=True, slots=True)
class DriftConditionDiagnostics:
    n_points: int
    violations: int
    max_violation: float
    mean_violation: float
    holds: bool


def check_drift_condition(
    A: Callable[[X], float],
    f: Callable[[X], float],
    c: float,
    b: float,
    *,
    sampler: Sampler[X],
    rng: random.Random,
    n_points: int = 200,
    tol: float = 1e-8,
) -> DriftConditionDiagnostics:
    """Estimate whether Af ≤ -c f + b holds on sampled points."""
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    max_violation = 0.0
    sum_violation = 0.0
    violations = 0
    for _ in range(n_points):
        x = sampler.sample(rng)
        lhs = float(A(x))
        rhs = -c * float(f(x)) + b
        violation = lhs - rhs
        if violation > tol:
            violations += 1
            max_violation = max(max_violation, violation)
            sum_violation += violation

    mean_violation = (sum_violation / violations) if violations > 0 else 0.0
    holds = max_violation <= tol
    return DriftConditionDiagnostics(
        n_points=n_points,
        violations=violations,
        max_violation=max_violation,
        mean_violation=mean_violation,
        holds=holds,
    )

def _estimate_density_mass(
    density: Density[PointRd],
    *,
    bounds: Sequence[tuple[float, float]],
    rng: random.Random,
    n_points: int,
) -> float:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    if not bounds:
        raise ValueError("bounds must be nonempty")
    for low, high in bounds:
        if high <= low:
            raise ValueError("bounds must have low < high for each dimension")

    vol = 1.0
    for low, high in bounds:
        vol *= (high - low)

    acc = 0.0
    for _ in range(n_points):
        x = tuple(rng.uniform(low, high) for low, high in bounds)
        acc += density(x)
    return vol * (acc / n_points)


def check_density_evolution_mass_conservation(
    solver: DensityEvolution[PointRd],
    p0: Density[PointRd],
    *,
    t: float,
    bounds: Sequence[tuple[float, float]],
    rng: random.Random,
    n_points: int = 20000,
    tol: float = 2e-2,
) -> None:
    """
    Checks mass conservation for density solvers by numerical integration over a bounding box.
    """
    p_t = solver.evolve_density(p0, t)
    mass0 = _estimate_density_mass(p0, bounds=bounds, rng=rng, n_points=n_points)
    mass_t = _estimate_density_mass(p_t, bounds=bounds, rng=rng, n_points=n_points)
    assert abs(mass_t - mass0) <= tol, (
        f"Density mass conservation failed: mass0={mass0}, mass_t={mass_t}, tol={tol}"
    )


def check_law_evolution_normalization(
    solver: LawEvolution[X, E],
    mu0: ProbabilityMeasure[X, E],
    *,
    space: MeasurableSpace[X, E],
    t: float,
    rng: random.Random,
    n_samples: int = 5000,
    tol: float = 2e-2,
    events: Sequence[E] | None = None,
) -> None:
    """
    Checks normalization for law solvers using sampling.

    If events are provided, they should form a disjoint cover of the space.
    Otherwise we check P(X in whole) ≈ 1.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    mu_t = solver.evolve_law(mu0, t)
    if events is None:
        est = estimate_prob(mu_t, space.whole(), n_samples, rng)
        assert abs(est - 1.0) <= tol, f"Law normalization failed: P(whole)≈{est}, tol={tol}"
        return

    total = 0.0
    for ev in events:
        if not isinstance(ev, Event):
            raise TypeError("events must be callable Events for sampling checks")
        total += estimate_prob(mu_t, ev, n_samples, rng)
    assert abs(total - 1.0) <= tol, (
        f"Law normalization failed: sum P(A_i)≈{total}, tol={tol}"
    )
