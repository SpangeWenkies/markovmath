from __future__ import annotations
from typing import Callable
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
)
from operator_layer import (
    DiscreteResolvent,
    Observable,
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


def check_discrete_resolvent_identity(
    kernel: MarkovKernel[X],
    resolvent: DiscreteResolvent[X],
    f: Observable[X],
    *,
    rng: random.Random,
    state_sampler: Sampler[X],
    n_states: int = 8,
    # LHS: U_\lambda f(x)
    n_paths_lhs: int = 2000,
    # RHS: f(x) + \lambda T(U_\lambda f)(x)
    n_outer: int = 400,  # samples for T(·) expectation
    n_paths_inner: int = 400,  # paths to estimate U_\lambda f at each sampled next-state
    tol: float = 0.15,
) -> None:
    """
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
      - If it’s slow, reduce n_states first, then reduce n_outer, then reduce n_paths_inner. 
        - If it fails sporadically, increase those (or loosen tol) or use bounded f
    """
    lam = resolvent.lam
    for i in range(n_states):
        x0 = state_sampler.sample(rng)

        # LHS estimate
        lhs = resolvent.estimate_U(f, x0, n_paths=n_paths_lhs, rng=rng)

        # RHS estimate: f(x0) + lam * E[ U f(X1) ]
        acc = 0.0
        for j in range(n_outer):
            x1 = kernel.law(x0).sample(rng)
            u_x1 = resolvent.estimate_U(f, x1, n_paths=n_paths_inner, rng=rng)
            acc += u_x1
        rhs = f(x0) + lam * (acc / n_outer)

        err = abs(lhs - rhs)
        assert err <= tol, (
            "Discounted resolvent identity check failed:\n"
            f"  |U f(x) - (f(x) + \lambda T(U f)(x))| = {err}\n"
            f"  \lambda={lam}, tol={tol}\n"
            f"  lhs={lhs}, rhs={rhs}\n"
        )
