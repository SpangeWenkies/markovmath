from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Callable, runtime_checkable
import random
import math
from core_interfaces import MetricSpace, Sampler, X, Event, E, MarkovKernel, MeasurableSpace, Measure

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
            f"Triangle violated: d(x,z)={dxz} > d(x,y)+d(y,z)={dxy+dyz}"
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
                    assert mA <= mB + tol, f"Monotonicity suspect: μ(A)={mA} > μ(B)={mB}"

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
    z: float = 4.0,          # how many standard errors you allow
    min_se: float = 1e-12,   # avoid division by 0
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
                f"allowed≈{z*se} (z={z}, se={se})"
            )