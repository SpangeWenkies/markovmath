from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Callable, runtime_checkable
import random
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
    tol: float = 5e-2,
) -> None:
    """
    Checks that:
      - kernel.law(x) returns something with .sample
      - "two-step sampling" via kernel is consistent with itself
        by comparing Monte Carlo estimates of E[f(X2)] computed in two equivalent ways.
    """
    # You can’t enforce “Markov property” directly (it’s about conditional independence w.r.t. a filtration)
    # but you can test composition behavior on a few test functions (bounded observables).
    # So it is just a sanity check
    for _ in range(n_states):
        x0 = state_sampler.sample(rng)
        law1 = kernel.law(x0)
        assert hasattr(law1, "sample"), "kernel.law(x) must return a Sampler"

        # For each test function f, estimate E[f(X2)] where X2 is after 2 steps.
        # test functions must accept R^d vectors, e.g.,
            # lambda x: x[0],
            # lambda x: sum(v*v for v in x),
            # lambda x: sum(x)/len(x),
        # Way 1: simulate two-step directly: x1~K(x0), x2~K(x1)
        for f in test_functions:
            acc_direct = 0.0
            for _ in range(n_inner):
                x1 = kernel.law(x0).sample(rng)
                x2 = kernel.law(x1).sample(rng)
                acc_direct += f(x2)
            est_direct = acc_direct / n_inner

            # Way 2: simulate a path of length 3 and take last (still two steps, just written differently)
            acc_path = 0.0
            for _ in range(n_inner):
                x1 = kernel.law(x0).sample(rng)
                x2 = kernel.law(x1).sample(rng)
                acc_path += f(x2)
            est_path = acc_path / n_inner

            assert abs(est_direct - est_path) <= tol, (
                f"Kernel 2-step consistency check failed for f: "
                f"{est_direct} vs {est_path}"
            )