from core_interfaces import (
    PointRd, 
    LpMetricRd, 
    StdBorelSpaceRd, 
    generate_event_family, 
    MarkovProcess, 
    NormalRd, 
    RandomWalkKernelRd,
    estimate_prob,
    event_key,
    Union
)
from contract_checks import (
    check_metric_contract,
    check_kernel_contracts,
    approx_subset,
    approx_disjoint,
)
import seaborn as sns
import pandas as pd
import random

if __name__ == "__main__":
    rng = random.Random(0)

    # --- Demo parameters (edit these) ---
    d = 3   # dimension of R^d
    p = 2.0 # set to 1<=p<=math.inf (math.inf for supremum norm) (for this range Minkowski inequality holds)
    step_std = 0.5
    init_std = 1.0
    n_steps = 200
    max_depth = 3

    # initial point
    origin: PointRd = tuple(0.0 for _ in range(d))

    # build some open balls (generators) in R^d using the chosen L^p metric
    metric = LpMetricRd(p=p)
    borel = StdBorelSpaceRd(metric=metric)
    gens = [
        borel.ball(center=origin, radius=1.0),
        borel.ball(center=tuple(1.0 for _ in range(d)), radius=0.75),
        borel.ball(center=tuple(-1.0 for _ in range(d)), radius=0.5),
    ]
    
    # metric contract inputs
    n_pairs = 200
    n_triples = 200

    # kernel contract check functions f: R^d -> R
    test_functions = [
        lambda x: x[0],
        lambda x: sum(v*v for v in x),
        lambda x: sum(x)/len(x),
    ]

    # -----------------------------------

    fam = generate_event_family(gens, max_depth=max_depth)
    print(f"Generated {len(fam)} events (R^{d}, p={p}, depth={max_depth}).")

    

    # TODO: do sanity and contract checks (monte carlo style) using this generated set, or maybe use estimate_prob
        # we must make the contract checks work in R^d first

    # Random walk in R^d
    mp = MarkovProcess(
        init=NormalRd(mean=origin, std=init_std),
        kernel=RandomWalkKernelRd(step_std=step_std),
    )
    path = mp.sample_path(n_steps, rng=rng)

    check_metric_contract(
        metric,
        point_sampler=mp.init,
        rng=random.Random(123),
        n_pairs=n_pairs,
        n_triples=n_pairs,
    )
    print("Metric contract check: OK (heuristic).")

    check_kernel_contracts(
            mp.kernel,
            rng=random.Random(456),
            state_sampler=mp.init,
            test_functions=test_functions,
            n_states=10,
            n_inner=1000,
            tol=5e-2,
        )
    print("Kernel contract check: OK (heuristic).")

    # Estimate probabilities of generated events under init and one-step laws
    # Beware: fam can explode. Cap to a manageable number for the demo.
    events = fam[: min(len(fam), 40)]

    mc_n = 10_000  # samples per probability estimate (tune up/down)
    probs_init = []
    probs_one_step = []

    law1 = mp.kernel.law(origin)

    for i, A in enumerate(events):
        # separate RNG streams per event so estimates are reproducible and comparable.
        pA_init = estimate_prob(mp.init, A, mc_n, rng=random.Random(10_000 + i))
        pA_1    = estimate_prob(law1,   A, mc_n, rng=random.Random(20_000 + i))

        probs_init.append((pA_init, A))
        probs_one_step.append((pA_1, A))

    probs_init.sort(key=lambda t: t[0])
    probs_one_step.sort(key=lambda t: t[0])

    print("\nLowest-prob events under init:")
    for pA, A in probs_init[:5]:
        print(f"  P_init≈{pA:.4f}  event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nHighest-prob events under init:")
    for pA, A in probs_init[-5:][::-1]:
        print(f"  P_init≈{pA:.4f}  event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nLowest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[:5]:
        print(f"  P_1≈{pA:.4f}     event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nHighest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[-5:][::-1]:
        print(f"  P_1≈{pA:.4f}     event={event_key(A) if 'event_key' in globals() else repr(A)}")

    # “Measure-like” sanity checks using approx_subset / approx_disjoint
    # These are Monte Carlo heuristics on the finite event family.
    tol_prob = 0.07
    subset_trials = 30
    disjoint_trials = 30
    subset_n = 3000
    disjoint_n = 3000

    # Build a lookup for init probs to avoid recomputation
    p_init_map = {A: p for (p, A) in probs_init}

    print("\nHeuristic monotonicity checks (under init law):")
    for k in range(subset_trials):
        A = rng.choice(events) # Axiom of choice used?
        B = rng.choice(events)
        if approx_subset(A, B, sampler=mp.init, rng=random.Random(30_000 + k), n=subset_n):
            if p_init_map[A] > p_init_map[B] + tol_prob:
                print("  WARNING: monotonicity suspect")
                print(f"    P(A)≈{p_init_map[A]:.4f} > P(B)≈{p_init_map[B]:.4f}")
                print(f"    A={event_key(A) if 'event_key' in globals() else repr(A)}")
                print(f"    B={event_key(B) if 'event_key' in globals() else repr(B)}")

    print("\nHeuristic additivity checks on disjoint pairs (under init law):")
    for k in range(disjoint_trials):
        A = rng.choice(events)
        B = rng.choice(events)
        if approx_disjoint(A, B, sampler=mp.init, rng=random.Random(40_000 + k), n=disjoint_n):
            union_event = Union((A, B))
            p_union = estimate_prob(mp.init, union_event, mc_n, rng=random.Random(50_000 + k))
            err = abs(p_union - (p_init_map[A] + p_init_map[B]))
            if err > tol_prob:
                print("  WARNING: additivity suspect on approx-disjoint pair")
                print(f"    P(A∪B)≈{p_union:.4f} vs P(A)+P(B)≈{(p_init_map[A]+p_init_map[B]):.4f} (err={err:.4f})")

    # Plot time vs point_i for i = 1,...,d (shown as x1..xd)
    df = pd.DataFrame(path, columns=[f"x{i}" for i in range(1, d + 1)])
    df.insert(0, "t", range(len(path)))  # time column first

    df_long = df.melt(id_vars="t", var_name="coord", value_name="value")
    sns.lineplot(data=df_long, x="t", y="value", hue="coord")