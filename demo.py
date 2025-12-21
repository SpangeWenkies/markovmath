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
    Union,
    Complement,
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
import math

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

    # we choose balls with overlap in R^d for the demo
    gens = [
        borel.ball(center=origin, radius=1.2 * math.sqrt(d)),
        borel.ball(center=tuple(0.6 for _ in range(d)), radius=1.1 * math.sqrt(d)),
        borel.ball(center=tuple(-0.6 for _ in range(d)), radius=1.1 * math.sqrt(d)),
    ]
    
    # metric contract inputs
    n_pairs = 200
    n_triples = 200

    # kernel contract inputs (in the RW demo the n_inner you choose depends on the test function, n_states, z_kernel, step_std)
    # TODO: generalize this
    # n_states = 10
    # n_inner = 1000
    # min_se = 1e-12 # avoid divide by zero
    # z_kernel =

    # kernel contract check functions f: R^d -> R, that do not blow up in variance in R^d
        # if using variance blow up test functions we must increase n_inner
    # test_functions = [
    #     lambda x: x[0],
    #     lambda x: math.tanh(x[0]),  # does not blow up as is bounded and smooth
    #     lambda x: math.cos(x[0]),   # does not blow up as is bounded and smooth
    #     lambda x: 1.0 if gens[0](x) else 0.0,  # indicator of a generator ball (moderate-probability event of being inside ball)
    # ]

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

    # check_kernel_contracts(
    #         mp.kernel,
    #         rng=random.Random(456),
    #         state_sampler=mp.init,
    #         test_functions=test_functions,
    #         n_states=n_states,
    #         n_inner=n_inner,
    #         z=z_kernel,
    #         min_se=min_se,
    #     )
    # print("Kernel contract check: OK (heuristic).")

    # TODO: put the following checks in the contract_checks file 

    # Estimate probabilities of generated events under init and one-step laws
    # Beware: fam can explode. Cap to a manageable number for the demo by following:
        # events = [borel.whole(), borel.empty(), *gens]
        # events += rng.sample(fam, k=min(40 - len(events), len(fam)))

    mc_n = 10_000  # samples per probability estimate (tune up/down)
    probs_init = []
    probs_one_step = []

    law1 = mp.kernel.law(origin)

    for i, A in enumerate(fam):
        # separate RNG streams per event so estimates are reproducible and comparable.
        pA_init = estimate_prob(mp.init, A, mc_n, rng=random.Random(10_000 + i))
        pA_1    = estimate_prob(law1,   A, mc_n, rng=random.Random(20_000 + i))

        probs_init.append((pA_init, A))
        probs_one_step.append((pA_1, A))

    probs_init.sort(key=lambda t: t[0])
    probs_one_step.sort(key=lambda t: t[0])

    print("\nLowest-prob events under init:")
    for pA, A in probs_init[:5]:
        print(f"  P_init≈{pA:.6g}  event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nHighest-prob events under init:")
    for pA, A in probs_init[-5:][::-1]:
        print(f"  P_init≈{pA:.6g}  event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nLowest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[:5]:
        print(f"  P_1≈{pA:.6g}     event={event_key(A) if 'event_key' in globals() else repr(A)}")

    print("\nHighest-prob events after 1 step from origin:")
    for pA, A in probs_one_step[-5:][::-1]:
        print(f"  P_1≈{pA:.6g}     event={event_key(A) if 'event_key' in globals() else repr(A)}")

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
        A = rng.choice(fam) # Axiom of choice used?
        B = rng.choice(fam)
        if approx_subset(A, B, sampler=mp.init, rng=random.Random(30_000 + k), n=subset_n):
            if p_init_map[A] > p_init_map[B] + tol_prob:
                print("  WARNING: monotonicity suspect")
                print(f"    P(A)≈{p_init_map[A]:.4f} > P(B)≈{p_init_map[B]:.4f}")
                print(f"    A={event_key(A) if 'event_key' in globals() else repr(A)}")
                print(f"    B={event_key(B) if 'event_key' in globals() else repr(B)}")

    print("\nHeuristic additivity checks on disjoint pairs (under init law):")
    for k in range(disjoint_trials):
        A = rng.choice(fam)
        B = rng.choice(fam)
        if approx_disjoint(A, B, sampler=mp.init, rng=random.Random(40_000 + k), n=disjoint_n):
            union_event = Union((A, B))
            p_union = estimate_prob(mp.init, union_event, mc_n, rng=random.Random(50_000 + k))
            err = abs(p_union - (p_init_map[A] + p_init_map[B]))
            if err > tol_prob:
                print("  WARNING: additivity suspect on approx-disjoint pair")
                print(f"    P(A∪B)≈{p_union:.4f} vs P(A)+P(B)≈{(p_init_map[A]+p_init_map[B]):.4f} (err={err:.4f})")

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
    print(f"Complement additivity: P(A)+P(Ac)≈{pA+pAc:.4f} (should be ≈ 1)")

    # note following plot has nothing to do with the generated events
    
    # Plot time vs point_i for i = 1,...,d (shown as x1..xd)
    df = pd.DataFrame(path, columns=[f"x{i}" for i in range(1, d + 1)])
    df.insert(0, "t", range(len(path)))  # time column first

    df_long = df.melt(id_vars="t", var_name="coord", value_name="value")
    sns.lineplot(data=df_long, x="t", y="value", hue="coord")