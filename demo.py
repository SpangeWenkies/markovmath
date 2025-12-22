from core_interfaces import (
    PointRd,
    LpMetricRd,
    StdBorelSpaceRd,
    generate_event_family,
    MarkovProcess,
    NormalRd,
    # RandomWalkKernelRd,
    CorrelatedGaussianNoiseRd,
    DriftingCorrelatedGaussianRandomWalkKernelRd,
)
from contract_checks import (
    check_metric_contract,
    # check_kernel_contracts,
    check_event_probabilities_monotonicity_additivity,
)
import seaborn as sns
import pandas as pd
import random
import math

if __name__ == "__main__":
    rng = random.Random(0)

    # --- Demo parameters (edit these) ---
    d = 3  # dimension of R^d
    p = 2.0  # set to 1<=p<=math.inf (math.inf for supremum norm) (for this range Minkowski inequality holds)
    step_std = 0.5
    init_std = 1.0
    n_steps = 200
    max_depth = 3
    stds = (0.5, 0.2, 1.0)
    corr = (
        (1.0, 0.3, -0.1),
        (0.3, 1.0, 0.25),
        (-0.1, 0.25, 1.0),
    )
    shift = 0.2

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

    base = CorrelatedGaussianNoiseRd(stds=stds, corr=corr)

    drift_scalar = 0.2
    drift = tuple(drift_scalar for _ in range(d))   # or any vector of length d

    kernel = DriftingCorrelatedGaussianRandomWalkKernelRd(noise=base, drift=drift)


    # Correlated Random walk in R^d
    mp = MarkovProcess(
        init=NormalRd(mean=origin, std=1.0),  # still ok as init
        kernel=kernel,
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

    check_event_probabilities_monotonicity_additivity(
        mp=mp,
        fam=fam,
        origin=origin,
        rng=rng,
        gens=gens,
        borel=borel,
        tol_prob=0.07,
        subset_trials=30,
        disjoint_trials=30,
        subset_n=3000,
        disjoint_n=3000,
        mc_n=10_000,  # samples per probability estimate (tune up/down)
    )

    # note following plot has nothing to do with the generated events

    # Plot time vs point_i for i = 1,...,d (shown as x1..xd)
    df = pd.DataFrame(path, columns=[f"x{i}" for i in range(1, d + 1)])
    df.insert(0, "t", range(len(path)))  # time column first

    df_long = df.melt(id_vars="t", var_name="coord", value_name="value")
    sns.lineplot(data=df_long, x="t", y="value", hue="coord")
