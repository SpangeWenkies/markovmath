from core_interfaces import (
    PointRd, 
    LpMetricRd, 
    StdBorelSpaceRd, 
    generate_event_family, 
    MarkovProcess, 
    NormalRd, 
    RandomWalkKernelRd
)
from contract_checks import (
    check_metric_contract_in_R,
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

    # following function must be made R^d

    # check_metric_contract_in_R(
    #     metric,
    #     point_sampler=mp.init,
    #     rng=random.Random(123),
    #     n_pairs=n_pairs,
    #     n_triples=n_pairs,
    # )
    # print("Metric contract check: OK (heuristic).")

    # Plot time vs point_i for i = 1,...,d (shown as x1..xd)
    df = pd.DataFrame(path, columns=[f"x{i}" for i in range(1, d + 1)])
    df.insert(0, "t", range(len(path)))  # time column first

    df_long = df.melt(id_vars="t", var_name="coord", value_name="value")
    sns.lineplot(data=df_long, x="t", y="value", hue="coord")