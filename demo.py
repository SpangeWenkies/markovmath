from core_interfaces import (
    PointRd,
    LpMetricRd,
    StdBorelSpaceRd,
    generate_event_family,
    MarkovProcess,
    NormalRd,
    RandomWalkKernelRd,
    CorrelatedGaussianNoiseRd,
    DriftingCorrelatedGaussianRandomWalkKernelRd,
    LaplaceRandomWalkKernelRd,
    StudentTRandomWalkKernelRd,
    UniformBallRandomWalkKernelRd,

)
from contract_checks import (
    check_metric_contract,
    check_discrete_chapman_kolmogorov,
    check_discrete_resolvent_identity,
    check_discrete_martingale,
    check_continuous_chapman_kolmogorov,
    estimate_prob,
    check_event_probabilities_monotonicity_additivity,
)
from helper_funcs import rd_key
from operators import (
    DiscreteSemigroup,
    DiscreteResolvent,
    ContinuousSemigroup,
    ContinuousResolvent,
    SampledGenerator,
    ClosedFormGenerator,
    ForwardEquation,
    StationaryDistributionSolver,
    FiniteStateCTMCAdjoint,
    ou_adjoint_1d,
)
from operators.test_functions import (
    coordinate,
    squared_norm,
    sin_frequency,
    payoff_call,
    payoff_put,
)
import seaborn as sns
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    rng = random.Random(0)
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save_fig(name: str) -> None:
        path = os.path.join(output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

    def series_from_path(path: list[PointRd]) -> list[float]:
        return [float(x[0]) for x in path]

    def normal_density(mean: float, std: float):
        var = std * std

        def density(x: float) -> float:
            return math.exp(-0.5 * ((x - mean) ** 2) / var) / math.sqrt(
                2.0 * math.pi * var
            )

        return density

    def animate_path_r2(path: list[PointRd], title: str, filename: str) -> None:
        df = pd.DataFrame(path, columns=["x", "y"])
        xs = df["x"].tolist()
        ys = df["y"].tolist()
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = 0.2
        frame_indices = list(range(2, len(xs) + 1, 4))
        fig, ax = plt.subplots(figsize=(5, 5))

        def update_r2(frame_idx: int):
            ax.cla()
            ax.plot(
                xs[:frame_idx],
                ys[:frame_idx],
                linewidth=1.2,
                color="tab:blue",
            )
            ax.scatter(xs[0], ys[0], c="green", s=20)
            ax.scatter(xs[frame_idx - 1], ys[frame_idx - 1], c="red", s=20)
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            return []

        gif_path = os.path.join(output_dir, filename)
        anim = animation.FuncAnimation(
            fig,
            update_r2,
            frames=frame_indices,
            interval=80,
            blit=False,
        )
        anim.save(gif_path, writer=animation.PillowWriter(fps=12))
        plt.close(fig)

    def animate_path_r3(path: list[PointRd], title: str, filename: str) -> None:
        df = pd.DataFrame(path, columns=["x", "y", "z"])
        xs = df["x"].tolist()
        ys = df["y"].tolist()
        zs = df["z"].tolist()
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        pad = 0.2
        frame_indices = list(range(2, len(xs) + 1, 4))
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

        def update_r3(frame_idx: int):
            ax.cla()
            ax.plot(
                xs[:frame_idx],
                ys[:frame_idx],
                zs[:frame_idx],
                linewidth=1.2,
                color="tab:blue",
            )
            ax.scatter(xs[0], ys[0], zs[0], c="green", s=20)
            ax.scatter(
                xs[frame_idx - 1],
                ys[frame_idx - 1],
                zs[frame_idx - 1],
                c="red",
                s=20,
            )
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_zlim(z_min - pad, z_max + pad)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=20, azim=35)
            return []

        gif_path = os.path.join(output_dir, filename)
        anim = animation.FuncAnimation(
            fig,
            update_r3,
            frames=frame_indices,
            interval=80,
            blit=False,
        )
        anim.save(gif_path, writer=animation.PillowWriter(fps=12))
        plt.close(fig)

    def animate_paths_r2(
        paths: dict[str, list[PointRd]], title: str, filename: str
    ) -> None:
        labels = list(paths.keys())
        coords = {
            label: (
                [float(p[0]) for p in path],
                [float(p[1]) for p in path],
            )
            for label, path in paths.items()
        }
        x_min = min(min(xs) for xs, _ys in coords.values())
        x_max = max(max(xs) for xs, _ys in coords.values())
        y_min = min(min(ys) for _xs, ys in coords.values())
        y_max = max(max(ys) for _xs, ys in coords.values())
        pad = 0.2
        min_len = min(len(path) for path in paths.values())
        frame_indices = list(range(2, min_len + 1, 4))
        colors = plt.cm.tab10(range(len(labels)))
        fig, ax = plt.subplots(figsize=(6, 5))

        def update_multi(frame_idx: int):
            ax.cla()
            for color, label in zip(colors, labels):
                xs, ys = coords[label]
                ax.plot(xs[:frame_idx], ys[:frame_idx], linewidth=1.2, color=color)
                ax.scatter(xs[0], ys[0], c=[color], s=18, marker="o")
                ax.scatter(
                    xs[frame_idx - 1],
                    ys[frame_idx - 1],
                    c=[color],
                    s=22,
                    marker="x",
                )
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(labels, loc="upper left", fontsize=8)
            ax.set_aspect("equal")
            return []

        gif_path = os.path.join(output_dir, filename)
        anim = animation.FuncAnimation(
            fig,
            update_multi,
            frames=frame_indices,
            interval=80,
            blit=False,
        )
        anim.save(gif_path, writer=animation.PillowWriter(fps=12))
        plt.close(fig)

    # --- Markov processes in R^1 ---
    origin_1d: PointRd = (0.0,)
    standard_kernel = RandomWalkKernelRd(step_std=1.0)
    standard_mp = MarkovProcess(
        init=NormalRd(mean=origin_1d, std=1.0), kernel=standard_kernel
    )

    nonstandard_kernel = RandomWalkKernelRd(step_std=0.4)
    nonstandard_mp = MarkovProcess(
        init=NormalRd(mean=origin_1d, std=2.0), kernel=nonstandard_kernel
    )

    drift_noise = CorrelatedGaussianNoiseRd(stds=(0.6,), corr=((1.0,),))
    drift_kernel = DriftingCorrelatedGaussianRandomWalkKernelRd(
        noise=drift_noise,
        drift=(0.12,),
    )
    drift_mp = MarkovProcess(
        init=NormalRd(mean=origin_1d, std=1.0), kernel=drift_kernel
    )

    shifted_mp = MarkovProcess(
        init=NormalRd(mean=(2.0,), std=0.8), kernel=nonstandard_kernel
    )

    n_steps_1d = 200
    paths_1d = {
        "standard": standard_mp.sample_path(n_steps_1d, rng=random.Random(1)),
        "nonstandard": nonstandard_mp.sample_path(n_steps_1d, rng=random.Random(2)),
        "drifted": drift_mp.sample_path(n_steps_1d, rng=random.Random(3)),
        "shifted": shifted_mp.sample_path(n_steps_1d, rng=random.Random(4)),
    }

    plt.figure(figsize=(9, 4))
    for label, path in paths_1d.items():
        plt.plot(series_from_path(path), label=label)
    plt.title("R^1 Markov process paths")
    plt.xlabel("Step")
    plt.ylabel("X_t")
    plt.legend()
    save_fig("markov_paths_r1.png")

    # --- Correlated processes in R^2 and R^3 ---
    corr_2d = ((1.0, 0.6), (0.6, 1.0))
    corr_noise_2d = CorrelatedGaussianNoiseRd(stds=(0.5, 0.8), corr=corr_2d)
    corr_kernel_2d = DriftingCorrelatedGaussianRandomWalkKernelRd(
        noise=corr_noise_2d, drift=(0.0, 0.0)
    )
    mp_2d = MarkovProcess(
        init=NormalRd(mean=(0.0, 0.0), std=0.5), kernel=corr_kernel_2d
    )
    path_2d = mp_2d.sample_path(300, rng=random.Random(5))
    df_2d = pd.DataFrame(path_2d, columns=["x", "y"])
    plt.figure(figsize=(5, 5))
    plt.plot(df_2d["x"], df_2d["y"], linewidth=1.2)
    plt.scatter(df_2d["x"].iloc[0], df_2d["y"].iloc[0], c="green", label="start")
    plt.scatter(df_2d["x"].iloc[-1], df_2d["y"].iloc[-1], c="red", label="end")
    plt.axis("equal")
    plt.title("Correlated random walk in R^2")
    plt.legend()
    save_fig("markov_paths_r2.png")

    animate_path_r2(
        path_2d,
        "Correlated random walk in R^2 (evolution)",
        "markov_paths_r2_evolution.gif",
    )

    corr_3d = (
        (1.0, 0.2, -0.1),
        (0.2, 1.0, 0.4),
        (-0.1, 0.4, 1.0),
    )
    corr_noise_3d = CorrelatedGaussianNoiseRd(stds=(0.5, 0.2, 1.0), corr=corr_3d)
    corr_kernel_3d = DriftingCorrelatedGaussianRandomWalkKernelRd(
        noise=corr_noise_3d, drift=(0.02, -0.01, 0.05)
    )

    mp_3d = MarkovProcess(
        init=NormalRd(mean=(0.0, 0.0, 0.0), std=0.4), kernel=corr_kernel_3d
    )
    path_3d = mp_3d.sample_path(250, rng=random.Random(6))
    df_3d = pd.DataFrame(path_3d, columns=["x", "y", "z"])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df_3d["x"], df_3d["y"], df_3d["z"], linewidth=1.0)
    ax.set_title("Correlated random walk in R^3")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    save_fig("markov_paths_r3.png")
    
    animate_path_r3(
        path_3d,
        "Correlated random walk in R^3 (evolution)",
        "markov_paths_r3_evolution.gif",
    )

    # --- Alternative random walk kernels ---
    alt_kernels = {
        "laplace": LaplaceRandomWalkKernelRd(scale=0.35),
        "student_t": StudentTRandomWalkKernelRd(df=4.0, scale=0.35),
        "uniform_ball": UniformBallRandomWalkKernelRd(radius=0.6),
    }
    alt_paths_2d = {}
    for label, kernel in alt_kernels.items():
        mp_alt_2d = MarkovProcess(
            init=NormalRd(mean=(0.0, 0.0), std=0.4),
            kernel=kernel,
        )
    
        alt_path_2d = mp_alt_2d.sample_path(240, rng=random.Random(20))
        alt_paths_2d[label] = alt_path_2d
        animate_path_r2(
            alt_path_2d,
            f"{label.replace('_', ' ').title()} random walk in R^2 (evolution)",
            f"{label}_random_walk_r2_evolution.gif",
        )

        mp_alt_3d = MarkovProcess(
            init=NormalRd(mean=(0.0, 0.0, 0.0), std=0.4),
            kernel=kernel,
        )
        alt_path_3d = mp_alt_3d.sample_path(220, rng=random.Random(21))
        animate_path_r3(
            alt_path_3d,
            f"{label.replace('_', ' ').title()} random walk in R^3 (evolution)",
            f"{label}_random_walk_r3_evolution.gif",
        )
    animate_paths_r2(
        alt_paths_2d,
        "Random walk kernels in R^2 (evolution)",
        "kernels_random_walk_r2_evolution.gif",
    )

    # --- LpMetricRd for p=1,2,inf ---
    p_values = [1.0, 2.0, math.inf]
    theta = [2.0 * math.pi * i / 200 for i in range(200)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, p_val in zip(axes, p_values):
        if p_val == math.inf:
            xs = [1, 1, -1, -1, 1]
            ys = [1, -1, -1, 1, 1]
        else:
            xs = [
                math.copysign(abs(math.cos(t)) ** (2.0 / p_val), math.cos(t))
                for t in theta
            ]
            ys = [
                math.copysign(abs(math.sin(t)) ** (2.0 / p_val), math.sin(t))
                for t in theta
            ]
        ax.plot(xs, ys)
        ax.set_title(f"Lp unit ball p={p_val}")
        ax.set_aspect("equal")
    save_fig("lp_metric_unit_balls.png")

    # --- generate_event_family demo ---
    d = 3
    p = 2.0
    max_depth = 3
    metric = LpMetricRd(p=p)
    borel = StdBorelSpaceRd(metric=metric)
    origin: PointRd = tuple(0.0 for _ in range(d))
    gens = [
        borel.ball(center=origin, radius=1.2 * math.sqrt(d)),
        borel.ball(center=tuple(0.6 for _ in range(d)), radius=1.1 * math.sqrt(d)),
        borel.ball(center=tuple(-0.6 for _ in range(d)), radius=1.1 * math.sqrt(d)),
    ]
    fam = generate_event_family(gens, max_depth=max_depth)
    sizes = []
    for depth in range(max_depth + 1):
        sizes.append((depth, len(generate_event_family(gens, max_depth=depth))))
    size_df = pd.DataFrame(sizes, columns=["depth", "count"])
    plt.figure(figsize=(5, 3))
    sns.barplot(data=size_df, x="depth", y="count")
    plt.title("Event family size vs depth")
    save_fig("event_family_sizes.png")

    print(f"Generated {len(fam)} events (R^{d}, p={p}, depth={max_depth}).")

    # --- Metric contract check ---

    check_metric_contract(
        metric,
        point_sampler=NormalRd(mean=origin, std=1.0),
        rng=random.Random(123),
        n_pairs=200,
        n_triples=200,
    )
    print("Metric contract check: OK (heuristic).")

    # --- Event probability checks ---
    mp_event = MarkovProcess(
        init=NormalRd(mean=origin, std=1.0),
        kernel=DriftingCorrelatedGaussianRandomWalkKernelRd(
            noise=corr_noise_3d, drift=(0.0, 0.0, 0.0)
        ),
    )

    check_event_probabilities_monotonicity_additivity(
        mp=mp_event,
        fam=fam,
        origin=origin,
        rng=random.Random(12085278),
        gens=gens,
        borel=borel,
        tol_prob=0.07,
        subset_trials=30,
        disjoint_trials=30,
        subset_n=3000,
        disjoint_n=3000,
        mc_n=10_000,
    )

    event_probs = [
        (f"event_{i}", estimate_prob(mp_event.init, ev, 5000, random.Random(100 + i)))
        for i, ev in enumerate(fam[:10])
    ]
    prob_df = pd.DataFrame(event_probs, columns=["event", "probability"])
    plt.figure(figsize=(6, 3))
    sns.barplot(data=prob_df, x="event", y="probability")
    plt.xticks(rotation=45, ha="right")
    plt.title("Sample event probabilities (init law)")
    save_fig("event_probabilities.png")

    # --- Test functions ---
    xs = [i / 10.0 for i in range(-40, 41)]
    test_df = pd.DataFrame(
        {
            "x": xs,
            "coordinate": [coordinate(0)((x,)) for x in xs],
            "squared_norm": [squared_norm(2.0)((x,)) for x in xs],
            "sin_freq": [sin_frequency([1.2])((x,)) for x in xs],
            "payoff_call": [payoff_call(1.0)((x,)) for x in xs],
            "payoff_put": [payoff_put(1.0)((x,)) for x in xs],
        }
    )
    plt.figure(figsize=(8, 4))
    for col in test_df.columns[1:]:
        plt.plot(test_df["x"], test_df[col], label=col)
    plt.legend()
    plt.title("Example test functions")
    save_fig("test_functions.png")

    # --- Discrete semigroup and resolvent estimates ---
    semigroup = DiscreteSemigroup(standard_kernel, key_fn=rd_key, cache={})
    resolvent = DiscreteResolvent(standard_kernel, lam=0.9, key_fn=rd_key, cache={})
    test_funcs = {
        "coord": coordinate(0),
        "square": squared_norm(2.0),
        "sin": sin_frequency([1.0]),
    }

    n_grid = list(range(0, 11))
    semi_rows = []
    for name, fn in test_funcs.items():
        for n in n_grid:
            semi_rows.append(
                {
                    "n": n,
                    "value": semigroup.estimate_Tn(
                        fn, origin_1d, n=n, n_samples=800, seed=1000 + n
                    ),
                    "function": name,
                }
            )
    semi_df = pd.DataFrame(semi_rows)
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=semi_df, x="n", y="value", hue="function", marker="o")
    plt.title("Discrete semigroup estimates T^n f(x0)")
    save_fig("discrete_semigroup.png")

    resolvent_lams = [0.6, 0.8, 0.9]
    res_rows = []
    for lam in resolvent_lams:
        res = DiscreteResolvent(standard_kernel, lam=lam)
        for name, fn in test_funcs.items():
            res_rows.append(
                {
                    "lambda": lam,
                    "value": res.estimate_U(fn, origin_1d, n_paths=800, seed=2000),
                    "function": name,
                }
            )
    res_df = pd.DataFrame(res_rows)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=res_df, x="lambda", y="value", hue="function")
    plt.title("Discrete resolvent estimates U_λ f(x0)")
    save_fig("discrete_resolvent.png")

    # --- Continuous semigroup and resolvent ---
    dt = 0.2
    cont_semigroup = ContinuousSemigroup(standard_kernel, dt=dt)
    # cont_resolvent = ContinuousResolvent(standard_kernel, dt=dt, alpha=1.0)
    t_grid = [0.0, 0.4, 0.8, 1.2, 1.6]
    cont_rows = []
    for name, fn in test_funcs.items():
        for t in t_grid:
            cont_rows.append(
                {
                    "t": t,
                    "value": cont_semigroup.estimate_Pt(
                        fn, origin_1d, t=t, n_samples=800, seed=3000 + int(t * 10)
                    ),
                    "function": name,
                }
            )
    cont_df = pd.DataFrame(cont_rows)
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=cont_df, x="t", y="value", hue="function", marker="o")
    plt.title("Continuous semigroup estimates P_t f(x0)")
    save_fig("continuous_semigroup.png")

    alpha_grid = [0.6, 1.0, 1.4]
    cont_res_rows = []
    for alpha in alpha_grid:
        res = ContinuousResolvent(standard_kernel, dt=dt, alpha=alpha)
        for name, fn in test_funcs.items():
            cont_res_rows.append(
                {
                    "alpha": alpha,
                    "value": res.estimate_Ralpha(fn, origin_1d, n_paths=800, seed=4000),
                    "function": name,
                }
            )
    cont_res_df = pd.DataFrame(cont_res_rows)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=cont_res_df, x="alpha", y="value", hue="function")
    plt.title("Continuous resolvent estimates R_α f(x0)")
    save_fig("continuous_resolvent.png")

    # --- Discrete vs continuous resolvent (common test function) ---
    compare_rows = []
    f_common = squared_norm(2.0)
    for alpha in [0.4, 0.8, 1.2, 1.6]:
        cont = ContinuousResolvent(standard_kernel, dt=dt, alpha=alpha)
        disc = DiscreteResolvent(standard_kernel, lam=math.exp(-alpha * dt))
        compare_rows.append(
            {
                "alpha": alpha,
                "value": cont.estimate_Ralpha(
                    f_common, origin_1d, n_paths=800, seed=5000
                ),
                "type": "continuous",
            }
        )
        compare_rows.append(
            {
                "alpha": alpha,
                "value": disc.estimate_U(f_common, origin_1d, n_paths=800, seed=5000),
                "type": "discrete",
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=compare_df, x="alpha", y="value", hue="type", marker="o")
    plt.title("Discrete vs continuous resolvent (common f)")
    save_fig("resolvent_compare.png")

    # --- Discrete vs continuous semigroup (common test function) ---
    sem_rows = []
    for t in t_grid:
        n = int(round(t / dt))
        sem_rows.append(
            {
                "t": t,
                "value": semigroup.estimate_Tn(
                    f_common, origin_1d, n=n, n_samples=800, seed=6000
                ),
                "type": "discrete",
            }
        )
        sem_rows.append(
            {
                "t": t,
                "value": cont_semigroup.estimate_Pt(
                    f_common, origin_1d, t=t, n_samples=800, seed=6000
                ),
                "type": "continuous",
            }
        )
    sem_df = pd.DataFrame(sem_rows)
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=sem_df, x="t", y="value", hue="type", marker="o")
    plt.title("Discrete vs continuous semigroup (common f)")
    save_fig("semigroup_compare.png")

    # --- Truncated vs geometric horizon (resolvent) ---
    trunc_vals = []
    geo_est = resolvent.estimate_U(f_common, origin_1d, n_paths=1000, seed=7000)
    for K in [1, 2, 4, 8, 16, 24]:
        trunc_vals.append(
            {
                "K": K,
                "value": resolvent.estimate_U_truncated(
                    f_common, origin_1d, K=K, n_paths=1000, seed=8000 + K
                ),
                "type": "truncated",
            }
        )
    trunc_df = pd.DataFrame(trunc_vals)
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=trunc_df, x="K", y="value", marker="o", label="truncated")
    plt.axhline(geo_est, color="red", linestyle="--", label="geometric")
    plt.title("Resolvent: truncated vs geometric horizon")
    plt.legend()
    save_fig("resolvent_trunc_vs_geo.png")

    # --- Chapman-Kolmogorov checks (semigroup) ---
    ck_rows = []
    for n in [1, 2, 4, 6, 8]:
        lhs, rhs = check_discrete_chapman_kolmogorov(
            semigroup,
            f_common,
            origin_1d,
            m=n,
            n=n,
            n_outer=300,
            n_inner=300,
            rng=random.Random(9000 + n),
        )
        ck_rows.append({"n": n, "diff": abs(lhs - rhs)})
    ck_df = pd.DataFrame(ck_rows)
    plt.figure(figsize=(6, 3))
    sns.barplot(data=ck_df, x="n", y="diff")
    plt.title("Chapman–Kolmogorov diff |T^{2n}f - T^n(T^n f)|")
    save_fig("chapman_kolmogorov.png")

    # --- Resolvent identity check ---
    lhs, rhs = check_discrete_resolvent_identity(
        resolvent,
        semigroup,
        f_common,
        origin_1d,
        n_outer=200,
        n_inner=200,
        rng=random.Random(9100),
    )
    plt.figure(figsize=(4, 3))
    sns.barplot(x=["lhs", "rhs"], y=[lhs, rhs])
    plt.title("Resolvent identity check")
    save_fig("resolvent_identity.png")

    # --- Martingale check ---
    generator = SampledGenerator(semigroup=semigroup, dt=1.0)

    def Af_fn(x):
        return generator.estimate_Af(f_common, x, n_samples=500, seed=9200)

    mn_estimate = check_discrete_martingale(
        standard_kernel,
        Af=Af_fn,
        f=f_common,
        x0=origin_1d,
        n_steps=20,
        n_paths=400,
        rng=random.Random(9300),
    )
    print(f"Martingale check E[M_n]≈{mn_estimate:.4f}")

    m_values = []
    for _ in range(400):
        x = origin_1d
        sum_af = 0.0
        for _k in range(20):
            sum_af += float(Af_fn(x))
            x = standard_kernel.law(x).sample(rng)
        m_values.append(float(f_common(x)) - float(f_common(origin_1d)) - sum_af)
    plt.figure(figsize=(6, 3))
    sns.histplot(m_values, bins=30, kde=True)
    plt.title("Martingale increments M_n histogram")
    save_fig("martingale_hist.png")

    # --- Forward equation demo (density evolution) ---
    class GaussianDensityEvolution:
        def __init__(self, mean0: float, std0: float, step_std: float) -> None:
            self.mean0 = mean0
            self.std0 = std0
            self.step_std = step_std

        def evolve_density(self, _p0, t: float):
            std_t = math.sqrt(self.std0**2 + t * self.step_std**2)
            return normal_density(self.mean0, std_t)

    forward = ForwardEquation(density_solver=GaussianDensityEvolution(0.0, 1.0, 1.0))
    xs = [i / 10.0 for i in range(-50, 51)]
    density_rows = []
    for t in [0.0, 0.5, 1.0, 2.0]:
        density = forward.forward_density_step(lambda _x: 0.0, t)
        density_rows.append(
            {
                "t": t,
                "values": [density(x) for x in xs],
            }
        )
    plt.figure(figsize=(7, 4))
    for row in density_rows:
        plt.plot(xs, row["values"], label=f"t={row['t']}")
    plt.legend()
    plt.title("Forward equation density evolution")
    save_fig("forward_equation.png")

    # --- Generator types and Af values ---
    ou_generator = ClosedFormGenerator(
        drift=lambda x: (-0.4 * x[0],),
        diffusion=lambda _x: ((0.6**2,),),
        fd_step=1e-3,
    )
    xs_grid = [(x,) for x in xs]
    gen_rows = []
    for name, fn in {"coord": coordinate(0), "square": squared_norm(2.0)}.items():
        for x in xs_grid:
            gen_rows.append(
                {
                    "x": x[0],
                    "value": generator.estimate_Af(fn, x, n_samples=400, seed=9400),
                    "function": name,
                    "generator": "sampled",
                }
            )
            gen_rows.append(
                {
                    "x": x[0],
                    "value": ou_generator.estimate_Af(fn, x),
                    "function": name,
                    "generator": "closed_form",
                }
            )
    gen_df = pd.DataFrame(gen_rows)
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=gen_df,
        x="x",
        y="value",
        hue="function",
        style="generator",
    )
    plt.title("Generator outputs Af for common test functions")
    save_fig("generator_outputs.png")

    # --- Adjoint generators ---
    adjoint = ou_adjoint_1d(kappa=0.4, theta=0.0, sigma=0.6)
    base_density = normal_density(0.0, 1.0)
    adjoint_density = adjoint.apply_to_density(base_density)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [base_density(x) for x in xs], label="p(x)")
    plt.plot(xs, [adjoint_density(x) for x in xs], label="A* p(x)")
    plt.legend()
    plt.title("Adjoint applied to a density (OU example)")
    save_fig("adjoint_density.png")

    states = ["A", "B", "C"]
    rate_matrix = [
        [-0.6, 0.4, 0.2],
        [0.1, -0.5, 0.4],
        [0.2, 0.3, -0.5],
    ]
    adjoint_ctmc = FiniteStateCTMCAdjoint(states=states, rate_matrix=rate_matrix)
    adjoint_matrix = pd.DataFrame(adjoint_ctmc._rates, index=states, columns=states)
    plt.figure(figsize=(4, 3))
    sns.heatmap(adjoint_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("CTMC adjoint rate matrix")
    save_fig("adjoint_ctmc.png")

    # --- Stationary distribution solver ---
    solver = StationaryDistributionSolver(adjoint=adjoint_ctmc, states=states, dt=0.1)
    p0 = [0.7, 0.2, 0.1]
    stationary_trunc = solver.solve_truncated(p0, lam=0.9, n_steps=50)
    stationary_geo = solver.solve_geometric(p0, lam=0.9, seed=9500)
    stat_df = pd.DataFrame(
        {
            "state": states,
            "truncated": stationary_trunc,
            "geometric": stationary_geo,
        }
    )
    stat_long = stat_df.melt(id_vars="state", var_name="method", value_name="prob")
    plt.figure(figsize=(5, 3))
    sns.barplot(data=stat_long, x="state", y="prob", hue="method")
    plt.title("StationaryDistributionSolver outputs")
    save_fig("stationary_solver.png")

    # --- Continuous Chapman-Kolmogorov check ---
    lhs, rhs = check_continuous_chapman_kolmogorov(
        cont_semigroup,
        f_common,
        origin_1d,
        s=0.4,
        t=0.8,
        n_outer=200,
        n_inner=200,
        rng=random.Random(9600),
    )
    plt.figure(figsize=(4, 3))
    sns.barplot(x=["lhs", "rhs"], y=[lhs, rhs])
    plt.title("Continuous Chapman–Kolmogorov check")
    save_fig("continuous_ck.png")

    print(f"Demo outputs saved in {output_dir}/")
