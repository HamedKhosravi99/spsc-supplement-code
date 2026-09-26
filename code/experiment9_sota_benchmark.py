"""
Experiment 9: Comparison with State-of-the-Art on Russac et al. Benchmark

Reproduces the non-stationary linear bandit benchmark of Russac et al.
(NeurIPS 2019) using our synthetic environment: d=2, r=1, K=4, T=6000,
sigma_eps=1, 50 arms, 30 seeds.

Compares SPSC against D-LinUCB, SW-LinUCB, OFUL, Oracle-LinUCB, and
Reset-LinUCB (with oracle change-point knowledge).

Outputs
-------
  experiment9_sota_benchmark.png  — 2-panel figure
    (a) Cumulative regret curves
    (b) Final regret bar chart
"""

import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply(scale=0.9)          # a size smaller than the other appendix figures

from environments import LowRankLDSEnvironment
from algorithm import SPSC_Algorithm1, LinUCB, OracleLinUCB, RunMetrics, SWLinUCB, RestartLinUCB, OracleResetLinUCB

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Parameters (Russac et al. benchmark setup)
# ---------------------------------------------------------------------------

D           = 2
R           = 1
K           = 4
T           = 6000
SIGMA_EPS   = 1.0
PROBE_EVERY = 30
PROBE_COST  = 0.1
WINDOW      = 100
N_ACTIONS   = 50
N_SEEDS     = 30

NAMES = ["SPSC-Alg1", "Oracle-LinUCB", "D-LinUCB", "SW-LinUCB",
         "OFUL", "Reset-LinUCB"]

COLORS = {
    "SPSC-Alg1":     "#1f77b4",
    "Oracle-LinUCB": "#2ca02c",
    "D-LinUCB":      "#ff7f0e",
    "SW-LinUCB":     "#9467bd",
    "OFUL":          "#d62728",
    "Reset-LinUCB":  "#8c564b",
}
LABELS = {
    "SPSC-Alg1":     "SPSC (Alg. 1)",
    "Oracle-LinUCB": "Oracle",
    "D-LinUCB":      "D-LinUCB",
    "SW-LinUCB":     "SW-LinUCB",
    "OFUL":          "OFUL",
    "Reset-LinUCB":  "Reset-LinUCB",
}
STYLES = {
    "SPSC-Alg1":     "-",
    "Oracle-LinUCB": ":",
    "D-LinUCB":      "--",
    "SW-LinUCB":     "-.",
    "OFUL":          "--",
    "Reset-LinUCB":  ":",
}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def make_env(seed):
    return LowRankLDSEnvironment(d=D, r=R, K=K, T=T,
                                  sigma_eps=SIGMA_EPS,
                                  n_actions=N_ACTIONS,
                                  seed=seed * 100)


def run_all(n_seeds):
    results = {n: [] for n in NAMES}

    for seed in range(n_seeds):
        print(f"  seed {seed+1}/{n_seeds} ...", end="\r", flush=True)

        env = make_env(seed)
        results["SPSC-Alg1"].append(
            SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                            window=WINDOW, lam=1.0, delta=0.05, seed=seed).run()
        )

        env = make_env(seed)
        results["Oracle-LinUCB"].append(
            OracleLinUCB(env, window=WINDOW, lam=1.0, delta=0.05,
                         seed=seed + 1000).run()
        )

        # D-LinUCB (discounted ridge, gamma=0.998)
        env = make_env(seed)
        results["D-LinUCB"].append(
            LinUCB(env, lam=1.0, delta=0.05, seed=seed + 2000,
                   forgetting_factor=0.998).run()
        )

        # SW-LinUCB (sliding window W=200)
        env = make_env(seed)
        results["SW-LinUCB"].append(
            SWLinUCB(env, window=200, lam=1.0, delta=0.05,
                     seed=seed + 3000).run()
        )

        # OFUL (standard LinUCB, no forgetting, no window)
        env = make_env(seed)
        results["OFUL"].append(
            LinUCB(env, lam=1.0, delta=0.05, seed=seed + 4000).run()
        )

        # Reset-LinUCB (oracle segment boundaries)
        env = make_env(seed)
        results["Reset-LinUCB"].append(
            OracleResetLinUCB(env, lam=1.0, delta=0.05,
                              seed=seed + 5000).run()
        )

    print(flush=True)
    return results


def agg(runs, attr):
    data = np.stack([getattr(r, attr) for r in runs])
    mean = data.mean(axis=0)
    se   = data.std(axis=0) / np.sqrt(len(runs))
    return mean, se


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, env_ref, out_path):
    t_axis = np.arange(1, T + 1)
    change_pts = env_ref.tau[1:]

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(2), gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel (a): Cumulative regret curves
    ax = axes[0]
    for name in NAMES:
        mean, se = agg(results[name], "cumulative_control_regret")
        ax.plot(t_axis, mean, color=COLORS[name], ls=STYLES[name], lw=1.2,
                label=LABELS[name], zorder=3)
        ax.fill_between(t_axis, mean - se, mean + se,
                        color=COLORS[name], alpha=0.12, zorder=2)
    for cp in change_pts:
        ax.axvline(cp, color="gray", ls=":", lw=1.0, alpha=0.65)
    ax.set_xlabel("round $t$")
    ax.set_ylabel("cumulative control regret")
    figstyle.panel(ax, "(a)")
    ax.set_xlim(1, T); ax.set_ylim(bottom=0)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(0, hi_ * 1.5)       # headroom for the legend
    ax.legend(loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.5, fontsize=7)

    # Panel (b): Final regret bar chart
    ax = axes[1]
    final_means, final_ses = [], []
    for name in NAMES:
        finals = np.array([r.cumulative_control_regret[-1] for r in results[name]])
        final_means.append(finals.mean())
        final_ses.append(finals.std() / np.sqrt(len(finals)))

    x = np.arange(len(NAMES))
    bars = ax.bar(x, final_means, yerr=final_ses,
                  color=[COLORS[n] for n in NAMES], capsize=2)
    ax.set_ylim(0, max(np.array(final_means) + np.array(final_ses)) * 1.15)   # room for the value labels
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n].split("(")[0].strip() for n in NAMES],
                       rotation=25, ha="right")
    ax.set_ylabel("final control regret")
    figstyle.panel(ax, "(b)")
    ax.yaxis.grid(True, alpha=0.3)

    # Annotate bars
    for i, (m, s) in enumerate(zip(final_means, final_ses)):
        ax.text(i, m + s + 50, f"{m:.0f}", ha="center", va="bottom", fontsize=6.5)

    figstyle.no_title(
        f"Experiment 9: SOTA Benchmark (Russac et al. NeurIPS 2019)  |  "
        f"$d={D}$, $r={R}$, $K={K}$, $T={T}$, $\\sigma_\\varepsilon={SIGMA_EPS}$, "
        f"{N_ACTIONS} arms  "
        f"($n={N_SEEDS}$ seeds)", y=1.02,
    )
    figstyle.save(out_path)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(results):
    print()
    print("=" * 80)
    print("Experiment 9 — SOTA Benchmark (Russac et al. NeurIPS 2019)")
    print(f"d={D}, r={R}, K={K}, T={T}, sigma_eps={SIGMA_EPS}, "
          f"n_actions={N_ACTIONS}, n_seeds={N_SEEDS}")
    print("-" * 80)
    header = (f"{'Algorithm':<28}  {'Control (mean±SE)':>18}  "
              f"{'Costed (mean±SE)':>18}  {'vs D-LinUCB':>12}")
    print(header)
    print("-" * 80)

    dlinucb_ctrl = np.array([r.cumulative_control_regret[-1]
                             for r in results["D-LinUCB"]]).mean()

    for name in NAMES:
        ctrl = np.array([r.cumulative_control_regret[-1] for r in results[name]])
        cost = np.array([r.cumulative_costed_regret[-1] for r in results[name]])
        n = len(ctrl)
        pct = (ctrl.mean() - dlinucb_ctrl) / dlinucb_ctrl * 100
        sign = "+" if pct >= 0 else ""
        vs = f"{sign}{pct:.1f}%" if name != "D-LinUCB" else "---"
        print(f"  {LABELS[name]:<26}  {ctrl.mean():>7.0f} ± {ctrl.std()/np.sqrt(n):>4.0f}    "
              f"{cost.mean():>7.0f} ± {cost.std()/np.sqrt(n):>4.0f}    {vs:>12}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 9: SOTA Benchmark")
    print(f"  d={D}, r={R}, K={K}, T={T}, sigma_eps={SIGMA_EPS}")
    print(f"  n_actions={N_ACTIONS}, n_seeds={N_SEEDS}")
    print("=" * 60)

    env_ref = make_env(0)
    print("\nRunning seeds...")
    results = figstyle.cached("exp9_sota_benchmark", lambda: run_all(N_SEEDS))
    print_table(results)
    make_figure(results, env_ref,
                os.path.join(OUT_DIR, "experiment9_sota_benchmark.png"))
    print("\nDone.")
