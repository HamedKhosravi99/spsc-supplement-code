"""
Experiment 4: The method adapts after change points.

Shows instantaneous (per-round) regret smoothed over a rolling window,
highlighting that SPSC recovers faster than ambient LinUCB after each
segment boundary where the true subspace changes.

Also shows the subspace error trajectory for SPSC across the full horizon,
making the re-learning behavior visible.

Two panels:
  (a) Smoothed instantaneous control regret vs time — SPSC vs LinUCB vs Oracle
  (b) SPSC subspace error ||P_hat_k - P_k*||_2 vs time (probe rounds only)

Outputs
-------
  experiment4_changepoint_recovery.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

from environments import LowRankLDSEnvironment
from algorithm import SPSC_Algorithm1, LinUCB, OracleLinUCB

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

D           = 4
R           = 1
K           = 4
T           = 6000
PROBE_EVERY = 30
PROBE_COST  = 0.1
WINDOW      = 100
N_SEEDS     = 10
SMOOTH_W    = 60       # smoothing window for instantaneous regret

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "SPSC-Alg1":     "#1f77b4",
    "LinUCB":        "#d62728",
    "Oracle-LinUCB": "#2ca02c",
}
STYLES = {
    "SPSC-Alg1":     "-",
    "LinUCB":        "--",
    "Oracle-LinUCB": ":",
}
LABELS = {
    "SPSC-Alg1":     "SPSC (Alg. 1)",
    "LinUCB":        "LinUCB",
    "Oracle-LinUCB": "Oracle",
}


def make_env(seed):
    return LowRankLDSEnvironment(d=D, r=R, K=K, T=T, seed=seed * 100)


def smooth(x, w):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_all(n_seeds):
    names   = ["SPSC-Alg1", "LinUCB", "Oracle-LinUCB"]
    results = {n: [] for n in names}

    for seed in range(n_seeds):
        print(f"  seed {seed+1}/{n_seeds}", end="\r", flush=True)

        env = make_env(seed)
        results["SPSC-Alg1"].append(
            SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                            window=WINDOW, lam=1.0, delta=0.05, seed=seed).run()
        )
        env = make_env(seed)
        results["LinUCB"].append(
            LinUCB(env, lam=1.0, delta=0.05, seed=seed + 1000).run()
        )
        env = make_env(seed)
        results["Oracle-LinUCB"].append(
            OracleLinUCB(env, window=WINDOW, lam=1.0, delta=0.05,
                         seed=seed + 2000).run()
        )

    print(flush=True)
    return results


# ---------------------------------------------------------------------------
# Recovery-window analysis
# ---------------------------------------------------------------------------

def recovery_stats(results, env_ref, post_cp_window=200):
    """
    For each change point (segment k >= 1), compute average control regret
    in the first `post_cp_window` rounds of the new segment.
    Returns dict: name -> array of shape (K-1,) mean regret per change point.
    """
    stats = {}
    for name, runs in results.items():
        per_cp = []
        for k in range(1, K):
            cp = env_ref.tau[k]
            end = min(cp + post_cp_window, T)
            vals = np.stack(
                [r.control_regret[cp:end] for r in runs]
            ).mean(axis=0)
            per_cp.append(vals.mean())
        stats[name] = np.array(per_cp)
    return stats


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, env_ref, out_path):
    t_axis    = np.arange(1, T + 1)
    change_pts = env_ref.tau[1:]

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(2))

    # ---- Panel (a): smoothed instantaneous control regret ----
    ax = axes[0]
    for name, runs in results.items():
        mean_inst = np.stack([r.control_regret for r in runs]).mean(axis=0)
        se_inst   = np.stack([r.control_regret for r in runs]).std(axis=0) / np.sqrt(N_SEEDS)
        sm_mean   = smooth(mean_inst, SMOOTH_W)
        sm_se     = smooth(se_inst,   SMOOTH_W)
        ax.plot(t_axis, sm_mean,
                color=COLORS[name], ls=STYLES[name], lw=1.2,
                label=LABELS[name], zorder=3)
        ax.fill_between(t_axis, sm_mean - sm_se, sm_mean + sm_se,
                        color=COLORS[name], alpha=0.18, zorder=2)

    ymax = ax.get_ylim()[1]
    for cp in change_pts:
        ax.axvline(cp, color="gray", ls=":", lw=0.8, alpha=0.7, zorder=1)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel("regret per round")
    figstyle.panel(ax, "(a)")
    ax.set_ylim(0, ymax * 1.55)                     # headroom for the legend
    ax.legend(loc="upper right")
    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)

    # ---- Panel (b): SPSC subspace error trajectory ----
    ax = axes[1]
    spsc_runs = results["SPSC-Alg1"]

    # Mean subspace error at probe rounds (interpolated for display)
    all_errs = np.full((N_SEEDS, T), np.nan)
    for i, run in enumerate(spsc_runs):
        all_errs[i] = run.subspace_error

    # Forward-fill subspace error (constant between probes) for continuous display
    for i in range(N_SEEDS):
        last = np.nan
        for t in range(T):
            if not np.isnan(all_errs[i, t]):
                last = all_errs[i, t]
            else:
                all_errs[i, t] = last

    mean_sub = np.nanmean(all_errs, axis=0)
    se_sub   = np.nanstd(all_errs, axis=0) / np.sqrt(N_SEEDS)

    ax.plot(t_axis, mean_sub, color=COLORS["SPSC-Alg1"], lw=1.1,
            label="Mean subspace error", zorder=3)
    ax.fill_between(t_axis, mean_sub - se_sub, mean_sub + se_sub,
                    color=COLORS["SPSC-Alg1"], alpha=0.20, zorder=2)

    # Mark change points — error resets to ~1 (new random subspace)
    ymax2 = ax.get_ylim()[1]
    for cp in change_pts:
        ax.axvline(cp, color="gray", ls=":", lw=0.8, alpha=0.7, zorder=1)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"subspace error $\|\widehat{P}_k - P_k^\star\|_2$")
    figstyle.panel(ax, "(b)")

    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)

    figstyle.no_title(
        f"Experiment 4: Change-Point Adaptation  |  "
        f"$d={D}$, $r={R}$, $K={K}$, $T={T}$, probe every {PROBE_EVERY}  "
        f"({N_SEEDS} seeds, bands $= \\pm 1$ SE)", y=1.02,
    )

    figstyle.save(out_path)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(results, env_ref):
    n = N_SEEDS
    post_cp = 200

    print()
    print("=" * 72)
    print(f"Experiment 4 — Avg control regret in first {post_cp} rounds after each change point")
    print(f"  d={D}, r={R}, K={K}, T={T}, probe_every={PROBE_EVERY}")
    print("-" * 72)
    header = f"  {'Algorithm':<22}"
    for k in range(1, K):
        header += f"  {'CP '+str(k)+' (t='+str(env_ref.tau[k])+')':>18}"
    header += f"  {'Mean':>10}"
    print(header)
    print("-" * 72)

    rec = recovery_stats(results, env_ref, post_cp_window=post_cp)
    for name in ["SPSC-Alg1", "LinUCB", "Oracle-LinUCB"]:
        row = f"  {name:<22}"
        for v in rec[name]:
            row += f"  {v:>18.4f}"
        row += f"  {rec[name].mean():>10.4f}"
        print(row)
    print("-" * 72)

    # Speedup: LinUCB / SPSC mean post-CP regret
    spsc_mean = rec["SPSC-Alg1"].mean()
    lin_mean  = rec["LinUCB"].mean()
    print(f"  SPSC recovery advantage vs LinUCB: "
          f"{lin_mean:.4f} / {spsc_mean:.4f} = {lin_mean/spsc_mean:.2f}x")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env_ref = make_env(0)

    print("=" * 60)
    print("Experiment 4: Change-Point Adaptation")
    print(f"  d={D}, r={R}, K={K}, T={T}")
    print(f"  probe_every={PROBE_EVERY}, smooth_w={SMOOTH_W}")
    print(f"  n_seeds={N_SEEDS}")
    print("=" * 60)

    print("\nRunning seeds...")
    results = figstyle.cached("exp4_changepoint_recovery", lambda: run_all(N_SEEDS))

    print_table(results, env_ref)

    make_figure(
        results, env_ref,
        out_path=os.path.join(OUT_DIR, "experiment4_changepoint_recovery.png"),
    )

    print("\nDone.")
