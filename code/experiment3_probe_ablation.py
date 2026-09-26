"""
Experiment 3: Better subspace recovery leads to better control.

Sweeps probe_every in {5, 10, 20, 30, 50, 100, 300} and records:
  - final cumulative costed regret
  - final cumulative control regret
  - average subspace error in the second half of each segment
    (once the subspace estimate has warmed up)

Two panels:
  (a) Regret vs probe rate (probes per round = 1/probe_every)
  (b) Control regret vs subspace error — shows the direct mechanistic link

Outputs
-------
  experiment3_probe_ablation.png
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
from algorithm import SPSC_Algorithm1

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

D           = 4
R           = 1
K           = 4
T           = 6000
PROBE_COST  = 0.1
WINDOW      = 100
N_SEEDS     = 10

# Probe schedules to sweep; label for display
PROBE_EVERY_VALS = [5, 10, 20, 30, 50, 100, 300]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(seed):
    return LowRankLDSEnvironment(d=D, r=R, K=K, T=T, seed=seed * 100)


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

def run_sweep(probe_every_vals, n_seeds):
    """
    Returns dict: probe_every -> list of RunMetrics.
    """
    results = {}
    total = len(probe_every_vals) * n_seeds
    done  = 0

    for pe in probe_every_vals:
        runs = []
        for seed in range(n_seeds):
            done += 1
            print(f"  [{done}/{total}] probe_every={pe}, seed={seed+1}", end="\r", flush=True)
            env = make_env(seed)
            run = SPSC_Algorithm1(env, probe_every=pe, probe_cost=PROBE_COST,
                                  window=WINDOW, lam=1.0, delta=0.05, seed=seed).run()
            runs.append(run)
        results[pe] = runs

    print(flush=True)
    return results


def avg_late_subspace_error(runs, env_ref, fraction=0.5):
    """
    Average subspace error over probe rounds in the *second half* of each segment,
    where the estimator has had time to warm up.
    """
    errors = []
    for run in runs:
        for k in range(K):
            seg_start = env_ref.tau[k]
            seg_len   = env_ref.segment_lengths[k]
            late_start = seg_start + int(seg_len * fraction)
            for t in range(late_start, seg_start + seg_len):
                if run.probe_flags[t] and not np.isnan(run.subspace_error[t]):
                    errors.append(run.subspace_error[t])
    return float(np.mean(errors)) if errors else np.nan


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(sweep_results, env_ref, out_path):
    n = N_SEEDS
    probe_every_vals = sorted(sweep_results.keys())
    probe_rates      = [1.0 / pe for pe in probe_every_vals]  # probes per round

    costed_mean, costed_se     = [], []
    control_mean, control_se   = [], []
    sub_err_mean = []

    for pe in probe_every_vals:
        runs = sweep_results[pe]
        costed  = np.array([r.cumulative_costed_regret[-1]  for r in runs])
        control = np.array([r.cumulative_control_regret[-1] for r in runs])
        costed_mean.append(costed.mean());  costed_se.append(costed.std()/np.sqrt(n))
        control_mean.append(control.mean()); control_se.append(control.std()/np.sqrt(n))
        sub_err_mean.append(avg_late_subspace_error(runs, env_ref))

    costed_mean  = np.array(costed_mean)
    costed_se    = np.array(costed_se)
    control_mean = np.array(control_mean)
    control_se   = np.array(control_se)
    sub_err_mean = np.array(sub_err_mean)
    probe_rates  = np.array(probe_rates)

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(2))

    # ---- Panel (a): regret vs probe period ----
    periods = np.array(probe_every_vals)
    ax = axes[0]
    ax.errorbar(periods, costed_mean, yerr=costed_se,
                fmt='o-', color="#1f77b4", capsize=2, label="costed regret")
    ax.errorbar(periods, control_mean, yerr=control_se,
                fmt='s--', color="#ff7f0e", capsize=2, label="control regret")
    ax.set_xscale("log")
    ax.set_xticks(periods)
    ax.set_xticklabels([str(pe) for pe in periods])
    ax.minorticks_off()
    ax.set_xlabel("probe period (rounds between probes)")
    ax.set_ylabel("final regret")
    figstyle.panel(ax, "(a)")
    ax.legend(loc="upper left")

    # ---- Panel (b): control regret vs subspace error, each point labelled by its probe period ----
    ax = axes[1]
    valid = ~np.isnan(sub_err_mean)
    ax.errorbar(sub_err_mean[valid], control_mean[valid], yerr=control_se[valid],
                fmt='none', lw=0.9, color="gray", capsize=2, zorder=2)
    ax.scatter(sub_err_mean[valid], control_mean[valid], c=np.log(periods[valid]),
               cmap="viridis", s=36, zorder=3, edgecolors="black", linewidths=0.5)
    offsets = {5: (6, 2), 10: (-6, -12), 20: (5, -12), 30: (6, 2), 50: (6, 2), 100: (6, 2), 300: (-8, -12)}
    for i, pe in enumerate(periods[valid]):
        ax.annotate(f"{pe}", xy=(sub_err_mean[valid][i], control_mean[valid][i]),
                    xytext=offsets.get(int(pe), (6, 2)), textcoords="offset points",
                    ha="right" if offsets.get(int(pe), (6, 2))[0] < 0 else "left", fontsize=8.5)
    ax.set_xlabel("late-segment subspace error")
    ax.set_ylabel("final control regret")
    figstyle.panel(ax, "(b)")
    ax.text(0.97, 0.05, "labels: probe period", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.35")

    figstyle.no_title(
        f"Experiment 3: Probe-Rate Ablation  |  "
        f"$d={D}$, $r={R}$, $K={K}$, $T={T}$, $c={PROBE_COST}$, $W={WINDOW}$  "
        f"({N_SEEDS} seeds, bands $= \\pm 1$ SE)", y=1.02,
    )

    figstyle.save(out_path)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(sweep_results, env_ref):
    n = N_SEEDS
    print()
    print("=" * 80)
    print("Experiment 3 — Probe-rate ablation summary")
    print(f"  d={D}, r={R}, K={K}, T={T}, W={WINDOW}, c={PROBE_COST}")
    print("-" * 80)
    print(f"  {'pe':>5}  {'rate':>7}  {'costed±SE':>18}  {'control±SE':>18}  "
          f"{'sub_err':>9}  {'probes':>6}")
    print("-" * 80)
    for pe in sorted(sweep_results.keys()):
        runs    = sweep_results[pe]
        costed  = np.array([r.cumulative_costed_regret[-1]  for r in runs])
        control = np.array([r.cumulative_control_regret[-1] for r in runs])
        probes  = np.array([r.total_probes                  for r in runs])
        sub_err = avg_late_subspace_error(runs, env_ref)
        print(f"  {pe:>5}  {1/pe:>7.4f}  "
              f"{costed.mean():>8.1f}±{costed.std()/np.sqrt(n):>5.1f}  "
              f"{control.mean():>8.1f}±{control.std()/np.sqrt(n):>5.1f}  "
              f"{sub_err:>9.4f}  {probes.mean():>6.0f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env_ref = make_env(0)

    print("=" * 60)
    print("Experiment 3: Probe-Rate Ablation")
    print(f"  d={D}, r={R}, K={K}, T={T}")
    print(f"  probe_every sweep: {PROBE_EVERY_VALS}")
    print(f"  n_seeds={N_SEEDS}")
    print("=" * 60)

    print("\nRunning sweep...")
    sweep_results = figstyle.cached("exp3_probe_ablation", lambda: run_sweep(PROBE_EVERY_VALS, N_SEEDS))

    print_table(sweep_results, env_ref)

    make_figure(
        sweep_results, env_ref,
        out_path=os.path.join(OUT_DIR, "experiment3_probe_ablation.png"),
    )

    print("\nDone.")
