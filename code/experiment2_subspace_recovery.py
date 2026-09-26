"""
Experiment 2: Probing improves subspace recovery.

Shows that subspace estimation error ||P_hat_k - P_k*||_2 decays at the
theoretical rate 1/sqrt(m) as a function of the number of probes m within
a segment.  Validates the concentration bound behind Theorem (subspace recovery).

Outputs
-------
  experiment2_subspace_recovery.png
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
PROBE_EVERY = 10          # dense probing so we get a long decay curve
PROBE_COST  = 0.1
WINDOW      = 100
N_SEEDS     = 20          # more seeds for smoother scatter

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(seed):
    return LowRankLDSEnvironment(d=D, r=R, K=K, T=T, seed=seed * 100)


# ---------------------------------------------------------------------------
# Collect (probe_index_in_segment, subspace_error) pairs
# ---------------------------------------------------------------------------

def collect_decay_data(n_seeds):
    """
    Returns list of (m, err) pairs — one per probe round, all seeds and segments.
    m = probe index within the segment (1, 2, 3, ...).
    """
    pairs = []

    for seed in range(n_seeds):
        print(f"  seed {seed+1}/{n_seeds}", end="\r", flush=True)
        env = make_env(seed)
        run = SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                              window=WINDOW, lam=1.0, delta=0.05, seed=seed).run()

        # Walk through probe rounds, counting probes within each segment
        probe_ts = np.where(run.probe_flags)[0]
        seg_probe_count = {}  # seg_index -> count so far
        for ti in probe_ts:
            k = env.seg_of[ti]
            seg_probe_count[k] = seg_probe_count.get(k, 0) + 1
            m   = seg_probe_count[k]
            err = run.subspace_error[ti]
            if not np.isnan(err):
                pairs.append((m, err))

    print(flush=True)
    return pairs


# ---------------------------------------------------------------------------
# Binned mean ± SE
# ---------------------------------------------------------------------------

def bin_decay(pairs, max_m=None):
    ms   = np.array([p[0] for p in pairs])
    errs = np.array([p[1] for p in pairs])
    if max_m is None:
        max_m = int(ms.max())
    bins = np.arange(1, max_m + 1)
    mean, se, counts = [], [], []
    for m in bins:
        idx = ms == m
        if idx.sum() > 0:
            v = errs[idx]
            mean.append(v.mean())
            se.append(v.std() / np.sqrt(idx.sum()))
            counts.append(idx.sum())
        else:
            mean.append(np.nan)
            se.append(np.nan)
            counts.append(0)
    return bins, np.array(mean), np.array(se), np.array(counts)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pairs, out_path):
    ms   = np.array([p[0] for p in pairs])
    errs = np.array([p[1] for p in pairs])
    max_m = int(ms.max())

    bins, mean_err, se_err, counts = bin_decay(pairs, max_m=min(max_m, 150))
    # Keep only bins with enough observations
    valid = counts >= 5
    bins_v  = bins[valid]
    mean_v  = mean_err[valid]
    se_v    = se_err[valid]

    # Fit scale to first binned mean: scale / sqrt(1) = mean_v[0]
    scale = mean_v[0] if len(mean_v) > 0 else 1.0
    m_fit = np.linspace(1, bins_v[-1], 300)

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(2))

    # ---- Panel (a): scatter + binned mean + theory curve ----
    ax = axes[0]
    # Scatter raw points (thin, many seeds)
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(ms))
    ax.scatter(ms + jitter, errs, s=4, alpha=0.08, color="#1f77b4", zorder=1, lw=0)
    # Binned mean ± 1 SE
    ax.fill_between(bins_v, mean_v - se_v, mean_v + se_v, color="#e6550d", alpha=0.25, lw=0, zorder=2)
    ax.plot(bins_v, mean_v, color="#e6550d", lw=1.6, zorder=3, label="binned mean $\\pm$ 1 SE")
    # Theory 1/sqrt(m) curve
    ax.plot(m_fit, scale / np.sqrt(m_fit), color="black", ls="--", lw=1.2,
            label=r"reference $\propto m^{-1/2}$", zorder=4)
    ax.set_xlabel("probes in the segment $m$")
    ax.set_ylabel(r"$\|\widehat{P}_k - P_k^\star\|_2$")
    figstyle.panel(ax, "(a)")
    ax.legend()
    ax.set_xlim(0, min(max_m + 2, 152))
    ax.set_ylim(bottom=0)

    # ---- Panel (b): log-log version to show power law clearly ----
    ax = axes[1]
    ax.scatter(ms, errs, s=4, alpha=0.08, color="#1f77b4", zorder=1, lw=0)
    ax.fill_between(bins_v, mean_v - se_v, mean_v + se_v, color="#e6550d", alpha=0.25, lw=0, zorder=2)
    ax.plot(bins_v, mean_v, color="#e6550d", lw=1.6, zorder=3, label="binned mean $\\pm$ 1 SE")
    ax.plot(m_fit, scale / np.sqrt(m_fit), color="black", ls="--", lw=1.2,
            label=r"reference $\propto m^{-1/2}$", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("probes in the segment $m$")
    ax.set_ylabel(r"$\|\widehat{P}_k - P_k^\star\|_2$")
    figstyle.panel(ax, "(b)")
    ax.legend()

    figstyle.no_title(
        f"Experiment 2: Subspace Recovery Rate  |  "
        f"$d={D}$, $r={R}$, $K={K}$, probe every {PROBE_EVERY}  "
        f"({N_SEEDS} seeds $\\times$ {K} segments)", y=1.02,
    )

    figstyle.save(out_path)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(pairs):
    ms   = np.array([p[0] for p in pairs])
    errs = np.array([p[1] for p in pairs])

    print()
    print("=" * 55)
    print("Experiment 2 — Subspace error by probe count (binned)")
    print(f"  d={D}, r={R}, probe_every={PROBE_EVERY}, seeds={N_SEEDS}")
    print("-" * 55)
    print(f"  {'m':>4}   {'mean err':>10}   {'SE':>8}   {'n':>6}")
    print("-" * 55)
    for m_show in [1, 2, 5, 10, 20, 50, 100]:
        idx = ms == m_show
        if idx.sum() >= 3:
            v = errs[idx]
            print(f"  {m_show:>4}   {v.mean():>10.4f}   "
                  f"{v.std()/np.sqrt(idx.sum()):>8.4f}   {idx.sum():>6}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2: Subspace Recovery Rate")
    print(f"  d={D}, r={R}, K={K}, T={T}")
    print(f"  probe_every={PROBE_EVERY}, n_seeds={N_SEEDS}")
    print("=" * 60)

    print("\nRunning seeds...")
    pairs = figstyle.cached("exp2_subspace_recovery", lambda: collect_decay_data(N_SEEDS))
    print(f"  Total (m, err) pairs collected: {len(pairs)}")

    print_table(pairs)

    make_figure(
        pairs,
        out_path=os.path.join(OUT_DIR, "experiment2_subspace_recovery_notitle.png"),
    )

    print("\nDone.")
