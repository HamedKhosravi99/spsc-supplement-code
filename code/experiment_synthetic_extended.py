"""
Experiment 1 Extended: Synthetic operating-regime benchmark (10 methods).

IDENTICAL to experiment1_synthetic_final.py but adds 7 more baselines:
  SPSC-Adaptive, SW-LinUCB, D-LinUCB, Restart-LinUCB,
  LowRank-Reward, LowOFUL, VOFUL
"""

import os, sys, time, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from environments import LowRankLDSEnvironment
from algorithm import (
    SPSC_Algorithm1, SPSC_Adaptive, LinUCB, OracleLinUCB,
    SWLinUCB, RestartLinUCB, LowRankRewardUCB, LowOFUL, VOFUL,
)

# =========================================================================
# Parameters — IDENTICAL to experiment1_synthetic_final.py
# =========================================================================
DS = [5, 10, 20, 30, 45, 60, 80, 100]
RS = [1, 3, 5, 10, 15, 20]
K = 10
T = 5000
SIGMA_EPS = 0.3
SPEC_RAD = 0.99
SIGMA_ETA = 0.04
N_ACTIONS = 40
PROBE_EVERY = 50
PROBE_COST = 0.1
WINDOW = 400
LAM = 0.01
DELTA = 0.05
FEATURE_DECAY = 1.5
N_SEEDS = 10
T_SIXTH = T ** (1.0 / 6.0)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COMPETITORS = ["LinUCB", "D-LinUCB", "SW-LinUCB", "Restart-LinUCB",
               "LowRank-Reward", "LowOFUL", "VOFUL"]


def final_stats(runs, attr="cumulative_costed_regret"):
    v = np.array([getattr(r, attr)[-1] for r in runs])
    return v.mean(), v.std() / np.sqrt(len(v))


# =========================================================================
# Run grid
# =========================================================================
results = {}

print(f"Synthetic Operating-Regime Benchmark (Extended — 10 methods)")
print(f"T={T} K={K} pe={PROBE_EVERY} W={WINDOW} lam={LAM} fd={FEATURE_DECAY}")
print(f"Grid: d={DS}, r={RS}, seeds={N_SEEDS}")
print(f"T^(1/6)={T_SIXTH:.2f}")
print("-" * 80)

for d in DS:
    for r in RS:
        if r >= d:
            continue
        t0 = time.time()
        res = {"SPSC": [], "SPSC-Adaptive": [], "LinUCB": [], "Oracle": [],
               "SW-LinUCB": [], "D-LinUCB": [], "Restart-LinUCB": [],
               "LowRank-Reward": [], "LowOFUL": [], "VOFUL": []}
        for seed in range(N_SEEDS):
            kw = dict(K=K, T=T, sigma_eps=SIGMA_EPS, spectral_radius=SPEC_RAD,
                      n_actions=N_ACTIONS, sigma_eta=SIGMA_ETA, seed=seed * 100,
                      piecewise_constant=True, feature_decay=FEATURE_DECAY)

            # --- Original 3 methods (unchanged) ---
            env = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["SPSC"].append(SPSC_Algorithm1(
                env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                window=WINDOW, lam=LAM, delta=DELTA, seed=seed,
                normalize_gamma_by_d=True).run())
            env2 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["LinUCB"].append(LinUCB(
                env2, lam=LAM, delta=DELTA, seed=seed).run())
            env3 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["Oracle"].append(OracleLinUCB(
                env3, window=WINDOW, lam=LAM, delta=DELTA, seed=seed).run())

            # --- 7 new methods ---
            env4 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["SPSC-Adaptive"].append(SPSC_Adaptive(
                env4, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                window=WINDOW, lam=LAM, delta=DELTA, seed=seed,
                m_relearn=30, det_window=50, cusum_threshold=3.0,
                warmup=100).run())
            env5 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["D-LinUCB"].append(LinUCB(
                env5, lam=LAM, delta=DELTA, seed=seed + 3000,
                forgetting_factor=0.998).run())
            env6 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["SW-LinUCB"].append(SWLinUCB(
                env6, window=WINDOW, lam=LAM, delta=DELTA,
                seed=seed + 4000).run())
            env7 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["Restart-LinUCB"].append(RestartLinUCB(
                env7, restart_period=T // K,
                lam=LAM, delta=DELTA, seed=seed + 5000).run())
            env8 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["LowRank-Reward"].append(LowRankRewardUCB(
                env8, window=WINDOW, pca_warmup=50,
                lam=LAM, delta=DELTA, seed=seed + 6000).run())
            env9 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["LowOFUL"].append(LowOFUL(
                env9, lam=LAM, delta=DELTA, seed=seed + 7000,
                pca_warmup=30, subspace_update_freq=20).run())
            env10 = LowRankLDSEnvironment(d=d, r=r, **kw)
            res["VOFUL"].append(VOFUL(
                env10, lam=LAM, delta=DELTA, seed=seed + 8000,
                pca_warmup=30, subspace_update_freq=20).run())

        results[(d, r)] = res
        s, _ = final_stats(res["SPSC"])
        a, _ = final_stats(res["SPSC-Adaptive"])
        l, _ = final_stats(res["LinUCB"])
        o, _ = final_stats(res["Oracle"])
        sw, _ = final_stats(res["SW-LinUCB"])
        dl, _ = final_stats(res["D-LinUCB"])
        lo, _ = final_stats(res["LowOFUL"])
        vo, _ = final_stats(res["VOFUL"])
        best_comp = min(final_stats(res[m])[0] for m in COMPETITORS)
        print(f"d={d:3d} r={r:2d} | SPSC={s:7.0f} Adpt={a:7.0f} Lin={l:7.0f} "
              f"Orc={o:7.0f} SW={sw:7.0f} DL={dl:7.0f} LO={lo:7.0f} VO={vo:7.0f} | "
              f"S/L={s/l:.3f} S/Best={s/max(best_comp,1):.3f} [{time.time()-t0:.0f}s]",
              flush=True)

# =========================================================================
# Collect ratio grids
# =========================================================================
valid = sorted(results.keys())
d_vals = sorted(set(d for d, r in valid))
r_vals = sorted(set(r for d, r in valid))

ratio_grid = np.full((len(r_vals), len(d_vals)), np.nan)
theory_grid = np.full((len(r_vals), len(d_vals)), np.nan)
oracle_grid = np.full((len(r_vals), len(d_vals)), np.nan)

for i, r in enumerate(r_vals):
    for j, d in enumerate(d_vals):
        if (d, r) not in results:
            continue
        s, _ = final_stats(results[(d, r)]["SPSC"])
        l, _ = final_stats(results[(d, r)]["LinUCB"])
        o, _ = final_stats(results[(d, r)]["Oracle"])
        ratio_grid[i, j] = s / l if l > 0 else np.nan
        theory_grid[i, j] = np.sqrt(r / d)
        oracle_grid[i, j] = o / l if l > 0 else np.nan

# =========================================================================
# Figure — 4-panel (same as original)
# =========================================================================
fig = plt.figure(figsize=(18, 14))

# --- Panel (a): Phase-transition heatmap ---
ax1 = fig.add_subplot(2, 2, 1)
norm1 = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=3.0)
im1 = ax1.imshow(ratio_grid, cmap="RdYlBu_r", norm=norm1, aspect="auto",
                  origin="lower")
for i in range(len(r_vals)):
    for j in range(len(d_vals)):
        v = ratio_grid[i, j]
        if np.isnan(v):
            ax1.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
        else:
            color = "white" if (v < 0.6 or v > 2.0) else "black"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color=color)
ax1.set_xticks(range(len(d_vals)))
ax1.set_xticklabels(d_vals)
ax1.set_yticks(range(len(r_vals)))
ax1.set_yticklabels(r_vals)
ax1.set_xlabel("Ambient dimension $d$", fontsize=12)
ax1.set_ylabel("Latent rank $r$", fontsize=12)
ax1.set_title("(a) SPSC / LinUCB regret ratio\n(blue < 1 = SPSC wins)", fontsize=12)
cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
cb1.set_label("Ratio", fontsize=10)

for i, r in enumerate(r_vals):
    boundary_d = r + T_SIXTH
    if d_vals[0] <= boundary_d <= d_vals[-1]:
        j_frac = np.interp(boundary_d, d_vals, range(len(d_vals)))
        ax1.plot(j_frac, i, "k*", markersize=10, zorder=5)
ax1.plot([], [], "k*", markersize=10, label=f"$d - r = T^{{1/6}} \\approx {T_SIXTH:.1f}$")
ax1.legend(loc="upper left", fontsize=9)

# --- Panel (b): Ratio vs d for each r (line plot) ---
ax2 = fig.add_subplot(2, 2, 2)
cmap = plt.cm.viridis
colors = [cmap(i / max(len(r_vals) - 1, 1)) for i in range(len(r_vals))]

for i, r in enumerate(r_vals):
    ds_valid = []
    rats_valid = []
    thrs_valid = []
    for j, d in enumerate(d_vals):
        if not np.isnan(ratio_grid[i, j]):
            ds_valid.append(d)
            rats_valid.append(ratio_grid[i, j])
            thrs_valid.append(theory_grid[i, j])
    if ds_valid:
        ax2.plot(ds_valid, rats_valid, "o-", color=colors[i], lw=2,
                 markersize=6, label=f"$r={r}$", zorder=3)
        ax2.plot(ds_valid, thrs_valid, "--", color=colors[i], lw=1,
                 alpha=0.5, zorder=2)

ax2.axhline(1.0, color="black", ls=":", lw=1.5, alpha=0.7)
ax2.fill_between([d_vals[0], d_vals[-1]], 0, 1, alpha=0.06, color="blue",
                  label="SPSC wins")
ax2.fill_between([d_vals[0], d_vals[-1]], 1, 5, alpha=0.06, color="red",
                  label="LinUCB wins")
ax2.set_xlabel("Ambient dimension $d$", fontsize=12)
ax2.set_ylabel("SPSC / LinUCB ratio", fontsize=12)
ax2.set_title("(b) Regret ratio vs $d$\n(solid = empirical, dashed = $\\sqrt{r/d}$ theory)",
              fontsize=12)
ax2.legend(fontsize=8, ncol=2, loc="upper right")
ax2.set_ylim(0, 4)
ax2.set_xlim(d_vals[0] - 2, d_vals[-1] + 2)

# --- Panel (c): Empirical ratio vs theory sqrt(r/d) scatter ---
ax3 = fig.add_subplot(2, 2, 3)
emp_all = []
thr_all = []
d_all = []
r_all = []
for i, r in enumerate(r_vals):
    for j, d in enumerate(d_vals):
        if not np.isnan(ratio_grid[i, j]):
            emp_all.append(ratio_grid[i, j])
            thr_all.append(theory_grid[i, j])
            d_all.append(d)
            r_all.append(r)

emp_all = np.array(emp_all)
thr_all = np.array(thr_all)
d_all = np.array(d_all)

sc = ax3.scatter(thr_all, emp_all, c=d_all, cmap="plasma", s=80,
                  edgecolors="black", linewidths=0.5, zorder=3)
ax3.plot([0, 1.5], [0, 1.5], "k--", lw=1, alpha=0.4, label="Empirical = Theory")
ax3.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax3.set_xlabel(r"Theory: $\sqrt{r/d}$", fontsize=12)
ax3.set_ylabel("Empirical: SPSC / LinUCB", fontsize=12)
ax3.set_title("(c) Theory vs empirical ratio\n(colored by $d$)", fontsize=12)
cb3 = plt.colorbar(sc, ax=ax3, shrink=0.85)
cb3.set_label("$d$", fontsize=10)
ax3.legend(fontsize=9)
ax3.set_xlim(0, 1.0)
ax3.set_ylim(0, 4.0)

# --- Panel (d): Regret comparison at r=10 ---
ax4 = fig.add_subplot(2, 2, 4)
r_target = 10
ds_slice = [d for d in d_vals if (d, r_target) in results]
spsc_vals = []
lin_vals = []
orc_vals = []
for d in ds_slice:
    s, _ = final_stats(results[(d, r_target)]["SPSC"])
    l, _ = final_stats(results[(d, r_target)]["LinUCB"])
    o, _ = final_stats(results[(d, r_target)]["Oracle"])
    spsc_vals.append(s)
    lin_vals.append(l)
    orc_vals.append(o)

x = np.arange(len(ds_slice))
w = 0.28
bars_s = ax4.bar(x - w, spsc_vals, w, label="SPSC", color="#1f77b4", edgecolor="black", lw=0.5)
bars_l = ax4.bar(x, lin_vals, w, label="LinUCB", color="#d62728", edgecolor="black", lw=0.5)
bars_o = ax4.bar(x + w, orc_vals, w, label="Oracle", color="#2ca02c", edgecolor="black", lw=0.5)

ax4.set_xticks(x)
ax4.set_xticklabels([f"$d$={d}" for d in ds_slice], fontsize=9)
ax4.set_ylabel("Cumulative costed regret", fontsize=12)
ax4.set_title(f"(d) Regret comparison at $r={r_target}$\n(SPSC overtakes LinUCB as $d$ grows)",
              fontsize=12)
ax4.legend(fontsize=9)

for i in range(len(ds_slice)):
    if spsc_vals[i] < lin_vals[i]:
        ax4.annotate("SPSC\nwins", xy=(x[i] - w, spsc_vals[i]),
                     xytext=(x[i] - w, spsc_vals[i] + 200),
                     fontsize=7, ha="center", color="#1f77b4",
                     arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
        break

fig.suptitle(
    "Synthetic Operating-Regime Benchmark: Phase Transition (Extended 10 Methods)\n"
    f"$T$={T}, $K$={K}, probe every {PROBE_EVERY}, "
    f"feature decay={FEATURE_DECAY}, {N_SEEDS} seeds",
    fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "experiment_synthetic_extended.png")
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved: {out_path}")

# =========================================================================
# Summary table — extended
# =========================================================================
print("\n" + "=" * 130)
print("SUMMARY (Extended — 10 methods)")
print("=" * 130)
print(f"{'d':>4} {'r':>3} | {'SPSC':>8} {'Adapt':>8} {'LinUCB':>8} {'Oracle':>8} "
      f"{'SW-Lin':>8} {'D-Lin':>8} {'Restart':>8} {'LowRank':>8} {'LowOFUL':>8} {'VOFUL':>8} | "
      f"{'S/L':>6} {'S/Best':>6} | {'verdict':>12}")
print("-" * 130)
for d in d_vals:
    for r in r_vals:
        if (d, r) not in results:
            continue
        res = results[(d, r)]
        s, _ = final_stats(res["SPSC"])
        a, _ = final_stats(res["SPSC-Adaptive"])
        l, _ = final_stats(res["LinUCB"])
        o, _ = final_stats(res["Oracle"])
        sw, _ = final_stats(res["SW-LinUCB"])
        dl, _ = final_stats(res["D-LinUCB"])
        re_, _ = final_stats(res["Restart-LinUCB"])
        lr, _ = final_stats(res["LowRank-Reward"])
        lo, _ = final_stats(res["LowOFUL"])
        vo, _ = final_stats(res["VOFUL"])
        rat = s / l
        best_comp = min(final_stats(res[m])[0] for m in COMPETITORS)
        s_best = s / max(best_comp, 1)
        if s < best_comp:
            verdict = "SPSC wins"
        elif a < best_comp:
            verdict = "Adapt wins"
        elif rat < 0.95:
            verdict = "SPSC wins"
        elif rat < 1.05:
            verdict = "tie"
        else:
            verdict = "LinUCB wins"
        print(f"{d:4d} {r:3d} | {s:8.0f} {a:8.0f} {l:8.0f} {o:8.0f} "
              f"{sw:8.0f} {dl:8.0f} {re_:8.0f} {lr:8.0f} {lo:8.0f} {vo:8.0f} | "
              f"{rat:6.3f} {s_best:6.3f} | {verdict:>12}")
print("=" * 130)
