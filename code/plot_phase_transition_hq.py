"""
High-resolution, paper-quality Figure 1 using the *reproducible* values
from `appendix_tables.tex` / `tab:app-synthetic` (40 cells, with d in
{5,10,20,30,45,60,80,100}, r in {1,3,5,10,15,20}, r<d).

These values are produced by `experiment_synthetic_extended.py`
(K=10, T=5000, 40 actions, probe period 50, 10 seeds), so the figure
is fully reproducible from the script + data dict below.

Two-panel layout:
  (a) Phase-transition contour map
  (b) Crossover ribbon: ratio vs d for each r (log scale)

Output: `figures/experiment1_synthetic_phase.png`

Usage:
    cd code/
    python3 plot_phase_transition_hq.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
from scipy.interpolate import griddata


# -------------------------------------------------------------------------
# Paper-quality matplotlib settings
# -------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "font.family":        "serif",
    "font.size":          14,
    "axes.labelsize":     17.5,
    "axes.titlesize":     17,
    "axes.titleweight":   "bold",
    "axes.linewidth":     1.4,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "legend.fontsize":    15.5,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "text.color":         "black",
    "axes.labelcolor":    "black",
    "xtick.color":        "black",
    "ytick.color":        "black",
    "lines.linewidth":    2.6,
    "lines.markersize":   9,
    "lines.markeredgewidth": 1.4,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.7,
    "axes.axisbelow":     True,
})


# -------------------------------------------------------------------------
# Data — extracted directly from appendix_tables.tex `tab:app-synthetic`.
# Format: (d, r): (SPSC, LinUCB, Oracle)
# Reproduced by `experiment_synthetic_extended.py` (K=10, T=5000,
# 40 actions, probe period 50, 10 seeds, sigma_eps=0.3, spec_rad=0.99,
# feature_decay=1.5).
# -------------------------------------------------------------------------
data = {
    (  5,  1): ( 355,  203,    8), (  5,  3): ( 339,  221,  134),
    ( 10,  1): ( 270,  255,   24), ( 10,  3): ( 389,  399,  130),
    ( 10,  5): ( 465,  487,  254),
    ( 20,  1): ( 231,  251,   21), ( 20,  3): ( 353,  456,  134),
    ( 20,  5): ( 420,  565,  232), ( 20, 10): ( 631,  800,  498),
    ( 20, 15): ( 805,  899,  736),
    ( 30,  1): ( 189,  220,   24), ( 30,  3): ( 307,  407,  125),
    ( 30,  5): ( 372,  544,  217), ( 30, 10): ( 572,  775,  451),
    ( 30, 15): ( 741,  930,  663), ( 30, 20): ( 890, 1018,  838),
    ( 45,  1): ( 165,  188,   25), ( 45,  3): ( 271,  371,  118),
    ( 45,  5): ( 360,  504,  212), ( 45, 10): ( 499,  675,  395),
    ( 45, 15): ( 651,  819,  577), ( 45, 20): ( 811,  973,  758),
    ( 60,  1): ( 141,  165,   30), ( 60,  3): ( 229,  296,  117),
    ( 60,  5): ( 315,  414,  199), ( 60, 10): ( 441,  605,  367),
    ( 60, 15): ( 588,  743,  524), ( 60, 20): ( 708,  853,  663),
    ( 80,  1): ( 110,  118,   27), ( 80,  3): ( 231,  290,  116),
    ( 80,  5): ( 279,  381,  187), ( 80, 10): ( 419,  542,  345),
    ( 80, 15): ( 534,  671,  486), ( 80, 20): ( 640,  767,  599),
    (100,  1): ( 107,  112,   28), (100,  3): ( 192,  238,  107),
    (100,  5): ( 272,  346,  183), (100, 10): ( 368,  471,  317),
    (100, 15): ( 466,  573,  430), (100, 20): ( 609,  726,  572),
}

d_vals = sorted(set(d for d, r in data))
r_vals = sorted(set(r for d, r in data))
T_SIXTH = 5000 ** (1.0 / 6.0)


# -------------------------------------------------------------------------
# Figure layout (matches plot_phase_transition_hq.py)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(16.0, 6.3),
    gridspec_kw={"width_ratios": [1.15, 1]},
)


# =========================================================================
# Panel (a): Phase-transition contour map
# =========================================================================
ax = axes[0]

pts, vals = [], []
for (d, r), (s, l, o) in data.items():
    pts.append((d, r))
    vals.append(s / l)
pts = np.array(pts)
vals = np.array(vals)

di = np.linspace(3, 102, 300)
ri = np.linspace(0.5, 21, 300)
DI, RI = np.meshgrid(di, ri)
ZI = griddata(pts, vals, (DI, RI), method="cubic")

norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
cf = ax.contourf(
    DI, RI, ZI,
    levels=np.arange(0.5, 2.05, 0.05),
    cmap="RdBu_r", norm=norm, alpha=0.88,
)

# Empirical ratio = 1 contour (the win/lose boundary)
cs = ax.contour(
    DI, RI, ZI, levels=[1.0],
    colors=["black"], linewidths=3.0, linestyles=["-"],
)
ax.clabel(cs, fmt="ratio = 1.0", fontsize=12, inline_spacing=10,
          manual=[(15, 4)])

# Rule of thumb  d - r = T^{1/6} (a heuristic, not a prediction of the theorem)
# drawn only inside the plotted range, so it does not run along the frame
d_line = np.linspace(T_SIXTH + 0.5, T_SIXTH + 20.6, 100)
r_line = d_line - T_SIXTH
ax.plot(
    d_line, r_line,
    color="#222222", ls="--", lw=2.8, alpha=0.7,
    label=rf"Rule of thumb ($d - r = T^{{1/6}} \approx {T_SIXTH:.1f}$)",
)

# Per-cell verdict markers (binary, matching tab:app-synthetic verdicts)
for (d, r), (s, l, o) in data.items():
    ratio = s / l
    if ratio < 1.0:
        ax.plot(d, r, "o", color="#1a5276",
                ms=10, mec="white", mew=1.5, zorder=5)
    else:
        ax.plot(d, r, "^", color="#922b21",
                ms=10, mec="white", mew=1.5, zorder=5)

# Legend handles
ax.plot([], [], "o", color="#1a5276", ms=10, mec="white",
        label=r"SPSC wins (ratio $< 1$)")
ax.plot([], [], "^", color="#922b21", ms=10, mec="white",
        label=r"LinUCB wins (ratio $> 1$)")

cb = plt.colorbar(cf, ax=ax, shrink=0.92, pad=0.02)
cb.set_label("SPSC / LinUCB regret ratio", fontsize=15)
cb.set_ticks([0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0])
cb.ax.tick_params(labelsize=12)

ax.set_xlabel(r"Ambient dimension $d$")
ax.set_ylabel(r"Latent rank $r$")
# legend inside, in a band above the r = 20 row, so it hides no cell
leg = ax.legend(loc="upper left", ncol=1, framealpha=1.0, edgecolor="black")
leg.get_frame().set_linewidth(1.0)
for t in leg.get_texts():                      # a thin outline prints the small legend text darker
    t.set_path_effects([pe.withStroke(linewidth=0.28, foreground="black")])
ax.set_title("(a)", loc="left", fontweight="bold", fontsize=18, pad=8)
ax.set_xlim(3, 102)
ax.set_ylim(0.5, 29.5)                          # room for the legend above the r = 20 row
ax.set_yticks([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax.grid(False)


# =========================================================================
# Panel (b): Ratio vs d for each r (log-scale)
# =========================================================================
ax = axes[1]

r_groups = [1, 3, 5, 10, 15, 20]
cmap = plt.cm.viridis_r
colors = [cmap(i / max(len(r_groups) - 1, 1)) for i in range(len(r_groups))]

for idx, r in enumerate(r_groups):
    ds, rats = [], []
    for d in d_vals:
        if (d, r) in data:
            s, l, o = data[(d, r)]
            ds.append(d)
            rats.append(s / l)
    if len(ds) >= 2:
        ax.plot(
            ds, rats, "o-",
            color=colors[idx], lw=3.0, ms=10,
            mec="black", mew=1.0,
            label=rf"$r = {r}$", zorder=3,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"),
                          pe.Normal()],
        )

# Win/lose shading
ax.axhspan(0.5, 1, alpha=0.08, color="blue")
ax.axhspan(1, 3, alpha=0.08, color="red")
ax.axhline(1.0, color="black", lw=2.0, alpha=0.65)

ax.set_xlabel(r"Ambient dimension $d$")
ax.set_ylabel(r"SPSC / LinUCB regret ratio")

# Proxy legend entries for the win-region shading
from matplotlib.patches import Patch
spsc_patch = Patch(facecolor="blue", alpha=0.18,
                   label=r"SPSC wins (ratio $<$ 1)")
linucb_patch = Patch(facecolor="red", alpha=0.18,
                     label=r"LinUCB wins (ratio $>$ 1)")
handles, labels = ax.get_legend_handles_labels()
handles += [spsc_patch, linucb_patch]
labels   += [spsc_patch.get_label(), linucb_patch.get_label()]
leg_b = ax.legend(
    handles, labels,
    loc="upper right", title=r"Latent rank $r$",
    title_fontsize=15, framealpha=1.0, edgecolor="black",
    ncol=2, fontsize=14,
)
for t in leg_b.get_texts() + [leg_b.get_title()]:
    t.set_path_effects([pe.withStroke(linewidth=0.28, foreground="black")])
ax.set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)
ax.set_ylim(0.5, 2.5)
ax.set_xlim(3, 105)
ax.set_yscale("log")
ax.set_yticks([0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.2f}".rstrip("0").rstrip(".")))
ax.grid(True, alpha=0.30, which="both", linestyle="--", linewidth=0.7)


for a in list(axes) + [cb.ax]:
    for lab in (a.xaxis.label, a.yaxis.label):
        lab.set_path_effects([pe.withStroke(linewidth=0.28, foreground="black")])

# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------
plt.tight_layout(w_pad=4.0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures",
                   "experiment1_synthetic_phase.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, bbox_inches="tight", dpi=300, facecolor="white")
print(f"Saved: {OUT}")
print()
print("Summary of cells:")
print(f"  Total: {len(data)} cells")
print(f"  d values: {d_vals}")
print(f"  r values: {r_vals}")
spsc_wins = sum(1 for _, (s, l, o) in data.items() if s/l < 1.0)
linucb_wins = sum(1 for _, (s, l, o) in data.items() if s/l >= 1.0)
print(f"  SPSC wins (<1):  {spsc_wins}")
print(f"  LinUCB wins (>=1): {linucb_wins}")
