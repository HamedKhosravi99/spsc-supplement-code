"""
Shared print-scale style for the appendix figures.

The appendix includes each figure at \\textwidth (5.5 in), and the figures are drawn 5.5 in wide, so every font prints
at its nominal size (8.5-10.5 pt).  Three-panel figures use two rows so each panel keeps a readable size.  Panel titles
are only "(a)", "(b)", ..., since the LaTeX captions describe the panels.
Figures are saved straight into ../figures at 300 dpi.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 5.5
HEIGHT = {1: 3.1, 2: 2.9, 3: 5.2}
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def apply(scale=1.0):
    """scale > 1 for figures drawn on a wider canvas than they print (the three-panel rows), so the printed
    sizes stay the same."""
    k = scale
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5 * k,
        "axes.titlesize": 10.5 * k,
        "axes.labelsize": 10 * k,
        "xtick.labelsize": 9 * k,
        "ytick.labelsize": 9 * k,
        "legend.fontsize": 8.5 * k,
        "legend.framealpha": 0.92,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "lines.linewidth": 1.5 * k,
        "lines.markersize": 4.5 * k,
        "axes.linewidth": 0.8 * k,
        "xtick.major.width": 0.6 * k,
        "ytick.major.width": 0.6 * k,
        "xtick.major.size": 2.5 * k,
        "ytick.major.size": 2.5 * k,
        "grid.linewidth": 0.4 * k,
        "errorbar.capsize": 2 * k,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def size(ncols, height=None):
    return (WIDTH, height if height is not None else HEIGHT[ncols])


SCALE3 = 1.12         # three panels in one row: fonts print at about 7-8 pt


def size3():
    """One row of three panels, drawn 7.7 in wide and printed at \\textwidth."""
    return (7.7, 2.9)


def grid21(height=5.2, bottom_ratio=0.85):
    """(a) and (b) side by side on top, (c) across the full width below."""
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, bottom_ratio])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    return fig, axes


def grid3(height=None):
    """Three panels in two rows: (a) and (b) on top, (c) centred below."""
    fig = plt.figure(figsize=(WIDTH, height if height is not None else HEIGHT[3]))
    gs = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
    return fig, axes


def panel(ax, label):
    """Bold panel label at the top left, as in the main-text figures."""
    ax.set_title(label, loc="left", fontweight="bold", pad=3)


def no_title(*args, **kwargs):
    """The LaTeX caption carries the figure title."""
    return None


def save(out_path, fig=None):
    path = os.path.join(FIG_DIR, os.path.basename(out_path))
    fig = fig or plt.gcf()
    fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=1.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300, facecolor="white")
    print(f"Saved: {path}")
    return path


def cached(name, compute):
    """Run compute() once and keep its result in $FIG_CACHE_DIR/<name>.pkl, so a figure can be redrawn without
    rerunning its experiment.  Without FIG_CACHE_DIR the experiment always runs."""
    import pickle
    cache_dir = os.environ.get("FIG_CACHE_DIR")
    if not cache_dir:
        return compute()
    path = os.path.join(cache_dir, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = compute()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out
