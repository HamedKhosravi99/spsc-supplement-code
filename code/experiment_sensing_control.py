"""
Sensing control and compression control on every benchmark cell of the paper.

Probe-LinUCB is LinUCB given exactly the probes of Algorithm 1 (same rounds, same u_t, same y_t, same
cost), used as ordinary regression data in the ambient space.  Probe-SW-LinUCB (no reset) is the same
control for SPSC-Adaptive.  RandomSubspace is windowed ridge UCB in a fixed random r-dimensional
subspace, with no probes.  The synthetic grid also reruns LinUCB, SPSC-Alg1 and SPSC-Adaptive so that its
table is self-contained; on the real-data cells those rows come from the existing result files.

Usage:  python3 experiment_sensing_control.py [--workers 8] [--datasets grid covertype ...] [--seeds 10]
Output: results/experiment_sensing_control.json (merged per dataset)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "experiment_sensing_control.json")
GRID = [(d, r) for d in (5, 10, 20, 30, 45, 60, 80, 100) for r in (1, 3, 5, 10, 15, 20) if r < d]
CELLS = {
    "grid": GRID,
    "covertype": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "pendigits": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "satimage": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "mnist": [(d, r) for d in (55, 105, 200) for r in (5, 10, 20)],
    "fmnist": [(d, r) for d in (55, 105, 200) for r in (5, 10, 20)],
    "openbandit": [(d, r) for d in (55, 105) for r in (5, 10)],
    "movielens": [(d, r) for d in (55, 105, 200) for r in (5, 10, 20)],
}
# probe period, fee, window, ridge parameter, as in the scripts that produced the paper's tables
SETTINGS = {"grid": (50, 0.1, 400, 0.01)}
DEFAULT = (10, 0.02, 400, 0.01)
CONTROLS = ["Probe-LinUCB", "Probe-SW-LinUCB-noreset", "RandomSubspace"]
GRID_EXTRA = ["LinUCB", "SPSC-Alg1", "SPSC-Adaptive"]


def make_env(ds, d, r, seed):
    import environments as E
    s = seed * 13 + 7
    if ds == "grid":
        return E.LowRankLDSEnvironment(d=d, r=r, K=10, T=5000, sigma_eps=0.3, spectral_radius=0.99, n_actions=40,
                                       sigma_eta=0.04, seed=seed * 100, piecewise_constant=True, feature_decay=1.5)
    if ds == "movielens":
        return E.RealMovieLensEnvironment(d=d, r=r, n_actions=40, segment_size=500, n_segments=10, seed=s,
                                          regret="fitted", probes="fitted")
    from experiment_realdata_all import make_env as real_env
    return real_env(ds, d, r, seed)


def run_one(job):
    ds, d, r, name, seed = job
    from algorithm import SPSC_Algorithm1, SPSC_Adaptive, LinUCB, RandomSubspaceUCB
    from probe_baselines import ProbeLinUCB
    pe, c, W, lam = SETTINGS.get(ds, DEFAULT)
    env = make_env(ds, d, r, seed)
    t0 = time.time()
    if name == "Probe-LinUCB":
        m = ProbeLinUCB(env, probe_every=pe, probe_cost=c, window=None, lam=lam, delta=0.05, seed=seed).run()
    elif name == "Probe-SW-LinUCB-noreset":
        m = ProbeLinUCB(env, probe_every=pe, probe_cost=c, window=W, lam=lam, delta=0.05, seed=seed,
                        reset_at_boundaries=False).run()
    elif name == "RandomSubspace":
        m = RandomSubspaceUCB(env, window=W, lam=lam, delta=0.05, seed=seed).run()
    elif name == "LinUCB":
        m = LinUCB(env, lam=lam, delta=0.05, seed=seed + 1000).run()
    elif name == "SPSC-Alg1":
        m = SPSC_Algorithm1(env, probe_every=pe, probe_cost=c, window=W, lam=lam, delta=0.05, seed=seed,
                            normalize_gamma_by_d=True).run()
    elif name == "SPSC-Adaptive":
        m = SPSC_Adaptive(env, probe_every=pe, probe_cost=c, window=W, lam=lam, delta=0.05, seed=seed,
                          m_relearn=30, det_window=50, cusum_threshold=3.0, warmup=100).run()
    return dict(dataset=ds, d=d, r=r, method=name, seed=seed, control=float(m.cumulative_control_regret[-1]),
                costed=float(m.cumulative_costed_regret[-1]), probes=int(m.total_probes), secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=list(CELLS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    for ds in args.datasets:
        methods = CONTROLS + (GRID_EXTRA if ds == "grid" else [])
        jobs = [(ds, d, r, n, s) for (d, r) in CELLS[ds] for n in methods for s in range(args.seeds)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(run_one, jobs, chunksize=2))
        old = json.load(open(args.out)) if os.path.exists(args.out) else dict(rows=[], cells={})
        old["rows"] = [x for x in old["rows"] if x["dataset"] != ds] + rows
        old["cells"][ds] = CELLS[ds]
        with open(args.out, "w") as f:
            json.dump(old, f, indent=1)
        print(f"{ds}: {len(rows)} runs in {time.time() - t0:.0f}s", flush=True)
        for (d, r) in CELLS[ds]:
            line = f"  d={d:3d} r={r:2d}"
            for n in methods:
                c = [x["costed"] for x in rows if x["d"] == d and x["r"] == r and x["method"] == n]
                line += f"  {n} {np.mean(c):8.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
