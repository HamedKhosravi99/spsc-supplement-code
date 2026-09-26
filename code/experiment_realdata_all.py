"""
Every real-data cell of the paper (Covertype, Pendigits, Satimage, MNIST, Fashion-MNIST, Open Bandit) with
all twelve methods, in one consistent run.  Records control regret, costed regret (eq. dynreg, fee c per
probe) and the number of probes of every run.  Cells, environments, seeds and settings are those of the
grid scripts (experiment_<dataset>_grid.py, experiment_pendigits_extended.py, experiment_openbandit.py).

Usage:  python3 experiment_realdata_all.py [--workers 8] [--datasets covertype pendigits ...]
Output: results/experiment_realdata_all.json (merged per dataset)
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
OUT = os.path.join(HERE, "results", "experiment_realdata_all.json")
PROBE_EVERY, PROBE_COST, WINDOW, LAM, DELTA, N_SEEDS = 10, 0.02, 400, 0.01, 0.05, 10
SEG_SIZE, N_SEGMENTS = 500, 10
CELLS = {
    "covertype": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "pendigits": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "satimage": [(5, 1)] + [(d, r) for d in (55, 105) for r in (1, 10, 20, 30)],
    "mnist": [(d, r) for d in (55, 105, 200) for r in (5, 10, 20)],
    "fmnist": [(d, r) for d in (55, 105, 200) for r in (5, 10, 20)],
    "openbandit": [(d, r) for d in (55, 105) for r in (5, 10)],
}
METHODS = ["Oracle-LinUCB", "SPSC-Alg1", "SPSC-Adaptive", "LowOFUL", "VOFUL", "LowRank-Reward", "SW-LinUCB",
           "D-LinUCB", "Restart-LinUCB", "LinUCB", "LinTS", "SW-LinTS"]


def make_env(ds, d, r, seed):
    import environments as E
    s = seed * 13 + 7
    if ds == "covertype":
        return E.CovtypeEnvironment(d=d, r=r, K=N_SEGMENTS, T=SEG_SIZE * N_SEGMENTS, sigma_eps=0.3,
                                    spectral_radius=0.99, n_actions=40, seed=s, sigma_eta=0.04)
    cls = dict(pendigits=E.RealPendigitsEnvironment, satimage=E.RealSatimageEnvironment, mnist=E.MNISTEnvironment,
               fmnist=E.FashionMNISTEnvironment, openbandit=E.OpenBanditEnvironment)[ds]
    return cls(d=d, r=r, n_actions=40, segment_size=SEG_SIZE, n_segments=N_SEGMENTS, seed=s)


def run_one(job):
    ds, d, r, name, seed = job
    from algorithm import (SPSC_Algorithm1, SPSC_Adaptive, LinUCB, OracleLinUCB, SWLinUCB, RestartLinUCB,
                           LowRankRewardUCB, LowOFUL, VOFUL, LinTS, SWLinTS)
    env = make_env(ds, d, r, seed)
    T_env = env.T
    if name == "SPSC-Alg1":
        m = SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST, window=WINDOW, lam=LAM,
                            delta=DELTA, seed=seed, normalize_gamma_by_d=True).run()
    elif name == "SPSC-Adaptive":
        m = SPSC_Adaptive(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST, window=WINDOW, lam=LAM,
                          delta=DELTA, seed=seed, m_relearn=30, det_window=50, cusum_threshold=3.0,
                          warmup=100).run()
    elif name == "LinUCB":
        m = LinUCB(env, lam=LAM, delta=DELTA, seed=seed + 1000).run()
    elif name == "Oracle-LinUCB":
        m = OracleLinUCB(env, window=10000, lam=LAM, delta=DELTA, seed=seed + 2000).run()
    elif name == "D-LinUCB":
        m = LinUCB(env, lam=LAM, delta=DELTA, seed=seed + 3000, forgetting_factor=0.998).run()
    elif name == "SW-LinUCB":
        m = SWLinUCB(env, window=WINDOW, lam=LAM, delta=DELTA, seed=seed + 4000).run()
    elif name == "Restart-LinUCB":
        m = RestartLinUCB(env, restart_period=T_env // N_SEGMENTS, lam=LAM, delta=DELTA, seed=seed + 5000).run()
    elif name == "LowRank-Reward":
        m = LowRankRewardUCB(env, window=WINDOW, pca_warmup=50, lam=LAM, delta=DELTA, seed=seed + 6000).run()
    elif name == "LowOFUL":
        m = LowOFUL(env, lam=LAM, delta=DELTA, seed=seed + 7000, pca_warmup=30, subspace_update_freq=20).run()
    elif name == "VOFUL":
        m = VOFUL(env, lam=LAM, delta=DELTA, seed=seed + 8000, pca_warmup=30, subspace_update_freq=20).run()
    elif name == "LinTS":
        m = LinTS(env, lam=LAM, delta=DELTA, seed=seed + 9000).run()
    elif name == "SW-LinTS":
        m = SWLinTS(env, window=WINDOW, lam=LAM, delta=DELTA, seed=seed + 10000).run()
    else:
        raise ValueError(name)
    return dict(dataset=ds, d=d, r=r, method=name, seed=seed, control=float(m.cumulative_control_regret[-1]),
                costed=float(m.cumulative_costed_regret[-1]), probes=int(m.total_probes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=list(CELLS))
    args = ap.parse_args()
    rows = json.load(open(OUT))["rows"] if os.path.exists(OUT) else []
    rows = [x for x in rows if x["dataset"] not in args.datasets]
    for ds in args.datasets:
        jobs = [(ds, d, r, n, s) for (d, r) in CELLS[ds] for n in METHODS for s in range(N_SEEDS)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            new = list(ex.map(run_one, jobs, chunksize=4))
        rows += new
        with open(OUT, "w") as f:
            json.dump(dict(rows=rows, methods=METHODS, cells=CELLS), f, indent=1)
        print(f"{ds}: {len(new)} runs in {time.time() - t0:.0f}s", flush=True)
        for (d, r) in CELLS[ds]:
            line = f"  d={d:3d} r={r:2d} "
            for n in METHODS:
                c = np.array([x["costed"] for x in new if x["d"] == d and x["r"] == r and x["method"] == n])
                line += f"{c.mean():9.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
