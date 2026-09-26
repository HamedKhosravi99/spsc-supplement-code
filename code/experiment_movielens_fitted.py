"""
MovieLens grid with a consistent regret scale.

experiment_movielens_real_grid.py compares the best real rating in the action set with the fitted
reward x^T theta of the played movie, two different scales, and it replaces a probe by the nearest
movie.  Here regret is measured in the fitted model, max_x x^T theta_t minus x_t^T theta_t, rewards of
played movies are still their real ratings, and probes are answered by the fitted model plus noise.
Same cells, methods, seeds and settings as the grid script.

Usage:  python3 experiment_movielens_fitted.py [--workers 8]
Output: results/experiment_movielens_fitted.json (control and costed regret, probes per run)
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
# settings of experiment_movielens_real_grid.py (not imported: that file needs Python 3.12 syntax)
N_SEEDS, SEG_SIZE, N_SEGMENTS = 10, 500, 10
PROBE_EVERY, PROBE_COST, WINDOW, LAM, DELTA = 10, 0.02, 400, 0.01, 0.05
D_VALUES, R_VALUES = [55, 105, 200], [5, 10, 20]
METHOD_NAMES = ["Oracle-LinUCB", "SPSC-Alg1", "SPSC-Adaptive", "LowOFUL", "VOFUL", "LowRank-Reward",
                "SW-LinUCB", "D-LinUCB", "Restart-LinUCB", "LinUCB"]


def run_one(job):
    name, d, r, seed = job
    from environments import RealMovieLensEnvironment
    from algorithm import (SPSC_Algorithm1, SPSC_Adaptive, LinUCB, OracleLinUCB, SWLinUCB, RestartLinUCB,
                           LowRankRewardUCB, LowOFUL, VOFUL, LinTS, SWLinTS)
    env = RealMovieLensEnvironment(d=d, r=r, n_actions=40, segment_size=SEG_SIZE, n_segments=N_SEGMENTS,
                                   seed=seed * 13 + 7, regret="fitted", probes="fitted")
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
    return dict(method=name, d=d, r=r, seed=seed, control=float(m.cumulative_control_regret[-1]),
                costed=float(m.cumulative_costed_regret[-1]), probes=int(m.total_probes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    methods = METHOD_NAMES + ["LinTS", "SW-LinTS"]
    jobs = [(n, d, r, s) for d in D_VALUES for r in R_VALUES for n in methods for s in range(N_SEEDS)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    with open(os.path.join(HERE, "results", "experiment_movielens_fitted.json"), "w") as f:
        json.dump(dict(rows=rows, D_VALUES=D_VALUES, R_VALUES=R_VALUES, methods=methods), f, indent=1)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s")
    for metric in ("control", "costed"):
        print(f"\n== {metric} regret, mean over {N_SEEDS} seeds")
        print("   d   r " + "".join(f"{n[:11]:>12s}" for n in methods))
        for d in D_VALUES:
            for r in R_VALUES:
                line = f" {d:3d} {r:3d} "
                for n in methods:
                    c = np.array([x[metric] for x in rows if x["method"] == n and x["d"] == d and x["r"] == r])
                    line += f"{c.mean():12.1f}"
                print(line)


if __name__ == "__main__":
    main()
