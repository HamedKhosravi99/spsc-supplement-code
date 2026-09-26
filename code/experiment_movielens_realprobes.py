"""
MovieLens with every response a real rating: a probe is answered by the real rating of the nearest movie in the
round's action set (RealMovieLensEnvironment probes="snap"), so no response comes from the fitted model.  Regret is
still measured against each segment's ridge fit, as in experiment_movielens_fitted.py.  Only the SPSC rows change,
since the baselines never probe; their rows are those of results/experiment_movielens_fitted.json.

Usage:  python3 experiment_movielens_realprobes.py [--workers 8]
Output: results/experiment_movielens_realprobes.json
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
N_SEEDS, SEG_SIZE, N_SEGMENTS = 10, 500, 10
PROBE_EVERY, PROBE_COST, WINDOW, LAM, DELTA = 10, 0.02, 400, 0.01, 0.05
D_VALUES, R_VALUES = [55, 105, 200], [5, 10, 20]
METHODS = ["SPSC-Alg1", "SPSC-Adaptive"]


def run_one(job):
    name, d, r, seed = job
    from environments import RealMovieLensEnvironment
    from algorithm import SPSC_Algorithm1, SPSC_Adaptive
    env = RealMovieLensEnvironment(d=d, r=r, n_actions=40, segment_size=SEG_SIZE, n_segments=N_SEGMENTS,
                                   seed=seed * 13 + 7, regret="fitted", probes="snap")
    if name == "SPSC-Alg1":
        m = SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST, window=WINDOW, lam=LAM,
                            delta=DELTA, seed=seed, normalize_gamma_by_d=True).run()
    else:
        m = SPSC_Adaptive(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST, window=WINDOW, lam=LAM,
                          delta=DELTA, seed=seed, m_relearn=30, det_window=50, cusum_threshold=3.0,
                          warmup=100).run()
    return dict(method=name, d=d, r=r, seed=seed, control=float(m.cumulative_control_regret[-1]),
                costed=float(m.cumulative_costed_regret[-1]), probes=int(m.total_probes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(n, d, r, s) for d in D_VALUES for r in R_VALUES for n in METHODS for s in range(N_SEEDS)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    with open(os.path.join(HERE, "results", "experiment_movielens_realprobes.json"), "w") as f:
        json.dump(dict(rows=rows, D_VALUES=D_VALUES, R_VALUES=R_VALUES, methods=METHODS), f, indent=1)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
