"""
Window conditioning in the runs (App. G.1).  At every exploitation round with |W_t| >= 2r, records the smallest
eigenvalue of the window's projected design sum_{s in W_t} z_t(x_s) z_t(x_s)^T divided by |W_t|, on the three
benchmarks of Table 19 with 10 seeds, for Algorithm 1 with the settings of the runs.  With unit-norm actions
(R_A = 1), r times the minimum is kappa_z of condition (WC).

Usage:  python3 check_window_conditioning.py [--seeds 10] [--workers 8]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_alg1_settings import BENCHMARKS, make_env


def job(a):
    bname, seed = a
    from spsc_exact import SPSC, periodic_probe_rounds
    b = BENCHMARKS[bname]
    env = make_env(b, seed)
    alg = SPSC(env, periodic_probe_rounds(env, b["pe"]), r=b["r"], W=b["W"], lam=b["lam"], probe_cost=b["c"],
               seed=seed, sigma2_hat=env.sigma_eps ** 2, S_delta=0.0, refresh="every", radius="runs")
    alg.run()
    return bname, np.array(alg.cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(bn, s) for bn in BENCHMARKS for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(job, jobs))
    for bn in BENCHMARKS:
        c = np.concatenate([x[1] for x in rows if x[0] == bn])
        r = BENCHMARKS[bn]["r"]
        print(f"{bn:22s} rounds {len(c):6d}  min {c.min():.2e}  1st pct {np.percentile(c, 1):.2e}  "
              f"median {np.median(c):.2e}  r x min {r * c.min():.1e}  zero count {(c <= 1e-12).sum()}")


if __name__ == "__main__":
    main()
