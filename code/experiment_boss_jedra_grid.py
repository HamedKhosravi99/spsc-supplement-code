"""
BOSS and Jedra against SPSC on a (d, r) grid (Table 3 of the paper).

Runs SPSC (Algorithm 1), SPSC-Adaptive, BOSS and Jedra over (d, r) in {(55,5), (55,10), (105,5),
(105,10), (105,20), (200,5), (200,10), (200,20)} on the synthetic piecewise low-rank benchmark.
BOSS and Jedra learn one subspace in their first 200 rounds, as published, and are not restarted.
SPSC-Adaptive detects the changes itself.  Algorithm 1 is given the segment boundaries.

Output: results/experiment_boss_jedra_grid.json

Run time: ~2-4 h on 16 cores (8 cells x 10 seeds x 4 methods).
"""

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from environments import LowRankLDSEnvironment
from algorithm import SPSC_Algorithm1, SPSC_Adaptive, BOSSAdapted, JedraAdapted
from results_io import save_results

N_SEEDS = 10
T, K = 5000, 10
SIGMA_EPS, SPEC_RAD, SIGMA_ETA = 0.3, 0.99, 0.04
N_ACTIONS = 40
PROBE_EVERY, PROBE_COST = 50, 0.1
WINDOW, LAM, DELTA = 400, 0.01, 0.05
FEATURE_DECAY = 1.5

# The (d, r) cells of Table 3.
CELLS = [(55, 5), (55, 10), (105, 5), (105, 10), (105, 20),
         (200, 5), (200, 10), (200, 20)]

METHODS = ["SPSC-Alg1", "SPSC-Adaptive", "BOSS-adapted", "Jedra-adapted"]


def make_env(seed, d, r):
    return LowRankLDSEnvironment(
        d=d, r=r, K=K, T=T, sigma_eps=SIGMA_EPS, spectral_radius=SPEC_RAD,
        n_actions=N_ACTIONS, sigma_eta=SIGMA_ETA, seed=seed * 100,
        piecewise_constant=True, feature_decay=FEATURE_DECAY,
    )


def run_all(seed, d, r):
    env = make_env(seed, d, r)
    m = SPSC_Algorithm1(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                        window=WINDOW, lam=LAM, delta=DELTA, seed=seed,
                        normalize_gamma_by_d=True).run()
    out = {"SPSC-Alg1": float(m.cumulative_costed_regret[-1])}

    env = make_env(seed, d, r)
    m = SPSC_Adaptive(env, probe_every=PROBE_EVERY, probe_cost=PROBE_COST,
                      window=WINDOW, lam=LAM, delta=DELTA, seed=seed,
                      m_relearn=30, det_window=50, cusum_threshold=3.0,
                      warmup=100).run()
    out["SPSC-Adaptive"] = float(m.cumulative_costed_regret[-1])

    env = make_env(seed, d, r)
    m = BOSSAdapted(env, m_explore=200, lam=LAM, delta=DELTA,
                    seed=seed + 11000).run()
    out["BOSS-adapted"] = float(m.cumulative_control_regret[-1])

    env = make_env(seed, d, r)
    m = JedraAdapted(env, m_explore=200, lam=LAM, delta=DELTA,
                     seed=seed + 12000).run()
    out["Jedra-adapted"] = float(m.cumulative_control_regret[-1])
    return out


def run_cell(d, r):
    print(f"\n[d={d}, r={r}]")
    t0 = time.time()
    res = {m: [] for m in METHODS}
    for s in range(N_SEEDS):
        cell = run_all(s, d, r)
        for m in METHODS:
            res[m].append(cell[m])
    for m in METHODS:
        arr = np.array(res[m])
        print(f"  {m:<15} {arr.mean():>7.0f} +/- {arr.std()/np.sqrt(N_SEEDS):>5.0f}")
    print(f"  [{time.time()-t0:.1f}s]")
    return {m: np.array(v) for m, v in res.items()}


if __name__ == "__main__":
    print("=" * 80)
    print(f"BOSS/Jedra grid (Table 3)   T={T}, K={K}, seeds={N_SEEDS}")
    print("BOSS/Jedra learn one subspace in their first 200 rounds and are not restarted.")
    print("=" * 80)

    all_results = {}
    overall_t0 = time.time()
    for (d, r) in CELLS:
        all_results[(d, r)] = run_cell(d, r)
    print(f"\nTotal time: {(time.time()-overall_t0)/60:.1f} min")

    save_results(
        __file__,
        config={
            "N_SEEDS": N_SEEDS, "T": T, "K": K, "CELLS": CELLS,
            "SIGMA_EPS": SIGMA_EPS, "SPEC_RAD": SPEC_RAD, "SIGMA_ETA": SIGMA_ETA,
            "N_ACTIONS": N_ACTIONS, "PROBE_EVERY": PROBE_EVERY, "PROBE_COST": PROBE_COST,
            "WINDOW": WINDOW, "LAM": LAM, "DELTA": DELTA,
            "FEATURE_DECAY": FEATURE_DECAY, "METHODS": METHODS,
        },
        results=all_results,
    )
    print("Done.")
