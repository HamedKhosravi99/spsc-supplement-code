"""
Algorithm 1 as written (spsc_exact.SPSC) against the code behind the benchmark runs (SPSC_Algorithm1).

Part 1 checks the code:
  1. the lifted sample is unbiased,
  2. the basis refreshes after the 1st, 2nd, 4th, ... probe of every segment,
  3. with r = d the projected index is ambient ridge-UCB, as it must be by rotation invariance,
  4. Algorithm 1 with the two settings of App. E (a new basis after every probe, the standard ridge
     radius with no eps_bar term) reproduces SPSC_Algorithm1 exactly on three benchmarks.
Part 2 measures each setting on the same three benchmarks, with the paper's settings and seeds, next to
LinUCB and ridge-UCB in a random r-dimensional subspace with no probes (Table 19 of the paper reports LinUCB, the runs and the 1, 2, 4, ... refresh).

Usage:  python3 experiment_alg1_settings.py [--seeds 10] [--workers 8]
Output: results/experiment_alg1_settings.json
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

BENCHMARKS = {
    "synthetic d=30 r=5":   dict(env="synthetic", d=30, r=5, pe=50, W=400, lam=0.01, c=0.1),
    "synthetic d=100 r=10": dict(env="synthetic", d=100, r=10, pe=50, W=400, lam=0.01, c=0.1),
    "pendigits d=105 r=10": dict(env="pendigits", d=105, r=10, pe=10, W=400, lam=0.01, c=0.02),
}
METHODS = ["LinUCB", "runs (every probe, standard radius)", "runs with the 1, 2, 4, ... refresh",
           "Algorithm 1 as written", "random r-dim subspace, no probes"]


def make_env(b, seed):
    if b["env"] == "synthetic":
        from environments import LowRankLDSEnvironment
        return LowRankLDSEnvironment(d=b["d"], r=b["r"], K=10, T=5000, sigma_eps=0.3, spectral_radius=0.99,
                                     n_actions=40, sigma_eta=0.04, seed=seed * 100, piecewise_constant=True,
                                     feature_decay=1.5)
    if b["env"] == "pendigits":
        from environments import RealPendigitsEnvironment
        return RealPendigitsEnvironment(d=b["d"], r=b["r"], n_actions=40, segment_size=500, n_segments=10,
                                        seed=seed * 13 + 7)
    raise ValueError(f"unknown benchmark {b['env']}")


def run_one(job):
    bname, method, seed = job
    from algorithm import LinUCB, SPSC_Algorithm1, RandomSubspaceUCB
    from spsc_exact import SPSC, lam_min_of, periodic_probe_rounds
    b = BENCHMARKS[bname]
    env = make_env(b, seed)
    if method == "LinUCB":
        m = LinUCB(env, lam=b["lam"], delta=0.05, seed=seed).run()
    elif method == "random r-dim subspace, no probes":
        m = RandomSubspaceUCB(env, window=b["W"], lam=b["lam"], delta=0.05, seed=seed).run()
    elif method.startswith("runs"):
        m = SPSC_Algorithm1(env, probe_every=b["pe"], probe_cost=b["c"], window=b["W"], lam=b["lam"],
                            delta=0.05, seed=seed, normalize_gamma_by_d=True,
                            lazy_updates=method.startswith("runs with")).run()
    else:
        settings = dict(refresh="every", radius="runs") if method == "check" else {}
        # theta is fixed within every segment of these benchmarks, so S_delta = 0; the runs center at sigma^2
        m = SPSC(env, periodic_probe_rounds(env, b["pe"]), r=b["r"], W=b["W"], lam=b["lam"],
                 lam_min=lam_min_of(env), probe_cost=b["c"], delta=0.05, seed=seed,
                 sigma2_hat=env.sigma_eps ** 2, S_delta=0.0, **settings).run()
    return bname, method, seed, m.cumulative_costed_regret


def check_code(workers):
    from environments import LowRankLDSEnvironment
    from spsc_exact import K_inv, SPSC, lam_min_of, periodic_probe_rounds
    rng = np.random.default_rng(0)
    ok = True

    print("== 1. E[Kinv(y^2 u u^T)] = theta theta^T + (sigma^2/d) I under zero centering")
    for d in (5, 20):
        theta = rng.standard_normal(d)
        theta /= np.linalg.norm(theta)
        n, sig = 400_000, 0.3
        v = rng.standard_normal((n, d))
        u = np.sqrt(d) * v / np.linalg.norm(v, axis=1, keepdims=True)
        y2 = (u @ theta + sig * rng.standard_normal(n)) ** 2
        G = K_inv(np.einsum("n,ni,nj->ij", y2, u, u) / n, d)
        err = np.linalg.norm(G - np.outer(theta, theta) - sig ** 2 / d * np.eye(d), 2)
        print(f"   d={d}: ||mean G - target|| = {err:.4f}   (Monte Carlo, n={n})")
        ok &= err < 0.05

    print("== 2. refresh after the 1st, 2nd, 4th, ... probe of every segment")
    env = LowRankLDSEnvironment(d=20, r=3, K=4, T=2000, sigma_eps=0.3, n_actions=40, seed=1,
                                sigma_eta=0.04, piecewise_constant=True)
    probes = periodic_probe_rounds(env, 7)
    alg = SPSC(env, probes, r=3, W=400, lam=1.0, lam_min=lam_min_of(env), probe_cost=0.1, seed=0)
    m = alg.run()
    for start, length in zip(env.tau, env.segment_lengths):
        m_k = int(probes[start:start + length].sum())
        got = [N for t, N in alg.refresh_log if start <= t < start + length]
        ok &= got == [2 ** j for j in range(int(np.log2(m_k)) + 1)]
    print(f"   refresh counts {[N for _, N in alg.refresh_log][:10]}...  {'as expected' if ok else 'WRONG'}")
    diff = np.abs(m.cumulative_costed_regret - m.cumulative_control_regret - 0.1 * np.cumsum(probes)).max()
    print(f"   costed - control - c * probes: max |diff| = {diff:.1e}")
    ok &= diff < 1e-9

    print("== 3. r = d: Algorithm 1 equals ambient ridge-UCB on the same window")

    class Scaled(LowRankLDSEnvironment):
        # action norms in [0.5, 1], so the index has no exact ties when the window is empty
        def get_action_set(self, t, rng=None):
            A = super().get_action_set(t, rng)
            return A * np.random.default_rng(10_000 + t).uniform(0.5, 1.0, size=(len(A), 1))

    kw = dict(d=8, r=2, K=3, T=900, sigma_eps=0.3, n_actions=30, sigma_eta=0.04, piecewise_constant=True)
    for seed in (0, 1):
        env = Scaled(seed=seed, **kw)
        probes = periodic_probe_rounds(env, 11)
        alg = SPSC(env, probes, r=8, W=150, lam=1.0, lam_min=lam_min_of(env), probe_cost=0.1, seed=seed,
                   radius="runs")
        m_alg = alg.run()
        env2 = Scaled(seed=seed, **kw)
        rng2 = np.random.default_rng(seed)
        reg, win = np.zeros(env2.T), []
        for t in range(env2.T):
            if t in env2.tau:
                win = []
            A = env2.get_action_set(t, rng=rng2)
            r_opt = env2.optimal_reward(A, t)
            if probes[t]:
                z = rng2.standard_normal(8)
                x = np.sqrt(8) * z / (np.linalg.norm(z) + 1e-12)
                env2.step(x, t)
            else:
                win = [(s, x_s, y_s) for s, x_s, y_s in win if s >= t - 150]
                X = np.array([x_s for _, x_s, _ in win]).reshape(-1, 8)
                V = np.eye(8) + X.T @ X
                th = np.linalg.solve(V, X.T @ np.array([y_s for _, _, y_s in win]))
                beta = alg._radius(len(win), 0)
                width = np.sqrt(np.einsum("ij,ij->i", A, np.linalg.solve(V, A.T).T))
                x = A[int(np.argmax(A @ th + beta * width))]
                win.append((t, x, env2.step(x, t)))
            reg[t] = r_opt - x @ env2.theta[t]
        diff = np.abs(np.cumsum(reg) - m_alg.cumulative_control_regret).max()
        print(f"   seed {seed}: max |cumulative regret diff| = {diff:.1e}")
        ok &= diff < 1e-6

    print("== 4. Algorithm 1 with the two settings of App. E reproduces SPSC_Algorithm1")
    jobs = [(bn, mt, 0) for bn in BENCHMARKS for mt in ("check", "runs (every probe, standard radius)")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = {(bn, mt): c for bn, mt, _, c in ex.map(run_one, jobs)}
    for bn in BENCHMARKS:
        env = make_env(BENCHMARKS[bn], 0)
        drift = max(np.abs(env.theta[s:s + l] - env.theta[s]).max() for s, l in zip(env.tau, env.segment_lengths))
        diff = np.abs(out[(bn, "check")] - out[(bn, "runs (every probe, standard radius)")]).max()
        print(f"   {bn:22s} max |cumulative regret diff| = {diff:.1e}   (theta fixed in segments: {drift == 0})")
        ok &= diff < 1e-6 and drift == 0
    print("   PASS" if ok else "   FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not check_code(args.workers):
        sys.exit(1)

    jobs = [(bn, mt, s) for bn in BENCHMARKS for mt in METHODS for s in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [(bn, mt, s, float(c[-1])) for bn, mt, s, c in ex.map(run_one, jobs)]
    with open(os.path.join(HERE, "results", "experiment_alg1_settings.json"), "w") as f:
        json.dump(dict(rows=rows, benchmarks=BENCHMARKS), f, indent=1)
    print(f"\n== costed regret, mean +- SE over {args.seeds} seeds ({time.time() - t0:.0f}s)")
    for bn in BENCHMARKS:
        print(f"   {bn}")
        get = lambda mt: np.array([r[3] for r in rows if r[0] == bn and r[1] == mt])
        lin, runs = get("LinUCB").mean(), get(METHODS[1])
        for mt in METHODS:
            c = get(mt)
            se = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0.0
            paired = "" if mt in ("LinUCB", METHODS[1]) else \
                f"   vs runs {100 * (c.mean() / runs.mean() - 1):+5.1f}% (paired SE {100 * (c - runs).std(ddof=1) / np.sqrt(len(c)) / runs.mean():.1f})"
            print(f"      {mt:36s} {c.mean():8.1f} +- {se:5.1f}   vs LinUCB {100 * (c.mean() / lin - 1):+6.1f}%{paired}")


if __name__ == "__main__":
    main()
