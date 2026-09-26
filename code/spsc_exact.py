"""
Algorithm 1 of the paper (SPSC with known segment boundaries), coded line by line.

    1  Input: segments, probe rounds T^probe, probe law Q, rank r, window W, ridge lam, centering sigma2_hat,
              and any orthonormal U_hat
    2  for each segment k:
    3      M <- 0, N <- 0
    4      for t in I_k:
    5          if t in T^probe:
    6              draw u_t ~ Q, play it, and observe y_t
    7              G_t <- Kinv((y_t^2 - sigma2_hat) u_t u_t^T)
    8              M <- M + G_t, N <- N + 1
    9              U_hat <- TopEig_r(M / N)
    10         else:
    11             W_t <- the exploitation rounds s of the segment with t - W <= s < t,  z_t(x) = U_hat^T x
    12             V_t <- lam I_r + sum_{s in W_t} z_t(x_s) z_t(x_s)^T,  b_t <- sum_{s in W_t} z_t(x_s) y_s
    13             a_hat <- V_t^{-1} b_t
    14             play x_t = argmax_x z_t(x)^T a_hat + beta_t ||z_t(x)||_{V_t^{-1}} and observe y_t

The radius is beta^(r,W) = sigma_eff sqrt(r log(1 + W R_A^2 / (lam r)) + 2 log(2 K T^2 / delta)) + sqrt(lam) S_w,
with sigma_eff = sigma_eps + R_A S_delta and S_delta a known bound on ||theta_t - mu_t|| (2 S_w always works).

Two options give the settings of the experiments.
- radius="runs" is the ridge radius of the benchmark runs (App. E), sigma_eps sqrt(r log((1 + n R_A^2/lam)/delta))
  + sqrt(lam) S_w with n the number of rounds in the window.  With it this class produces exactly the runs of
  SPSC_Algorithm1 in algorithm.py, which experiment_alg1_settings.py checks.
- refresh="doubling" recomputes the basis only after the 1st, 2nd, 4th, ... probe of a segment, the variant in the
  last column of Table 19.

Regret is eq. (dynreg): max_x x^T mu_t - x_t^T mu_t, plus c on probe rounds (x_t = u_t there).
subspace_error records ||(I - U U^T) mu_t|| / ||mu_t||, the part of the mean the basis misses.
cond records lambda_min(Z^T Z)/|W_t| at exploitation rounds with |W_t| >= 2r, the quantity of condition (WC).
"""
from __future__ import annotations

import numpy as np

from algorithm import RunMetrics


def K_inv(N: np.ndarray, d: int) -> np.ndarray:
    """K^{-1}(N) = (d+2)/(2d) N - tr(N)/(2d) I  (eq. K_inverse_intro)."""
    return (d + 2) / (2.0 * d) * N - np.trace(N) / (2.0 * d) * np.eye(d)


def periodic_probe_rounds(env, period: int) -> np.ndarray:
    """Evenly spaced probes at rate 1/period, starting with the first round of every segment."""
    probe = np.zeros(env.T, dtype=bool)
    for start, length in zip(env.tau, env.segment_lengths):
        probe[start:start + length:period] = True
    return probe


def box_radius(env, r: int, W: int, lam: float, delta: float, S_delta: float) -> float:
    """beta^(r,W) of Algorithm 1."""
    K, T = len(env.tau), env.T
    R_A, S_w = env.L_x, env.S
    sigma_eff = env.sigma_eps + R_A * S_delta
    return (sigma_eff * np.sqrt(r * np.log(1 + W * R_A ** 2 / (lam * r)) + 2 * np.log(2 * K * T ** 2 / delta))
            + np.sqrt(lam) * S_w)


class SPSC:
    """
    Algorithm 1.  probe_rounds is the set T^probe as a boolean array of length T, sigma2_hat the
    centering, S_delta the known bound on ||theta_t - mu_t|| (None means 2 S_w).
    """

    def __init__(self, env, probe_rounds: np.ndarray, r: int, W: int, lam: float, probe_cost: float,
                 delta: float = 0.05, seed: int = 0, sigma2_hat: float = 0.0, S_delta: float | None = None,
                 refresh: str = "every", radius: str = "box"):
        assert refresh in ("every", "doubling") and radius in ("box", "runs")
        self.env, self.probe_rounds, self.r, self.W, self.lam = env, probe_rounds, r, W, lam
        self.c, self.delta, self.sigma2_hat = probe_cost, delta, sigma2_hat
        self.refresh, self.radius = refresh, radius
        self.rng = np.random.default_rng(seed)
        S_delta = 2 * env.S if S_delta is None else S_delta
        self.beta = box_radius(env, r, W, lam, delta, S_delta)
        self.refresh_log = []            # (t, N) at every refresh, for the checks
        self.cond = []                   # lambda_min(Z^T Z)/|W_t| at exploit rounds with |W_t| >= 2r (condition WC)

    def _radius(self, n: int) -> float:
        env = self.env
        if self.radius == "runs":
            arg = max(1.0 + min(n, self.W) * env.L_x ** 2 / self.lam, 1.0 + 1e-12)
            return env.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta)) + np.sqrt(self.lam) * env.S
        return self.beta

    def run(self) -> RunMetrics:
        env, r, W = self.env, self.r, self.W
        d, T = env.d, env.T
        metrics = RunMetrics(name="SPSC (Algorithm 1)", T=T)

        U = np.eye(d, r)                                   # line 1: any orthonormal U_hat
        for start, length in zip(env.tau, env.segment_lengths):   # line 2
            M, N, window = np.zeros((d, d)), 0, []         # line 3: window holds (s, x_s, y_s)
            for t in range(start, start + length):         # line 4
                action_set = env.get_action_set(t, rng=self.rng)
                r_opt = env.optimal_reward(action_set, t)
                if self.probe_rounds[t]:                   # line 5
                    z = self.rng.standard_normal(d)
                    u = np.sqrt(d) * z / (np.linalg.norm(z) + 1e-12)     # line 6: scaled-sphere probe
                    y = env.step(u, t)
                    M += K_inv((y ** 2 - self.sigma2_hat) * np.outer(u, u), d)   # lines 7-8
                    N += 1
                    if self.refresh == "every" or N & (N - 1) == 0:      # line 9
                        U = np.linalg.eigh(M / N)[1][:, -r:]
                        self.refresh_log.append((t, N))
                    x_t = u
                    metrics.probe_flags[t] = True
                else:
                    window = [(s, x_s, y_s) for s, x_s, y_s in window if s >= t - W]   # line 11
                    Zw = np.array([x_s for _, x_s, _ in window]).reshape(-1, d) @ U
                    yw = np.array([y_s for _, _, y_s in window])
                    V = self.lam * np.eye(r) + Zw.T @ Zw                                # line 12
                    if len(window) >= 2 * r:
                        self.cond.append(float(np.linalg.eigvalsh(Zw.T @ Zw)[0] / len(window)))
                    a_hat = np.linalg.solve(V, Zw.T @ yw)                               # line 13
                    beta_t = self._radius(len(window))
                    Z = action_set @ U                                                  # line 14
                    width = np.sqrt(np.einsum("ij,ij->i", Z, np.linalg.solve(V, Z.T).T))
                    x_t = action_set[int(np.argmax(Z @ a_hat + beta_t * width))]
                    y = env.step(x_t, t)
                    window.append((t, x_t, y))
                metrics.control_regret[t] = r_opt - float(x_t @ env.theta[t])
                metrics.costed_regret[t] = metrics.control_regret[t] + self.c * self.probe_rounds[t]
                mu = env.theta[t]
                metrics.subspace_error[t] = np.linalg.norm(mu - U @ (U.T @ mu)) / max(np.linalg.norm(mu), 1e-12)
        return metrics
