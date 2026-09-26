"""
Algorithm 1 of the paper (SPSC with known segment boundaries), coded line by line.

    1  Input: segments, probe rounds T^probe, probe law Q, rank r, window W, ridge lam, centering sigma2_hat,
              a lower bound on lambda_min, and any orthonormal U_hat
    2  for each segment k:
    3      M <- 0, N <- 0, and empty the window
    4      for t in I_k:
    5          if t in T^probe:
    6              play u_t ~ Q, observe y_t, M <- M + Kinv((y_t^2 - sigma2_hat) u_t u_t^T), N <- N + 1
    7              if N is a power of two: U_hat <- TopEig_r(M / N)
    8          else:
    9              fit ridge regression (a_hat, V) on z(x_s) = U_hat^T x_s over the exploitation
                   rounds of the segment among the last W rounds
    10             play x_t = argmax_x z(x)^T a_hat + beta_tilde_t ||z(x)||_{V^{-1}}, observe y_t

The radius adds a projector-error term to the ridge radius, with the constants of the concentration results (App. B):
    beta_tilde_t = beta^(r,W) + R_A S_w sqrt(|W_t|) eps_bar_{k,t-1}
    beta^(r,W)   = sigma_eff sqrt(r log(1 + W R_A^2 / (lam r)) + 2 log(2 K T^2 / delta)) + sqrt(lam) S_w
    sigma_eff    = sigma_eps + R_A S_delta,  S_delta a known bound on ||theta_t - mu_t|| (2 S_w always works)
    eps_bar      = 1 if q < 2 q_star, else sqrt(2) C_sub sqrt(d log(12KdT/delta) / q),  q = probes before round t
    C_sub        = (8 sqrt(2) R_s + C') / lam_min,   R_s = 4 (S_w^2 + sigma_eps^2) log(8 T d^4 / delta) + sigma2_hat
    q_star       = C_sub^2 d log(12KdT/delta) / 4
C' absorbs the truncation bias of the matrix concentration theorem.  Its constant is not explicit, so the
code sets C' = 0, which only lowers eps_bar.

The benchmark runs use SPSC_Algorithm1 in algorithm.py.  It is this class with two settings (App. E):
refresh="every" (a new basis after every probe) and radius="runs" (the standard ridge radius
sigma_eps sqrt(r log((1 + n R_A^2/lam)/delta)) + sqrt(lam) S_w, with no eps_bar term).  The check in
experiment_alg1_settings.py confirms that the two produce identical runs.

Regret is eq. (dynreg): max_x x^T mu_t - x_t^T mu_t, plus c on probe rounds (x_t = u_t there).
subspace_error records ||(I - U U^T) mu_t|| / ||mu_t||, the part of the mean the basis misses.
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


def budget_probe_rounds(env, c0: float) -> np.ndarray:
    """m_k = min(l_k, ceil(c0 l_k^{2/3})) evenly spaced probes, starting with the first round of every segment."""
    probe = np.zeros(env.T, dtype=bool)
    for start, length in zip(env.tau, env.segment_lengths):
        m = min(length, int(np.ceil(c0 * length ** (2.0 / 3.0))))
        probe[start + np.floor(np.arange(m) * length / m).astype(int)] = True
    return probe


def lam_min_of(env) -> float:
    """Smallest nonzero eigenvalue of the segment second moments of mu_t (the lower bound Algorithm 1 is given)."""
    vals = []
    for start, length in zip(env.tau, env.segment_lengths):
        mu = env.theta[start:start + length]
        ev = np.linalg.eigvalsh(mu.T @ mu / length)
        vals.append(ev[ev > 1e-9 * ev[-1]].min())
    return float(min(vals))


def theory_constants(env, r: int, W: int, lam: float, delta: float, c: float, lam_min: float,
                     S_delta: float, sigma2_hat: float = 0.0) -> dict:
    """Constants of App. B for this instance, including the budget constant c0 of Theorem 4.1."""
    d, T, K = env.d, env.T, len(env.tau)
    R_A, S_w, sig = env.L_x, env.S, env.sigma_eps
    sigma_eff = sig + R_A * S_delta
    L_W = np.log(1 + W * R_A ** 2 / (lam * r))
    beta = sigma_eff * np.sqrt(r * L_W + 2 * np.log(2 * K * T ** 2 / delta)) + np.sqrt(lam) * S_w
    log_q = np.log(12 * K * d * T / delta)
    R_s = 4 * (S_w ** 2 + sig ** 2) * np.log(8 * T * d ** 4 / delta) + sigma2_hat
    C_sub = 8 * np.sqrt(2) * R_s / lam_min
    q_star = C_sub ** 2 * d * log_q / 4
    a0, b = 2.0, 3 + np.log2(T)
    B_prime = (4 * np.sqrt(6) * R_A * S_w * C_sub
               * np.sqrt(a0 * (1 + b) * (1 + np.log(T)) * r * d * L_W * log_q)
               + 8 * np.sqrt(2) * a0 * R_A * S_w * C_sub * np.sqrt(d * log_q))
    A = R_A * S_w + c
    c0 = (B_prime / (2 * A)) ** (2.0 / 3.0)
    return dict(sigma_eff=sigma_eff, beta=beta, R_s=R_s, C_sub=C_sub, q_star=q_star, c0=c0,
                m_k=[min(l, int(np.ceil(c0 * l ** (2.0 / 3.0)))) for l in env.segment_lengths])


class SPSC:
    """
    Algorithm 1.  probe_rounds is the set T^probe as a boolean array of length T, sigma2_hat the
    centering, S_delta the known bound on ||theta_t - mu_t|| (None means 2 S_w).
    refresh="every" and radius="runs" switch on the two settings of the benchmark runs.
    """

    def __init__(self, env, probe_rounds: np.ndarray, r: int, W: int, lam: float, lam_min: float,
                 probe_cost: float, delta: float = 0.05, seed: int = 0, sigma2_hat: float = 0.0,
                 S_delta: float | None = None, refresh: str = "doubling", radius: str = "box"):
        self.env, self.probe_rounds, self.r, self.W, self.lam = env, probe_rounds, r, W, lam
        self.c, self.delta, self.sigma2_hat = probe_cost, delta, sigma2_hat
        self.refresh, self.radius = refresh, radius
        self.rng = np.random.default_rng(seed)
        S_delta = 2 * env.S if S_delta is None else S_delta
        self.const = theory_constants(env, r, W, lam, delta, probe_cost, lam_min, S_delta, sigma2_hat)
        self.refresh_log = []            # (t, N) at every refresh, for the checks
        self.cond = []                   # lambda_min(Z^T Z)/|W_t| at exploit rounds with |W_t| >= 2r (condition WC)

    def _eps_bar(self, q: int) -> float:
        if q < 2 * self.const["q_star"]:
            return 1.0
        d, T, K = self.env.d, self.env.T, len(self.env.tau)
        return np.sqrt(2) * self.const["C_sub"] * np.sqrt(d * np.log(12 * K * d * T / self.delta) / q)

    def _radius(self, n: int, q: int) -> float:
        env = self.env
        if self.radius == "runs":
            arg = max(1.0 + min(n, self.W) * env.L_x ** 2 / self.lam, 1.0 + 1e-12)
            return env.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta)) + np.sqrt(self.lam) * env.S
        return self.const["beta"] + env.L_x * env.S * np.sqrt(n) * self._eps_bar(q)

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
                    u = np.sqrt(d) * z / (np.linalg.norm(z) + 1e-12)     # scaled-sphere probe
                    y = env.step(u, t)                     # line 6
                    M += K_inv((y ** 2 - self.sigma2_hat) * np.outer(u, u), d)
                    N += 1
                    if self.refresh == "every" or N & (N - 1) == 0:      # line 7
                        U = np.linalg.eigh(M / N)[1][:, -r:]
                        self.refresh_log.append((t, N))
                    x_t = u
                    metrics.probe_flags[t] = True
                else:
                    window = [(s, x_s, y_s) for s, x_s, y_s in window if s >= t - W]   # line 9
                    Zw = np.array([x_s for _, x_s, _ in window]).reshape(-1, d) @ U
                    yw = np.array([y_s for _, _, y_s in window])
                    V = self.lam * np.eye(r) + Zw.T @ Zw
                    if len(window) >= 2 * r:
                        self.cond.append(float(np.linalg.eigvalsh(Zw.T @ Zw)[0] / len(window)))
                    a_hat = np.linalg.solve(V, Zw.T @ yw)
                    beta_t = self._radius(len(window), N)
                    Z = action_set @ U                                                  # line 10
                    width = np.sqrt(np.einsum("ij,ij->i", Z, np.linalg.solve(V, Z.T).T))
                    x_t = action_set[int(np.argmax(Z @ a_hat + beta_t * width))]
                    y = env.step(x_t, t)
                    window.append((t, x_t, y))
                metrics.control_regret[t] = r_opt - float(x_t @ env.theta[t])
                metrics.costed_regret[t] = metrics.control_regret[t] + self.c * self.probe_rounds[t]
                mu = env.theta[t]
                metrics.subspace_error[t] = np.linalg.norm(mu - U @ (U.T @ mu)) / max(np.linalg.norm(mu), 1e-12)
        return metrics
