"""
Single-Play Subspace-Aware Control and baselines.

Algorithms
----------
SPSC_Algorithm1  -- Algorithm 1 from the paper:
    - Single-play: on probe rounds x_t^dep = u_t (no separate UCB action)
    - Windowed design matrix: only exploitation rounds in [t-W, t) per segment
    - Control regret on probe rounds = r_opt - u_t^T theta_t  (actual, not a proxy)
    - Costed regret adds c per probe round on top of control regret

LinUCB           -- standard ambient-space ridge UCB baseline (no subspace)
OracleLinUCB     -- UCB with oracle knowledge of true subspace (performance ceiling)

K^{-1} operator for sphere probes u = sqrt(d) * z/||z||, z~N(0,I_d)
----------------------------------------------------------------------
  K(M) = d/(d+2) * (tr(M) I_d + 2M)
  K^{-1}(N) = (d+2)/(2d) * N - tr(N)/(2d) * I_d
  Proof: set M = K^{-1}(N), compute tr(M) = tr(N)/d, substitute back. QED.

Metrics per round
-----------------
  costed_regret_t  = r_t* - r_t + c * 1{probe}
  control_regret_t = r_t* - r_t
  probe_flag_t     = bool
  subspace_error_t = ||P_hat_k - P_k*||_2  (set only on probe rounds)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from environments import LowRankLDSEnvironment


# ---------------------------------------------------------------------------
# K^{-1} operator (sphere probes)
# ---------------------------------------------------------------------------

def K_inverse(N: np.ndarray, d: int) -> np.ndarray:
    """
    K^{-1}(N) = (d+2)/(2d) * N - tr(N)/(2d) * I_d

    Valid for probe distribution u = sqrt(d) * z/||z||, z~N(0,I_d),
    which satisfies E[uu^T] = I_d and ||u||_2 = sqrt(d) exactly.
    """
    c = (d + 2) / (2.0 * d)
    return c * N - (np.trace(N) / (2.0 * d)) * np.eye(d)


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------
# (defined once; used by all algorithm classes below)

@dataclass
class RunMetrics:
    name: str
    T: int
    costed_regret: np.ndarray = field(init=False)
    control_regret: np.ndarray = field(init=False)
    probe_flags: np.ndarray = field(init=False)
    subspace_error: np.ndarray = field(init=False)

    def __post_init__(self):
        self.costed_regret  = np.zeros(self.T)
        self.control_regret = np.zeros(self.T)
        self.probe_flags    = np.zeros(self.T, dtype=bool)
        self.subspace_error = np.full(self.T, np.nan)

    @property
    def cumulative_costed_regret(self) -> np.ndarray:
        return np.cumsum(self.costed_regret)

    @property
    def cumulative_control_regret(self) -> np.ndarray:
        return np.cumsum(self.control_regret)

    @property
    def total_probes(self) -> int:
        return int(self.probe_flags.sum())


# ---------------------------------------------------------------------------
# Algorithm 1 — faithful single-play implementation
# ---------------------------------------------------------------------------

class SPSC_Algorithm1:
    """
    Implementation of Algorithm 1 from the paper:
    "Single-Play Subspace-Aware Control with Windowed r-Dimensional Exploitation"

    Single-play invariant
    ---------------------
    On probe rounds: x_t^dep = u_t  (the probe IS the only deployment).
    On exploitation rounds: x_t^dep = argmax UCB over A_t.
    Exactly one env.step() call per round.

    Windowed design matrix
    ----------------------
    V_tilde_t and b_tilde_t are built from exploitation rounds s in the
    current segment with t - W <= s < t.  This matches the W_t definition
    in Algorithm 1 exactly.  No cumulative summation across the whole segment.

    Regret accounting
    -----------------
    control_regret_t = r_opt - x_t^dep^T theta_t   (actual, every round)
    costed_regret_t  = control_regret_t + c * 1{probe round}
    """

    def __init__(
        self,
        env: "LowRankLDSEnvironment",
        probe_every: int = 5,
        probe_cost: float = 0.1,
        window: int = 200,
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 0,
        normalize_gamma_by_d: bool = False,
        lazy_updates: bool = False,
    ):
        self.env         = env
        self.probe_every = probe_every
        self.c           = probe_cost
        self.W           = window
        self.lam         = lam
        self.delta       = delta
        self.rng         = np.random.default_rng(seed)
        # Lazy (doubling-schedule) basis refresh: recompute U_hat only when the
        # per-segment probe count hits 1, 2, 4, 8, ...  A fixed basis within
        # each block makes the windowed elliptical-potential lemma standard per
        # block.  Default False keeps the every-probe refresh used in all runs.
        self.lazy_updates  = lazy_updates
        self.n_eig_updates = 0
        self.max_buf_len   = 0   # peak raw-action window buffer (memory audit)

        d, r = env.d, env.r
        self.d = d
        self.r = r

        self.L_x       = env.L_x
        self.sigma_eps = env.sigma_eps
        self.S         = env.S
        # K^{-1} operator norm for sphere probes
        self.K_inv_op  = (d + 2) / (2.0 * d)
        # When True, R_X_hat in gamma is divided by d, removing the factor that
        # grows with d due to probe vector norm sqrt(d).  This is appropriate for
        # d-scaling experiments where ||G_t||_op ∝ d but the signal norm S^2 is
        # d-independent.  Defaults to False to leave Experiments 1-4 unchanged.
        self._normalize_gamma_by_d = normalize_gamma_by_d
        # Per-segment observed G_t norms (for data-adaptive gamma)
        self._G_norms: list = []

    # ------------------------------------------------------------------
    # Probe schedule: first round of each segment + every probe_every steps
    # ------------------------------------------------------------------

    def _is_probe(self, t: int, k: int) -> bool:
        seg_start = self.env.tau[k]
        if t == seg_start:
            return True
        return ((t - seg_start) % self.probe_every) == 0

    # ------------------------------------------------------------------
    # Confidence radii
    # ------------------------------------------------------------------

    def _beta(self, n_exploit: int) -> float:
        """
        beta_t^{(r,W)}: confidence radius in r-dimensional reduced space,
        calibrated to the effective window size (at most W exploitation steps).

        beta = sigma_eps * sqrt(r * log((1 + n_exploit * L_x^2 / lam) / delta))
               + sqrt(lam) * S
        """
        effective_n = min(n_exploit, self.W)
        arg = max(1.0 + effective_n * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def _gamma(self, m_probe: int, M_hat: Optional[np.ndarray] = None) -> float:
        """
        gamma_t = Gamma_k + V_{k,t}(W): subspace mismatch + local drift radius.
        Uses data-adaptive eigengap and empirical G_t norms.
        """
        if m_probe < 2:
            return float(self.S)

        R_X_hat = (
            float(np.percentile(self._G_norms, 90))
            if len(self._G_norms) >= 2
            else self.K_inv_op * self.d * ((self.S + 1.0) ** 2 + self.sigma_eps ** 2)
        )
        # Optional: normalize by d so gamma is d-independent for sphere probes.
        # ||G_t||_op ∝ d due to ||u_t|| = sqrt(d); the signal norm S^2 is d-independent.
        if self._normalize_gamma_by_d:
            R_X_hat = R_X_hat / self.d

        if M_hat is not None:
            eigs = np.sort(np.linalg.eigvalsh(M_hat))[::-1]
            gap = max(float(eigs[self.r - 1] - eigs[self.r]), 1e-4) if len(eigs) > self.r else 1e-4
        else:
            gap = 1e-4

        return (
            8.0 * self.S * R_X_hat / gap
            * np.sqrt(np.log(2.0 * self.d / self.delta) / m_probe)
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> "RunMetrics":
        env = self.env
        d, r, T = self.d, self.r, env.T
        metrics = RunMetrics(name="SPSC-Alg1", T=T)

        # Subspace state
        M_sum       = np.zeros((d, d))
        U_hat       = np.eye(d, r)            # initial: first r standard basis cols
        P_hat       = U_hat @ U_hat.T
        m_probe_seg = 0
        current_k   = -1

        # Exploitation buffer: list of (x_s, y_s, s) for exploitation rounds
        # within the current segment.  Stores raw actions (not projected features)
        # so they can be reprojected with the current U_hat at each step —
        # this eliminates the stale-feature problem from changing U_hat.
        expl_buf: List[tuple] = []

        for t in range(T):
            k = env.seg_of[t]

            # ---- Segment boundary: reset accumulator, clear buffers ----
            if k != current_k:
                M_sum       = np.zeros((d, d))
                m_probe_seg = 0
                self._G_norms = []
                expl_buf    = []
                # Carry forward U_hat / P_hat until first probe updates them
                current_k = k

            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            # ==============================================================
            # PROBE ROUND — single play: x_t^dep = u_t
            # ==============================================================
            if self._is_probe(t, k):
                metrics.probe_flags[t] = True

                # Draw isotropic sphere probe: u = sqrt(d) * z/||z||
                z_probe = self.rng.standard_normal(d)
                u_t     = np.sqrt(d) * z_probe / (np.linalg.norm(z_probe) + 1e-12)

                # Deploy u_t — this is the ONLY action this round
                y_t = env.step(u_t, t)

                # Subspace estimation update
                s_t = y_t ** 2 - self.sigma_eps ** 2
                G_t = K_inverse(s_t * np.outer(u_t, u_t), d)

                self._G_norms.append(float(np.linalg.norm(G_t, ord=2)))
                M_sum       += G_t
                m_probe_seg += 1

                refresh = (not self.lazy_updates) or (m_probe_seg & (m_probe_seg - 1) == 0)
                if refresh:
                    M_hat_now = M_sum / m_probe_seg
                    eig_vals, eig_vecs = np.linalg.eigh(M_hat_now)
                    U_hat = eig_vecs[:, -r:]       # top-r eigenvectors, d x r
                    P_hat = U_hat @ U_hat.T
                    self.n_eig_updates += 1

                # Subspace error diagnostic (oracle ground truth)
                P_true = env.segment_projector(k)
                metrics.subspace_error[t] = np.linalg.norm(P_hat - P_true, ord=2)

                # Regret: actual reward from playing u_t
                r_t = float(u_t @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t]  = r_opt - r_t + self.c

            # ==============================================================
            # EXPLOITATION ROUND — windowed reduced-space UCB
            # ==============================================================
            else:
                # Build windowed design from exploitation rounds in [t-W, t)
                # within the current segment (Algorithm 1: W_t definition).
                # Reproject all stored actions with the CURRENT U_hat so all
                # features in the window share a consistent subspace basis —
                # eliminates regression inconsistency from changing U_hat.
                window_entries = [(x_s, y_s) for (x_s, y_s, s) in expl_buf if s >= t - self.W]

                V_tilde = self.lam * np.eye(r)
                b_tilde = np.zeros(r)
                n_exploit = len(window_entries)
                for x_s, y_s in window_entries:
                    z_s = U_hat.T @ x_s   # reproject with current U_hat
                    V_tilde += np.outer(z_s, z_s)
                    b_tilde += z_s * y_s

                a_hat   = np.linalg.solve(V_tilde, b_tilde)
                beta_t  = self._beta(n_exploit)

                # UCB index for each candidate action (Algorithm 1, line 10)
                Z       = action_set @ U_hat                          # (n_a, r)
                V_inv_Z = np.linalg.solve(V_tilde, Z.T).T            # (n_a, r)
                ellip   = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z)) # (n_a,)
                ucb     = Z @ a_hat + beta_t * ellip

                best_idx = int(np.argmax(ucb))
                x_dep    = action_set[best_idx]

                # Single play: deploy x_dep
                y_t = env.step(x_dep, t)

                # Store raw action (not projected feature) so future steps
                # can reproject with their current U_hat
                expl_buf.append((x_dep, y_t, t))
                if len(expl_buf) > self.max_buf_len:
                    self.max_buf_len = len(expl_buf)

                # Trim buffer to keep only rounds within [t-W, t] (saves memory)
                if len(expl_buf) > self.W + 10:
                    expl_buf = [(x_s, y_s, s) for (x_s, y_s, s) in expl_buf
                                if s >= t - self.W]

                # Subspace estimate unchanged on exploitation rounds
                r_t = float(x_dep @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Standard LinUCB (no subspace, full d x d design matrix)
# ---------------------------------------------------------------------------

class LinUCB:
    """
    Ridge regression UCB in full ambient space R^d.
    No probing, no subspace. Statistical regret O(d sqrt(T)).
    Resets at segment boundaries (same oracle info as SPSC).
    """

    def __init__(
        self,
        env: LowRankLDSEnvironment,
        probe_cost: float = 0.1,
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 1,
        forgetting_factor: float = 1.0,
    ):
        self.env       = env
        self.c         = probe_cost
        self.lam       = lam
        self.delta     = delta
        self.rng       = np.random.default_rng(seed)
        self.ff        = forgetting_factor
        self.S         = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x       = env.L_x

    def _beta(self, t: int) -> float:
        d   = self.env.d
        arg = max(1.0 + t * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(d * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def run(self) -> RunMetrics:
        env     = self.env
        d, T    = env.d, env.T
        metrics = RunMetrics(name="LinUCB", T=T)

        V       = self.lam * np.eye(d)
        b       = np.zeros(d)
        current_k = -1

        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                V = self.lam * np.eye(d)
                b = np.zeros(d)
                current_k = k

            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            theta_hat  = np.linalg.solve(V, b)
            beta_t     = self._beta(t)

            V_inv_A = np.linalg.solve(V, action_set.T).T
            ucb     = (
                action_set @ theta_hat
                + beta_t * np.sqrt(np.einsum('ij,ij->i', action_set, V_inv_A))
            )
            x_dep = action_set[int(np.argmax(ucb))]
            y     = env.step(x_dep, t)

            V = self.ff * V + np.outer(x_dep, x_dep)
            b = self.ff * b + x_dep * y

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Oracle LinUCB (knows true P_k* at all times, operates in R^r)
# ---------------------------------------------------------------------------

class OracleLinUCB:
    """
    UCB with oracle knowledge of the true r-dim subspace P_k* at every t.
    Uses the same windowed design matrix as SPSC_Algorithm1 for a fair comparison:
    only exploitation rounds in [t-W, t) within the current segment contribute
    to V_tilde and b_tilde.
    """

    def __init__(
        self,
        env: LowRankLDSEnvironment,
        window: int = 100,
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 2,
    ):
        self.env       = env
        self.W         = window
        self.lam       = lam
        self.delta     = delta
        self.rng       = np.random.default_rng(seed)
        self.S         = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x       = env.L_x

    def _beta_r(self, n_exploit: int) -> float:
        r   = self.env.r
        effective_n = min(n_exploit, self.W)
        arg = max(1.0 + effective_n * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(r * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def run(self) -> RunMetrics:
        env     = self.env
        d, r, T = env.d, env.r, env.T
        metrics = RunMetrics(name="Oracle-LinUCB", T=T)

        expl_buf: List[tuple] = []   # (z_s, y_s, s)
        current_k = -1

        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                expl_buf  = []
                current_k = k

            U_true     = env.B_list[k]
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            # Windowed design — same W as SPSC_Algorithm1
            window_buf = [(z_s, y_s) for (z_s, y_s, s) in expl_buf if s >= t - self.W]
            V_tilde = self.lam * np.eye(r)
            b_tilde = np.zeros(r)
            for z_s, y_s in window_buf:
                V_tilde += np.outer(z_s, z_s)
                b_tilde += z_s * y_s

            Z       = action_set @ U_true
            a_hat   = np.linalg.solve(V_tilde, b_tilde)
            beta_t  = self._beta_r(len(window_buf))

            V_inv_Z = np.linalg.solve(V_tilde, Z.T).T
            ucb     = Z @ a_hat + beta_t * np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))

            x_dep  = action_set[int(np.argmax(ucb))]
            y      = env.step(x_dep, t)
            z_dep  = U_true.T @ x_dep

            expl_buf.append((z_dep, y, t))
            if len(expl_buf) > self.W + 10:
                expl_buf = [(z_s, y_s, s) for (z_s, y_s, s) in expl_buf
                            if s >= t - self.W]

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Sliding-Window LinUCB (Cheung+ NeurIPS 2019)
# ---------------------------------------------------------------------------

class SWLinUCB:
    """Sliding-window LinUCB. Resets at oracle segment boundaries."""
    def __init__(self, env, window=200, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.W = window
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x

    def _beta(self, n):
        d = self.env.d
        arg = max(1.0 + n * self.L_x**2 / self.lam, 1.0 + 1e-12)
        return self.sigma_eps * np.sqrt(d * np.log(arg / self.delta)) + np.sqrt(self.lam) * self.S

    def run(self):
        env = self.env
        d, T = env.d, env.T
        metrics = RunMetrics(name="SW-LinUCB", T=T)
        buf = []
        current_k = -1
        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                buf = []
                current_k = k
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            win = [(xs, ys) for xs, ys, s in buf if s >= t - self.W]
            V = self.lam * np.eye(d)
            b = np.zeros(d)
            for xs, ys in win:
                V += np.outer(xs, xs)
                b += xs * ys
            theta_hat = np.linalg.solve(V, b)
            beta_t = self._beta(len(win))
            V_inv_A = np.linalg.solve(V, action_set.T).T
            ucb = action_set @ theta_hat + beta_t * np.sqrt(
                np.einsum('ij,ij->i', action_set, V_inv_A))
            x_dep = action_set[int(np.argmax(ucb))]
            y = env.step(x_dep, t)
            buf.append((x_dep, y, t))
            if len(buf) > self.W + 10:
                buf = [(xs, ys, s) for xs, ys, s in buf if s >= t - self.W]
            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t
        return metrics


# ---------------------------------------------------------------------------
# Baseline: Periodic-Restart LinUCB
# ---------------------------------------------------------------------------

class RestartLinUCB:
    """Periodic-restart LinUCB. Restarts every restart_period rounds."""
    def __init__(self, env, restart_period=500, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.restart_period = restart_period
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x

    def _beta(self, n):
        d = self.env.d
        arg = max(1.0 + n * self.L_x**2 / self.lam, 1.0 + 1e-12)
        return self.sigma_eps * np.sqrt(d * np.log(arg / self.delta)) + np.sqrt(self.lam) * self.S

    def run(self):
        env = self.env
        d, T = env.d, env.T
        metrics = RunMetrics(name="Restart-LinUCB", T=T)
        V = self.lam * np.eye(d)
        b = np.zeros(d)
        n_since = 0
        for t in range(T):
            if n_since >= self.restart_period:
                V = self.lam * np.eye(d)
                b = np.zeros(d)
                n_since = 0
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            theta_hat = np.linalg.solve(V, b)
            beta_t = self._beta(n_since)
            V_inv_A = np.linalg.solve(V, action_set.T).T
            ucb = action_set @ theta_hat + beta_t * np.sqrt(
                np.einsum('ij,ij->i', action_set, V_inv_A))
            x_dep = action_set[int(np.argmax(ucb))]
            y = env.step(x_dep, t)
            V += np.outer(x_dep, x_dep)
            b += x_dep * y
            n_since += 1
            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t
        return metrics


# ---------------------------------------------------------------------------
# Baseline: Oracle-Reset LinUCB (resets at true change points)
# ---------------------------------------------------------------------------

class OracleResetLinUCB:
    """LinUCB with oracle knowledge of change-point locations."""
    def __init__(self, env, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x

    def _beta(self, n):
        d = self.env.d
        arg = max(1.0 + n * self.L_x**2 / self.lam, 1.0 + 1e-12)
        return self.sigma_eps * np.sqrt(d * np.log(arg / self.delta)) + np.sqrt(self.lam) * self.S

    def run(self):
        env = self.env
        d, T = env.d, env.T
        metrics = RunMetrics(name="Reset-LinUCB", T=T)
        V = self.lam * np.eye(d)
        b = np.zeros(d)
        n_since = 0
        current_k = -1
        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                V = self.lam * np.eye(d)
                b = np.zeros(d)
                n_since = 0
                current_k = k
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            theta_hat = np.linalg.solve(V, b)
            beta_t = self._beta(n_since)
            V_inv_A = np.linalg.solve(V, action_set.T).T
            ucb = action_set @ theta_hat + beta_t * np.sqrt(
                np.einsum('ij,ij->i', action_set, V_inv_A))
            x_dep = action_set[int(np.argmax(ucb))]
            y = env.step(x_dep, t)
            V += np.outer(x_dep, x_dep)
            b += x_dep * y
            n_since += 1
            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t
        return metrics


# ---------------------------------------------------------------------------
# Baseline: LowRank Reward UCB (LowESTR/VOFUL spirit)
# ---------------------------------------------------------------------------

class LowRankRewardUCB:
    """Learn subspace from reward data, r-dim UCB."""
    def __init__(self, env, window=200, pca_warmup=50, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.W = window
        self.pca_warmup = pca_warmup
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x

    def _beta_r(self, n, gamma=0.0):
        r = self.env.r
        arg = max(1.0 + n * self.L_x**2 / self.lam, 1.0 + 1e-12)
        return (self.sigma_eps * np.sqrt(r * np.log(arg / self.delta))
                + np.sqrt(self.lam) * self.S
                + gamma * self.S * self.L_x)

    def run(self):
        env = self.env
        d, r, T = env.d, env.r, env.T
        metrics = RunMetrics(name="LowRank-Reward-UCB", T=T)
        xy_buf = []
        expl_buf = []
        P_hat = np.eye(d)[:, :r]
        current_k = -1
        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                xy_buf = []
                expl_buf = []
                P_hat = np.eye(d)[:, :r]
                current_k = k
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            if len(xy_buf) >= self.pca_warmup:
                M = np.zeros((d, d))
                recent = xy_buf[-(self.W):]
                for x_s, y_s, s in recent:
                    M += np.outer(x_s * y_s, x_s * y_s)
                M /= len(recent)
                try:
                    eigvals, eigvecs = np.linalg.eigh(M)
                    P_hat = eigvecs[:, -r:]
                except Exception:
                    pass
            Z = action_set @ P_hat
            win = [(zs, ys) for zs, ys, s in expl_buf if s >= t - self.W]
            V_r = self.lam * np.eye(r)
            b_r = np.zeros(r)
            for zs, ys in win:
                V_r += np.outer(zs, zs)
                b_r += zs * ys
            a_hat = np.linalg.solve(V_r, b_r)
            beta_t = self._beta_r(len(win), gamma=0.3)
            V_inv_Z = np.linalg.solve(V_r, Z.T).T
            ucb = Z @ a_hat + beta_t * np.sqrt(
                np.einsum('ij,ij->i', Z, V_inv_Z))
            idx = int(np.argmax(ucb))
            x_dep = action_set[idx]
            z_dep = Z[idx]
            y = env.step(x_dep, t)
            xy_buf.append((x_dep, y, t))
            expl_buf.append((z_dep, y, t))
            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t
        return metrics


# ---------------------------------------------------------------------------
# SPSC with robustness knobs (for Experiments A-C)
# ---------------------------------------------------------------------------

class SPSC_Robustness:
    """SPSC Algorithm 1 with robustness knobs for Experiments A-C."""

    def __init__(self, env, probe_every=30, probe_cost=0.1, window=200,
                 lam=1.0, delta=0.05, seed=0,
                 delta_sigma=0.0, eps_cross=0.0, coverage_dim=None):
        self.env = env
        self.probe_every = probe_every
        self.c = probe_cost
        self.W = window
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        d, r = env.d, env.r
        self.d, self.r = d, r
        self.L_x = env.L_x
        self.sigma_eps = env.sigma_eps
        self.S = env.S
        self.delta_sigma = delta_sigma
        self.eps_cross = eps_cross
        self.coverage_dim = coverage_dim if coverage_dim is not None else d

    def _is_probe(self, t, k):
        seg_start = self.env.tau[k]
        if t == seg_start:
            return True
        return ((t - seg_start) % self.probe_every) == 0

    def _beta(self, n):
        eff = min(n, self.W)
        arg = max(1.0 + eff * self.L_x**2 / self.lam, 1.0 + 1e-12)
        return (self.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta))
                + np.sqrt(self.lam) * self.S)

    def _gamma(self, m, G_norms, M_hat=None):
        if m < 2:
            return float(self.S)
        K_inv_op = (self.d + 2) / (2.0 * self.d)
        R_X_hat = (float(np.percentile(G_norms, 90)) if len(G_norms) >= 2
                   else K_inv_op * self.d * ((self.S + 1.0)**2 + self.sigma_eps**2))
        if M_hat is not None:
            eigs = np.sort(np.linalg.eigvalsh(M_hat))[::-1]
            gap = max(float(eigs[self.r - 1] - eigs[self.r]), 1e-4) if len(eigs) > self.r else 1e-4
        else:
            gap = 1e-4
        return 8.0 * self.S * R_X_hat / gap * np.sqrt(
            np.log(2.0 * self.d / self.delta) / m)

    def run(self):
        env = self.env
        d, r, T = self.d, self.r, env.T
        metrics = RunMetrics(name="SPSC-Robust", T=T)
        M_sum = np.zeros((d, d))
        U_hat = np.eye(d, r)
        m_probe_seg = 0
        current_k = -1
        G_norms = []
        expl_buf = []
        sigma_sq_used = self.sigma_eps**2 + self.delta_sigma
        for t in range(T):
            k = env.seg_of[t]
            if k != current_k:
                M_sum = np.zeros((d, d))
                m_probe_seg = 0
                G_norms = []
                expl_buf = []
                current_k = k
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            if self._is_probe(t, k):
                metrics.probe_flags[t] = True
                z_probe = self.rng.standard_normal(d)
                if self.coverage_dim < d:
                    z_probe[self.coverage_dim:] = 0.0
                norm_z = np.linalg.norm(z_probe)
                if norm_z < 1e-12:
                    z_probe[0] = 1.0
                    norm_z = 1.0
                u_t = np.sqrt(d) * z_probe / norm_z
                eps_base = self.rng.normal(0.0, self.sigma_eps)
                eps_base = np.clip(eps_base, -env.L_eps, env.L_eps)
                cross_term = self.eps_cross * float(u_t @ env.theta[t])
                y_t = float(u_t @ env.theta[t]) + eps_base + cross_term
                s_t = y_t**2 - sigma_sq_used
                G_t = K_inverse(s_t * np.outer(u_t, u_t), d)
                G_norms.append(float(np.linalg.norm(G_t, ord=2)))
                M_sum += G_t
                m_probe_seg += 1
                M_hat = None
                if m_probe_seg >= 2:
                    M_hat = M_sum / m_probe_seg
                    _, eig_vecs = np.linalg.eigh(M_hat)
                    U_hat = eig_vecs[:, -r:]
                P_true = env.segment_projector(k)
                P_hat = U_hat @ U_hat.T
                metrics.subspace_error[t] = np.linalg.norm(P_hat - P_true, ord=2)
                r_t = float(u_t @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t + self.c
            else:
                win = [(xs, ys) for xs, ys, s in expl_buf if s >= t - self.W]
                V = self.lam * np.eye(r)
                b = np.zeros(r)
                for xs, ys in win:
                    zs = U_hat.T @ xs
                    V += np.outer(zs, zs)
                    b += zs * ys
                a_hat = np.linalg.solve(V, b)
                bt = self._beta(len(win))
                M_hat_curr = M_sum / m_probe_seg if m_probe_seg >= 2 else None
                gt = self._gamma(m_probe_seg, G_norms, M_hat_curr)
                Z = action_set @ U_hat
                ViZ = np.linalg.solve(V, Z.T).T
                el = np.sqrt(np.einsum('ij,ij->i', Z, ViZ))
                xn = np.linalg.norm(action_set, axis=1)
                ucb = Z @ a_hat + bt * el + gt * xn
                x_dep = action_set[int(np.argmax(ucb))]
                eps_base = self.rng.normal(0.0, self.sigma_eps)
                eps_base = np.clip(eps_base, -env.L_eps, env.L_eps)
                cross_term = self.eps_cross * float(x_dep @ env.theta[t])
                y_t = float(x_dep @ env.theta[t]) + eps_base + cross_term
                expl_buf.append((x_dep, y_t, t))
                if len(expl_buf) > self.W + 10:
                    expl_buf = [(xs, ys, s) for xs, ys, s in expl_buf
                                if s >= t - self.W]
                r_t = float(x_dep @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t
        return metrics


# ---------------------------------------------------------------------------
# Algorithm 2 — Adaptive SPSC (unknown segment boundaries)
# ---------------------------------------------------------------------------

class SPSC_Adaptive:
    """
    Algorithm 2: Adaptive Single-Play Subspace-Calibrated Optimism.

    Removes the oracle segment-boundary assumption of Algorithm 1
    via a reward-degradation change-point detector.

    The detector tracks a CUSUM-like statistic on exploitation rewards:
    if the recent per-round reward drops significantly below the
    historical baseline (indicating the learned subspace has become
    stale after a regime change), the algorithm enters RECOVERY mode.

    In RECOVERY, it deploys m_relearn consecutive probes to rebuild
    the subspace from scratch, then transitions back to NORMAL mode.

    This is a practical instantiation of Algorithm 2 from the paper,
    replacing the theoretical M_hat-based detector (which requires
    sample sizes impractical at moderate d) with a reward-based
    signal that directly measures exploitation quality.
    """

    def __init__(
        self,
        env: "LowRankLDSEnvironment",
        probe_every: int = 5,
        probe_cost: float = 0.1,
        window: int = 200,
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 0,
        # Detector parameters
        m_relearn: int = 40,            # probes in recovery phase
        det_window: int = 50,           # exploitation rounds for recent reward window
        cusum_threshold: float = 3.0,   # multiples of reward std for detection
        warmup: int = 100,              # exploitation rounds before detector activates
    ):
        self.env          = env
        self.probe_every  = probe_every
        self.c            = probe_cost
        self.W            = window
        self.lam          = lam
        self.delta        = delta
        self.rng          = np.random.default_rng(seed)

        d, r = env.d, env.r
        self.d = d
        self.r = r

        self.L_x       = env.L_x
        self.sigma_eps = env.sigma_eps
        self.S         = env.S
        self.K_inv_op  = (d + 2) / (2.0 * d)

        # Detector config
        self.m_relearn       = m_relearn
        self.det_window      = det_window
        self.cusum_threshold = cusum_threshold
        self.warmup          = warmup

    # ------------------------------------------------------------------
    # Confidence radii (same as Algorithm 1)
    # ------------------------------------------------------------------

    def _beta(self, n_exploit: int) -> float:
        effective_n = min(n_exploit, self.W)
        arg = max(1.0 + effective_n * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def _gamma(self, m_probe: int, G_norms: list, M_hat=None) -> float:
        if m_probe < 2:
            return float(self.S)
        R_X_hat = (
            float(np.percentile(G_norms, 90))
            if len(G_norms) >= 2
            else self.K_inv_op * self.d * ((self.S + 1.0) ** 2 + self.sigma_eps ** 2)
        )
        if M_hat is not None:
            eigs = np.sort(np.linalg.eigvalsh(M_hat))[::-1]
            gap = max(float(eigs[self.r - 1] - eigs[self.r]), 1e-4) if len(eigs) > self.r else 1e-4
        else:
            gap = 1e-4
        return (
            8.0 * self.S * R_X_hat / gap
            * np.sqrt(np.log(2.0 * self.d / self.delta) / m_probe)
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> "RunMetrics":
        env = self.env
        d, r, T = self.d, self.r, env.T
        self.reset_times = []           # rounds at which the trigger fired
        metrics = RunMetrics(name="SPSC-Adaptive", T=T)

        # Subspace state
        M_sum       = np.zeros((d, d))
        U_hat       = np.eye(d, r)
        P_hat       = U_hat @ U_hat.T
        m_probe_tot = 0
        G_norms: list = []
        expl_buf: List[tuple] = []

        # Reward-based detector state
        reward_history: List[float] = []  # exploitation rewards since last reset
        n_exploit_since_reset = 0

        # Phase control
        RECOVERY = 0
        NORMAL   = 1
        phase    = RECOVERY
        rec_count = 0
        probe_counter = 0
        M_rec = np.zeros((d, d))

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            # ==============================================================
            # RECOVERY MODE — pure probing to build fresh subspace estimate
            # ==============================================================
            if phase == RECOVERY:
                metrics.probe_flags[t] = True

                z_probe = self.rng.standard_normal(d)
                u_t     = np.sqrt(d) * z_probe / (np.linalg.norm(z_probe) + 1e-12)
                y_t     = env.step(u_t, t)

                s_t = y_t ** 2 - self.sigma_eps ** 2
                G_t = K_inverse(s_t * np.outer(u_t, u_t), d)

                G_norms.append(float(np.linalg.norm(G_t, ord=2)))
                M_rec       += G_t
                m_probe_tot += 1
                rec_count   += 1

                if rec_count >= self.m_relearn and rec_count >= 2:
                    M_hat_now = M_rec / rec_count
                    eig_vals, eig_vecs = np.linalg.eigh(M_hat_now)
                    U_hat = eig_vecs[:, -r:]
                    P_hat = U_hat @ U_hat.T

                    M_sum = M_rec.copy()
                    phase = NORMAL
                    rec_count = 0
                    probe_counter = 0
                    expl_buf = []
                    reward_history = []
                    n_exploit_since_reset = 0

                # Subspace error diagnostic
                k = env.seg_of[t]
                P_true = env.segment_projector(k)
                metrics.subspace_error[t] = np.linalg.norm(P_hat - P_true, ord=2)

                r_t = float(u_t @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t]  = r_opt - r_t + self.c

            # ==============================================================
            # NORMAL MODE — interleaved probe + exploit
            # ==============================================================
            else:
                is_probe = (probe_counter % self.probe_every == 0)
                probe_counter += 1

                if is_probe:
                    # --- PROBE ROUND ---
                    metrics.probe_flags[t] = True

                    z_probe = self.rng.standard_normal(d)
                    u_t     = np.sqrt(d) * z_probe / (np.linalg.norm(z_probe) + 1e-12)
                    y_t     = env.step(u_t, t)

                    s_t = y_t ** 2 - self.sigma_eps ** 2
                    G_t = K_inverse(s_t * np.outer(u_t, u_t), d)

                    G_norms.append(float(np.linalg.norm(G_t, ord=2)))
                    M_sum       += G_t
                    m_probe_tot += 1

                    M_hat_now = M_sum / m_probe_tot
                    eig_vals, eig_vecs = np.linalg.eigh(M_hat_now)
                    U_hat = eig_vecs[:, -r:]
                    P_hat = U_hat @ U_hat.T

                    k = env.seg_of[t]
                    P_true = env.segment_projector(k)
                    metrics.subspace_error[t] = np.linalg.norm(P_hat - P_true, ord=2)

                    r_t = float(u_t @ env.theta[t])
                    metrics.control_regret[t] = r_opt - r_t
                    metrics.costed_regret[t]  = r_opt - r_t + self.c

                else:
                    # --- EXPLOITATION ROUND ---
                    window_entries = [(x_s, y_s) for (x_s, y_s, s) in expl_buf
                                     if s >= t - self.W]

                    V_tilde = self.lam * np.eye(r)
                    b_tilde = np.zeros(r)
                    for x_s, y_s in window_entries:
                        z_s = U_hat.T @ x_s
                        V_tilde += np.outer(z_s, z_s)
                        b_tilde += z_s * y_s

                    a_hat   = np.linalg.solve(V_tilde, b_tilde)
                    beta_t  = self._beta(len(window_entries))
                    M_hat_curr = M_sum / max(m_probe_tot, 1) if m_probe_tot >= 2 else None

                    Z       = action_set @ U_hat
                    V_inv_Z = np.linalg.solve(V_tilde, Z.T).T
                    ellip   = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))
                    ucb     = Z @ a_hat + beta_t * ellip

                    best_idx = int(np.argmax(ucb))
                    x_dep    = action_set[best_idx]
                    y_t      = env.step(x_dep, t)

                    expl_buf.append((x_dep, y_t, t))
                    if len(expl_buf) > self.W + 10:
                        expl_buf = [(x_s, y_s, s) for (x_s, y_s, s) in expl_buf
                                    if s >= t - self.W]

                    # --- REWARD-BASED CHANGE DETECTION ---
                    reward_history.append(y_t)
                    n_exploit_since_reset += 1

                    dw = self.det_window
                    if n_exploit_since_reset >= self.warmup + dw:
                        # Compare recent reward to historical baseline
                        recent = np.array(reward_history[-dw:])
                        historical = np.array(reward_history[:-dw])

                        mu_hist = historical.mean()
                        sigma_hist = max(historical.std(), 1e-6)
                        mu_recent = recent.mean()

                        # Standardized drop: how many sigma below historical mean?
                        z_score = (mu_hist - mu_recent) / (sigma_hist / np.sqrt(dw))

                        if z_score > self.cusum_threshold:
                            # DETECTED CHANGE — enter recovery
                            self.reset_times.append(t)
                            phase       = RECOVERY
                            rec_count   = 0
                            M_rec       = np.zeros((d, d))
                            M_sum       = np.zeros((d, d))
                            m_probe_tot = 0
                            G_norms     = []
                            expl_buf    = []
                            reward_history = []
                            n_exploit_since_reset = 0

                    r_t = float(x_dep @ env.theta[t])
                    metrics.control_regret[t] = r_opt - r_t
                    metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# SOTA Baseline: LowOFUL (Jun et al. 2019 spirit)
# ---------------------------------------------------------------------------

class LowOFUL:
    """
    Low-rank OFUL: learns a rank-r subspace from the action-reward outer
    products, then runs UCB in the learned r-dimensional subspace.

    PCA-based subspace recovery (in the spirit of Jun et al. 2019, Lu et al. 2021):
    maintain a running estimate of the reward parameter via nuclear-norm
    regularized ridge regression, extract the top-r subspace, and exploit
    in that subspace.

    Assumes stationary subspace. Under nonstationarity the subspace
    estimate degrades because past data from old segments pollutes the
    estimate, the critical weakness SPSC addresses.

    Resets at oracle segment boundaries are NOT given (this is the point:
    show that stationary low-rank methods fail under nonstationarity).
    """

    def __init__(
        self,
        env: "LowRankLDSEnvironment",
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 0,
        pca_warmup: int = 30,
        subspace_update_freq: int = 20,
    ):
        self.env    = env
        self.lam    = lam
        self.delta  = delta
        self.rng    = np.random.default_rng(seed)
        self.pca_warmup = pca_warmup
        self.subspace_update_freq = subspace_update_freq

        self.d = env.d
        self.r = env.r
        self.L_x       = env.L_x
        self.sigma_eps = env.sigma_eps
        self.S         = env.S

    def _beta_r(self, n_exploit: int) -> float:
        arg = max(1.0 + n_exploit * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def run(self) -> "RunMetrics":
        env = self.env
        d, r, T = self.d, self.r, env.T
        metrics = RunMetrics(name="LowOFUL", T=T)

        # Accumulate action-reward outer products for subspace estimation
        # M = (1/n) sum_t (y_t * x_t)(y_t * x_t)^T  (rank-r in expectation)
        M_sum     = np.zeros((d, d))
        n_total   = 0
        U_hat     = np.eye(d, r)

        # Ridge regression in learned subspace
        V_r       = self.lam * np.eye(r)
        b_r       = np.zeros(r)

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            # Project actions into current subspace
            Z = action_set @ U_hat

            # UCB action selection
            a_hat  = np.linalg.solve(V_r, b_r)
            beta_t = self._beta_r(n_total)

            V_inv_Z = np.linalg.solve(V_r, Z.T).T
            ellip   = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))
            ucb     = Z @ a_hat + beta_t * ellip

            best_idx = int(np.argmax(ucb))
            x_dep    = action_set[best_idx]
            y_t      = env.step(x_dep, t)

            # Update subspace estimate: accumulate (y*x)(y*x)^T
            yx = y_t * x_dep
            M_sum   += np.outer(yx, yx)
            n_total += 1

            # Periodically update subspace from accumulated data
            if n_total >= self.pca_warmup and n_total % self.subspace_update_freq == 0:
                M_hat = M_sum / n_total
                eig_vals, eig_vecs = np.linalg.eigh(M_hat)
                U_hat_new = eig_vecs[:, -r:]

                # Check if subspace changed significantly — if so, reset regression
                P_old = U_hat @ U_hat.T
                P_new = U_hat_new @ U_hat_new.T
                if np.linalg.norm(P_new - P_old, 'fro') > 0.3:
                    V_r = self.lam * np.eye(r)
                    b_r = np.zeros(r)

                U_hat = U_hat_new

            # Update ridge regression in current subspace
            z_dep = U_hat.T @ x_dep
            V_r  += np.outer(z_dep, z_dep)
            b_r  += z_dep * y_t

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# SOTA Baseline: VOFUL (variance-weighted PCA)
# ---------------------------------------------------------------------------

class VOFUL:
    """
    Variance-aware Optimistic algorithm for Low-rank linear bandits.

    Variance-weighted PCA for low-rank subspace recovery: weight the
    action-reward covariance by the per-action expected variance to
    give tighter subspace recovery than unweighted PCA, then exploit
    via r-dimensional ridge UCB.

    The variance weighting helps when different actions have different
    signal-to-noise ratios.  Formally:
        M_hat = (1/n) sum_t w_t * (y_t * x_t)(y_t * x_t)^T
    where w_t = 1 / (||x_t||^2 + sigma^2) normalizes by expected variance.

    Like LowOFUL, assumes STATIONARY subspace.  No oracle segment boundaries.
    """

    def __init__(
        self,
        env: "LowRankLDSEnvironment",
        lam: float = 1.0,
        delta: float = 0.05,
        seed: int = 0,
        pca_warmup: int = 30,
        subspace_update_freq: int = 20,
    ):
        self.env    = env
        self.lam    = lam
        self.delta  = delta
        self.rng    = np.random.default_rng(seed)
        self.pca_warmup = pca_warmup
        self.subspace_update_freq = subspace_update_freq

        self.d = env.d
        self.r = env.r
        self.L_x       = env.L_x
        self.sigma_eps = env.sigma_eps
        self.S         = env.S

    def _beta_r(self, n_exploit: int) -> float:
        arg = max(1.0 + n_exploit * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return (
            self.sigma_eps * np.sqrt(self.r * np.log(arg / self.delta))
            + np.sqrt(self.lam) * self.S
        )

    def run(self) -> "RunMetrics":
        env = self.env
        d, r, T = self.d, self.r, env.T
        metrics = RunMetrics(name="VOFUL", T=T)

        # Variance-weighted outer product accumulator
        M_sum     = np.zeros((d, d))
        w_sum     = 0.0
        n_total   = 0
        U_hat     = np.eye(d, r)

        # Ridge regression in learned subspace
        V_r       = self.lam * np.eye(r)
        b_r       = np.zeros(r)

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt      = env.optimal_reward(action_set, t)

            Z = action_set @ U_hat

            a_hat  = np.linalg.solve(V_r, b_r)
            beta_t = self._beta_r(n_total)

            V_inv_Z = np.linalg.solve(V_r, Z.T).T
            ellip   = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))
            ucb     = Z @ a_hat + beta_t * ellip

            best_idx = int(np.argmax(ucb))
            x_dep    = action_set[best_idx]
            y_t      = env.step(x_dep, t)

            # Variance-weighted subspace update
            x_norm_sq = float(np.dot(x_dep, x_dep))
            w_t = 1.0 / (x_norm_sq + self.sigma_eps ** 2)
            yx = y_t * x_dep
            M_sum   += w_t * np.outer(yx, yx)
            w_sum   += w_t
            n_total += 1

            # Periodically update subspace
            if n_total >= self.pca_warmup and n_total % self.subspace_update_freq == 0:
                M_hat = M_sum / w_sum
                eig_vals, eig_vecs = np.linalg.eigh(M_hat)
                U_hat_new = eig_vecs[:, -r:]

                P_old = U_hat @ U_hat.T
                P_new = U_hat_new @ U_hat_new.T
                if np.linalg.norm(P_new - P_old, 'fro') > 0.3:
                    V_r = self.lam * np.eye(r)
                    b_r = np.zeros(r)

                U_hat = U_hat_new

            z_dep = U_hat.T @ x_dep
            V_r  += np.outer(z_dep, z_dep)
            b_r  += z_dep * y_t

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t]  = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: BOSS-adapted (Duong et al. NeurIPS 2024)
# Learn ONE shared subspace from initial exploration, then exploit forever.
# Original BOSS assumes fixed subspace + known task boundaries + ellipsoid
# actions. This adaptation runs in our continuous-action setting.
# ---------------------------------------------------------------------------

class BOSSAdapted:
    """
    BOSS-adapted: explore for m_explore steps to learn a fixed subspace,
    then run projected LinUCB forever. Never updates subspace.
    Fails when subspace changes (designed for stationary subspace).
    """

    def __init__(self, env, m_explore=200, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.m_explore = m_explore
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.sigma_eps = env.sigma_eps
        self.S = env.S
        self.L_x = env.L_x

    def run(self):
        env = self.env
        d, r, T = env.d, env.r, env.T
        metrics = RunMetrics(name="BOSS-adapted", T=T)

        M_sum = np.zeros((d, d))
        U_hat = np.eye(d, r)
        V_r = self.lam * np.eye(r)
        b_r = np.zeros(r)

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)

            if t < self.m_explore:
                z = self.rng.standard_normal(d)
                u_t = np.sqrt(d) * z / (np.linalg.norm(z) + 1e-12)
                y_t = env.step(u_t, t)

                s_t = y_t ** 2 - self.sigma_eps ** 2
                G_t = K_inverse(s_t * np.outer(u_t, u_t), d)
                M_sum += G_t

                if t >= 2:
                    M_hat = M_sum / (t + 1)
                    eig_vals, eig_vecs = np.linalg.eigh(M_hat)
                    U_hat = eig_vecs[:, -r:]

                r_t = float(u_t @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t
            else:
                Z = action_set @ U_hat
                a_hat = np.linalg.solve(V_r, b_r)
                beta = self.sigma_eps * np.sqrt(
                    r * np.log(max(1.0 + t * self.L_x**2 / self.lam, 2.0) / self.delta)
                ) + np.sqrt(self.lam) * self.S
                V_inv_Z = np.linalg.solve(V_r, Z.T).T
                ellip = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))
                ucb = Z @ a_hat + beta * ellip

                x_dep = action_set[int(np.argmax(ucb))]
                y_t = env.step(x_dep, t)

                z_dep = U_hat.T @ x_dep
                V_r += np.outer(z_dep, z_dep)
                b_r += z_dep * y_t

                r_t = float(x_dep @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Jedra-adapted (Jedra et al. ICML 2024)
# Spectral subspace recovery once, then exploit with projected LinUCB.
# Original operates on discrete matrix bandits M_{i,j}. This adaptation
# uses their spectral recovery idea in our continuous vector setting.
# ---------------------------------------------------------------------------

class JedraAdapted:
    """
    Jedra-adapted: two-phase spectral method.
    Phase 1: uniform random exploration, build second-moment matrix, SVD.
    Phase 2: projected LinUCB with FIXED subspace (never re-estimated).
    Fails when subspace changes (designed for stationary low-rank matrix).
    """

    def __init__(self, env, m_explore=200, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.m_explore = m_explore
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.sigma_eps = env.sigma_eps
        self.S = env.S
        self.L_x = env.L_x

    def run(self):
        env = self.env
        d, r, T = env.d, env.r, env.T
        metrics = RunMetrics(name="Jedra-adapted", T=T)

        X_buf = []
        Y_buf = []
        U_hat = np.eye(d, r)
        V_r = self.lam * np.eye(r)
        b_r = np.zeros(r)

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)

            if t < self.m_explore:
                idx = self.rng.integers(len(action_set))
                x_dep = action_set[idx]
                y_t = env.step(x_dep, t)
                X_buf.append(x_dep)
                Y_buf.append(y_t)

                if t == self.m_explore - 1:
                    X = np.array(X_buf)
                    Y = np.array(Y_buf)
                    M = (X.T * Y) @ X / len(Y)
                    eig_vals, eig_vecs = np.linalg.eigh(M)
                    U_hat = eig_vecs[:, -r:]

                r_t = float(x_dep @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t
            else:
                Z = action_set @ U_hat
                a_hat = np.linalg.solve(V_r, b_r)
                beta = self.sigma_eps * np.sqrt(
                    r * np.log(max(1.0 + t * self.L_x**2 / self.lam, 2.0) / self.delta)
                ) + np.sqrt(self.lam) * self.S
                V_inv_Z = np.linalg.solve(V_r, Z.T).T
                ellip = np.sqrt(np.einsum('ij,ij->i', Z, V_inv_Z))
                ucb = Z @ a_hat + beta * ellip

                x_dep = action_set[int(np.argmax(ucb))]
                y_t = env.step(x_dep, t)

                z_dep = U_hat.T @ x_dep
                V_r += np.outer(z_dep, z_dep)
                b_r += z_dep * y_t

                r_t = float(x_dep @ env.theta[t])
                metrics.control_regret[t] = r_opt - r_t
                metrics.costed_regret[t] = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Linear Thompson Sampling (Agrawal & Goyal 2013)
# ---------------------------------------------------------------------------

class LinTS:
    """
    Linear Thompson Sampling in full ambient space R^d.
    Samples theta from N(theta_hat, v^2 V^{-1}) and plays greedy.
    Supports optional forgetting factor for D-LinTS variant.
    """

    def __init__(self, env, lam=1.0, delta=0.05, seed=1, forgetting_factor=1.0):
        self.env = env
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.ff = forgetting_factor
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x

    def run(self):
        env = self.env
        d, T = env.d, env.T
        metrics = RunMetrics(name="LinTS", T=T)

        V = self.lam * np.eye(d)
        b = np.zeros(d)

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)

            theta_hat = np.linalg.solve(V, b)
            v = self.sigma_eps * np.sqrt(d * np.log(max(t + 1, 2) / self.delta))
            try:
                V_inv = np.linalg.solve(V, np.eye(d))
                theta_tilde = self.rng.multivariate_normal(theta_hat, v**2 * V_inv)
            except np.linalg.LinAlgError:
                theta_tilde = theta_hat

            scores = action_set @ theta_tilde
            x_dep = action_set[int(np.argmax(scores))]
            y = env.step(x_dep, t)

            V = self.ff * V + np.outer(x_dep, x_dep)
            b = self.ff * b + x_dep * y

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t

        return metrics


# ---------------------------------------------------------------------------
# Baseline: Sliding-Window Linear Thompson Sampling
# ---------------------------------------------------------------------------

class SWLinTS:
    """Sliding-window Linear Thompson Sampling."""

    def __init__(self, env, window=200, lam=1.0, delta=0.05, seed=1):
        self.env = env
        self.W = window
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.sigma_eps = env.sigma_eps

    def run(self):
        env = self.env
        d, T = env.d, env.T
        metrics = RunMetrics(name="SW-LinTS", T=T)
        buf = []

        for t in range(T):
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)

            win = [(xs, ys) for xs, ys, s in buf if s >= t - self.W]
            V = self.lam * np.eye(d)
            b = np.zeros(d)
            for xs, ys in win:
                V += np.outer(xs, xs)
                b += xs * ys

            theta_hat = np.linalg.solve(V, b)
            v = self.sigma_eps * np.sqrt(d * np.log(max(t + 1, 2) / self.delta))
            try:
                V_inv = np.linalg.solve(V, np.eye(d))
                theta_tilde = self.rng.multivariate_normal(theta_hat, v**2 * V_inv)
            except np.linalg.LinAlgError:
                theta_tilde = theta_hat

            scores = action_set @ theta_tilde
            x_dep = action_set[int(np.argmax(scores))]
            y = env.step(x_dep, t)

            buf.append((x_dep, y, t))
            if len(buf) > self.W + 10:
                buf = [(xs, ys, s) for xs, ys, s in buf if s >= t - self.W]

            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t

        return metrics
