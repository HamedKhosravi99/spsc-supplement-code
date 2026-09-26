"""
Synthetic environment for the Low-Rank LDS Bandit paper.

Environment design rationale:
  - d=20 ambient, r=3 latent → 7x dimension reduction; probing should recover this
  - K=4 segments with distinct subspaces → algorithm must re-identify after each change
  - LDS within each segment with spectral radius ~0.7 → non-trivial temporal dynamics
  - Time-varying predictable second moment (not i.i.d.) → LDS structure is load-bearing
  - Action set: 80 fresh unit vectors per round → exploitable low-rank gap
"""

import numpy as np


class LowRankLDSEnvironment:
    """
    Piecewise-stationary bandit with low-rank LDS latent dynamics.

    theta_t = B_k* @ w_t,  t in segment k
    w_t = A_k @ w_{t-1} + eta_{t-1},  eta ~ N(0, Sigma_eta_k)
    y_t = x_t^T theta_t + eps_t,  eps_t ~ N(0, sigma_eps^2)  [bounded by L_eps]

    Parameters
    ----------
    d           : ambient dimension
    r           : latent rank
    K           : number of segments
    T           : horizon
    sigma_eps   : noise std (known to learner)
    spectral_radius : LDS spectral radius (< 1, controls within-segment dynamics)
    n_actions   : size of action set drawn per round (unit sphere)
    seed        : RNG seed
    """

    def __init__(
        self,
        d: int = 4,
        r: int = 1,
        K: int = 4,
        T: int = 6000,
        sigma_eps: float = 0.3,
        spectral_radius: float = 0.99,
        n_actions: int = 80,
        seed: int = 42,
        sigma_eta: float = 0.04,
        piecewise_constant: bool = False,
        feature_decay: float = 0.0,
    ):
        self.d = d
        self.r = r
        self.K = K
        self.T = T
        self.sigma_eps = sigma_eps
        self.spectral_radius = spectral_radius
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.sigma_eta = sigma_eta
        self.piecewise_constant = piecewise_constant
        self.feature_decay = feature_decay

        # Derived constants (accessible to algorithm for concentration bounds)
        self.L_x = 1.0       # action/probe norm bound (unit sphere)
        self.L = 1.0         # probe norm bound
        self.L_eps = 1.0     # noise bound (we clip to enforce a.s. bound)
        self.S = None        # set after trajectory generation

        # Build environment
        self._build_segments()
        self._build_subspaces()
        self._build_lds_params()
        self._generate_trajectory()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_segments(self):
        """Divide T rounds into K roughly equal segments."""
        base = self.T // self.K
        lengths = [base] * self.K
        lengths[-1] += self.T - sum(lengths)
        self.segment_lengths = lengths
        # change-point times: tau_k = start of segment k (0-indexed rounds)
        self.tau = [0]
        for l in lengths[:-1]:
            self.tau.append(self.tau[-1] + l)
        # segment index for each round t
        self.seg_of = np.zeros(self.T, dtype=int)
        for k, start in enumerate(self.tau):
            end = start + self.segment_lengths[k]
            self.seg_of[start:end] = k

    def _build_subspaces(self):
        """Generate K orthonormal d×r representation matrices with well-separated subspaces."""
        self.B_list = []
        for k in range(self.K):
            # Fresh random orthonormal frame; re-randomize until subspace gap is large enough
            while True:
                Q, _ = np.linalg.qr(self.rng.standard_normal((self.d, self.r)))
                B = Q[:, : self.r]
                if k == 0:
                    self.B_list.append(B)
                    break
                # Ensure subspace change: sin-theta distance to previous subspace > 0.5
                P_prev = self.B_list[k - 1] @ self.B_list[k - 1].T
                P_new = B @ B.T
                gap = np.linalg.norm(P_new - P_prev, ord=2)
                if gap > 0.7:  # ensures the algorithm must re-learn
                    self.B_list.append(B)
                    break

    def _build_lds_params(self):
        """Build stable LDS matrices A_k and innovation covariances Sigma_eta_k.

        A_k = spectral_radius * Q_k  where Q_k is random r×r orthogonal matrix.
        All eigenvalues of A_k have magnitude exactly spectral_radius.
        Sigma_eta_k = sigma_eta * I  (isotropic innovations).
        Together these give stationary covariance Sigma_w = sigma_eta/(1-rho^2) * I,
        so all r LDS components have equal variance, which is needed for
        full-rank subspace recovery from probe observations.
        """
        self.A_list = []
        self.Sigma_eta_list = []
        for k in range(self.K):
            # A_k = rho * I_r: scalar AR(1) per latent component.
            # Avoids oscillatory dynamics from random-orthogonal A_k, making
            # theta_t genuinely slowly-varying within each segment so that
            # windowed regression is effective.
            self.A_list.append(self.spectral_radius * np.eye(self.r))
            # Isotropic innovations: sigma_eta * I_r
            self.Sigma_eta_list.append(self.sigma_eta * np.eye(self.r))

    def _generate_trajectory(self):
        """Pre-generate the full trajectory of latent states and theta_t."""
        self.theta = np.zeros((self.T, self.d))
        self.w = np.zeros((self.T, self.r))

        t = 0
        for k in range(self.K):
            Bk = self.B_list[k]
            Ak = self.A_list[k]
            Sigma_k = self.Sigma_eta_list[k]
            n_k = self.segment_lengths[k]

            # Initialize latent state
            wt = self.rng.multivariate_normal(np.zeros(self.r), Sigma_k)

            if self.piecewise_constant:
                for step in range(n_k):
                    self.w[t] = wt
                    self.theta[t] = Bk @ wt
                    t += 1
            else:
                for step in range(n_k):
                    self.w[t] = wt
                    self.theta[t] = Bk @ wt
                    t += 1
                    if step < n_k - 1:
                        eta = self.rng.multivariate_normal(np.zeros(self.r), Sigma_k)
                        wt = Ak @ wt + eta

        self.S = float(np.max(np.linalg.norm(self.theta, axis=1)))

    # ------------------------------------------------------------------
    # Per-round interface
    # ------------------------------------------------------------------

    def get_action_set(self, t: int, rng=None):
        """
        Return an n_actions × d matrix of unit-sphere actions for round t.
        Fresh randomness so actions don't repeat.
        If feature_decay > 0, coordinates are scaled by (i+1)^{-feature_decay}
        before re-normalizing, concentrating energy in low-index dimensions.
        """
        _rng = rng if rng is not None else self.rng
        raw = _rng.standard_normal((self.n_actions, self.d))
        if self.feature_decay > 0:
            scale = np.arange(1, self.d + 1, dtype=float) ** (-self.feature_decay)
            raw *= scale[np.newaxis, :]
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    def step(self, action: np.ndarray, t: int) -> float:
        """
        Deploy `action` at round t; return observed reward y_t.
        Noise is drawn fresh each call (use for the actual algorithm run).
        """
        eps = self.rng.normal(0.0, self.sigma_eps)
        eps = np.clip(eps, -self.L_eps, self.L_eps)
        return float(action @ self.theta[t]) + eps

    def optimal_reward(self, action_set: np.ndarray, t: int) -> float:
        """max_{x in action_set} x^T theta_t (noiseless)."""
        return float(np.max(action_set @ self.theta[t]))

    def segment_projector(self, k: int) -> np.ndarray:
        """True segment projector P_k* = B_k* B_k*^T."""
        B = self.B_list[k]
        return B @ B.T

    # ------------------------------------------------------------------
    # Predictable second moment (oracle, for diagnostic)
    # ------------------------------------------------------------------

    def predictable_second_moment(self, t: int) -> np.ndarray:
        """
        Oracle E[theta_t theta_t^T | H_{t-1}].
        Under the LDS: = B_k* E[w_t w_t^T | H_{t-1}] B_k*^T.
        We approximate E[w_t w_t^T | H_{t-1}] = w_t w_t^T (deterministic at t given past).
        """
        k = self.seg_of[t]
        Bk = self.B_list[k]
        wt = self.w[t]
        return Bk @ np.outer(wt, wt) @ Bk.T
