"""
Pendigits *real-data-calibrated* bandit environment.

Uses UCI Pendigits (10,992 samples, 16 features, 10 digit classes).
All parameters derived from real data:
  - Features: 16 raw pen-stroke coords, truncated/augmented to target d
  - theta_k: per-segment OLS fit of features -> digit labels
  - Segments: data sorted by digit label creates natural non-stationarity
  - Low-rank: 10 digit classes with correlated stroke patterns

Data auto-downloaded via sklearn/OpenML.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


class RealPendigitsEnvironment:
    """
    Real-data-calibrated Pendigits bandit.

    Segments created by sorting data by digit label (natural regime changes).
    theta_k = per-segment OLS of features -> digit labels.
    Reward follows paper's linear model: y = x^T theta_t + eps.
    """

    def __init__(
        self,
        d: int = 16,
        r: int = 1,
        n_actions: int = 40,
        segment_size: int = 500,
        n_segments: int = 10,
        seed: int = 42,
    ):
        self.r = r
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self._load_and_prepare_data(d)
        self._build_segments(segment_size, n_segments)
        self._build_theta_and_subspaces()

        self.L_x = 1.0
        self.L = 1.0
        self.L_eps = 5.0
        self.spectral_radius = 0.0
        self.sigma_eta = 0.0

    def _load_and_prepare_data(self, target_d):
        """Load Pendigits, build features to match target_d."""
        data = fetch_openml('pendigits', version=1, as_frame=False, parser='auto')
        X_raw = data.data.astype(float)
        y_raw = data.target.astype(int)

        # Standardize
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw)
        d_raw = X_std.shape[1]  # 16

        if target_d <= d_raw:
            X_full = X_std[:, :target_d]
        else:
            # Add pairwise interactions
            interactions = []
            for i in range(d_raw):
                for j in range(i + 1, d_raw):
                    interactions.append(X_std[:, i] * X_std[:, j])
            X_inter = np.column_stack(interactions)  # 120 interactions
            X_full = np.hstack([X_std, X_inter])[:, :target_d]

        # Normalize to unit norm
        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        X_full /= np.maximum(norms, 1e-8)

        # Sort by digit label to create non-stationarity
        sort_idx = np.argsort(y_raw)
        self._features = X_full[sort_idx]
        self._labels = (y_raw[sort_idx].astype(float) - 4.5)  # center around 0
        self.d = X_full.shape[1]
        self._n_total = len(self._features)

    def _build_segments(self, segment_size, n_segments):
        self.K = n_segments
        total_needed = segment_size * n_segments
        if total_needed > self._n_total:
            segment_size = self._n_total // n_segments
            total_needed = segment_size * n_segments

        step = self._n_total // total_needed
        indices = np.arange(0, self._n_total, step)[:total_needed]
        self._seg_features = self._features[indices]
        self._seg_labels = self._labels[indices]

        self.T = total_needed
        self.segment_lengths = [segment_size] * self.K
        self.segment_lengths[-1] += self.T - sum(self.segment_lengths)

        self.tau = [0]
        for l in self.segment_lengths[:-1]:
            self.tau.append(self.tau[-1] + l)

        self.seg_of = np.zeros(self.T, dtype=int)
        for k, start in enumerate(self.tau):
            self.seg_of[start:start + self.segment_lengths[k]] = k

    def _build_theta_and_subspaces(self):
        d = self.d
        self.theta = np.zeros((self.T, d))
        self.B_list = []
        residuals = []

        for k in range(self.K):
            s = self.tau[k]
            e = s + self.segment_lengths[k]

            X_k = self._seg_features[s:e]
            y_k = self._seg_labels[s:e]

            lam = 1.0
            theta_k = np.linalg.solve(X_k.T @ X_k + lam * np.eye(d), X_k.T @ y_k)
            resid = y_k - X_k @ theta_k
            residuals.extend(resid.tolist())
            self.theta[s:e] = theta_k[np.newaxis, :]

            # Subspace: top-r directions
            n_sub = min(len(y_k), 500)
            W = np.diag(np.abs(y_k[:n_sub]) + 0.1)
            X_sub = X_k[:n_sub]
            _, _, Vt = np.linalg.svd(W @ X_sub, full_matrices=False)
            B_k = Vt[:self.r].T
            Q, _ = np.linalg.qr(
                np.column_stack([theta_k.reshape(-1, 1), B_k])
            )
            B_k = Q[:, :self.r]
            self.B_list.append(B_k)

        self.sigma_eps = float(np.std(residuals))
        self.S = float(np.max(np.linalg.norm(self.theta, axis=1)))

    # ------------------------------------------------------------------
    # Per-round interface
    # ------------------------------------------------------------------

    def get_action_set(self, t: int, rng=None):
        _rng = rng if rng is not None else self.rng
        k = self.seg_of[t]
        s = self.tau[k]
        e = s + self.segment_lengths[k]
        n = min(self.n_actions, e - s)
        chosen = _rng.choice(e - s, size=n, replace=False) + s
        return self._seg_features[chosen]

    def step(self, action: np.ndarray, t: int) -> float:
        """Linear model: y = x^T theta_t + eps. theta from real OLS."""
        eps = self.rng.normal(0.0, self.sigma_eps)
        eps = np.clip(eps, -self.L_eps, self.L_eps)
        return float(action @ self.theta[t]) + eps

    def optimal_reward(self, action_set: np.ndarray, t: int) -> float:
        return float(np.max(action_set @ self.theta[t]))

    def segment_projector(self, k: int) -> np.ndarray:
        B = self.B_list[k]
        return B @ B.T

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def svd_spectrum(self):
        thetas = np.array([self.theta[self.tau[k]] for k in range(self.K)])
        _, svals, _ = np.linalg.svd(thetas, full_matrices=False)
        return svals / svals.sum()
