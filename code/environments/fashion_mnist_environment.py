"""
Fashion-MNIST *real-data-calibrated* bandit environment.

Uses Fashion-MNIST (70,000 samples, 784 pixel features, 10 clothing categories).
All parameters derived from real data:
  - Features: 784 raw pixel values (28x28), reduced via random projection for d < 784
  - theta_k: per-segment OLS fit of features -> class labels
  - Segments: data sorted by class label creates natural non-stationarity
  - Low-rank: 10 clothing categories with correlated visual patterns

Data auto-downloaded via sklearn/OpenML.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


class FashionMNISTEnvironment:
    """
    Real-data-calibrated Fashion-MNIST bandit.

    Segments created by sorting data by class label (natural regime changes).
    theta_k = per-segment OLS of features -> class labels.
    Reward follows paper's linear model: y = x^T theta_t + eps.
    """

    def __init__(
        self,
        d: int = 100,
        r: int = 10,
        n_actions: int = 40,
        segment_size: int = 500,
        n_segments: int = 10,
        seed: int = 42,
    ):
        self.r = r
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self._load_and_prepare_data(d, seed)
        self._build_segments(segment_size, n_segments)
        self._build_theta_and_subspaces()

        self.L_x = 1.0
        self.L = 1.0
        self.L_eps = 5.0
        self.spectral_radius = 0.0
        self.sigma_eta = 0.0

    def _load_and_prepare_data(self, target_d, seed):
        """Load Fashion-MNIST, build features to match target_d."""
        data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
        X_raw = data.data.astype(float)
        y_raw = data.target.astype(int)

        d_raw = X_raw.shape[1]  # 784

        # Remove constant columns (some border pixels are always 0)
        col_std = X_raw.std(axis=0)
        active_cols = col_std > 1e-6
        X_active = X_raw[:, active_cols]

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_active)
        d_active = X_std.shape[1]

        if target_d <= d_active:
            # Random projection to target_d (preserves structure, doesn't optimize)
            proj_rng = np.random.default_rng(0)  # fixed seed for reproducibility
            P = proj_rng.standard_normal((d_active, target_d))
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            X_full = X_std @ P
        else:
            # Use all active features + pad with interactions if needed
            if target_d <= d_active:
                X_full = X_std[:, :target_d]
            else:
                # Random projection to expand (rarely needed for d > 700+)
                proj_rng = np.random.default_rng(0)
                P = proj_rng.standard_normal((d_active, target_d))
                P /= np.linalg.norm(P, axis=0, keepdims=True)
                X_full = X_std @ P

        # Normalize to unit norm
        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        X_full /= np.maximum(norms, 1e-8)

        # Sort by class label to create non-stationarity
        sort_idx = np.argsort(y_raw)
        self._features = X_full[sort_idx]
        self._labels = (y_raw[sort_idx].astype(float) - y_raw.mean())
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

    def get_action_set(self, t: int, rng=None):
        _rng = rng if rng is not None else self.rng
        k = self.seg_of[t]
        s = self.tau[k]
        e = s + self.segment_lengths[k]
        n = min(self.n_actions, e - s)
        chosen = _rng.choice(e - s, size=n, replace=False) + s
        return self._seg_features[chosen]

    def step(self, action: np.ndarray, t: int) -> float:
        eps = self.rng.normal(0.0, self.sigma_eps)
        eps = np.clip(eps, -self.L_eps, self.L_eps)
        return float(action @ self.theta[t]) + eps

    def optimal_reward(self, action_set: np.ndarray, t: int) -> float:
        return float(np.max(action_set @ self.theta[t]))

    def segment_projector(self, k: int) -> np.ndarray:
        B = self.B_list[k]
        return B @ B.T

    def svd_spectrum(self):
        thetas = np.array([self.theta[self.tau[k]] for k in range(self.K)])
        _, svals, _ = np.linalg.svd(thetas, full_matrices=False)
        return svals / svals.sum()
