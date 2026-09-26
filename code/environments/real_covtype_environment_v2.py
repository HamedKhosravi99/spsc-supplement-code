"""
Forest Covertype *real-data-calibrated* bandit environment (variable d).

UCI Forest Covertype (581K samples, 54 features, 7 cover types).
Supports target d by truncating or augmenting with pairwise interactions.
"""

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler


class RealCovtypeEnvironmentV2:
    """
    Real-data-calibrated Covertype bandit with configurable d.

    d <= 54: truncate standardized features.
    d > 54: augment with pairwise interactions of first 10 quantitative features.
    """

    def __init__(
        self,
        d: int = 55,
        r: int = 3,
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
        X_raw, y_raw = fetch_covtype(return_X_y=True)

        # Standardize all 54 features
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw[:, :54].astype(float))
        d_raw = X_std.shape[1]  # 54

        if target_d <= d_raw:
            X_full = X_std[:, :target_d]
        else:
            # Add pairwise interactions of first 10 quantitative features
            X_quant = X_std[:, :10]
            interactions = []
            for i in range(10):
                for j in range(i + 1, 10):
                    interactions.append(X_quant[:, i] * X_quant[:, j])
            X_inter = np.column_stack(interactions)  # 45 interactions
            X_pool = np.hstack([X_std, X_inter])  # 54 + 45 = 99

            if target_d <= X_pool.shape[1]:
                X_full = X_pool[:, :target_d]
            else:
                # Add more: pairwise of all 54 (capped)
                more = []
                for i in range(min(d_raw, 20)):
                    for j in range(i + 1, min(d_raw, 20)):
                        more.append(X_std[:, i] * X_std[:, j])
                X_more = np.column_stack(more)
                X_pool2 = np.hstack([X_pool, X_more])
                X_full = X_pool2[:, :target_d]

        # Normalize to unit norm
        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        X_full /= np.maximum(norms, 1e-8)

        # Sort by elevation (column 0 of raw data) for non-stationarity
        sort_idx = np.argsort(X_raw[:, 0])
        self._features = X_full[sort_idx]
        self._labels = (y_raw[sort_idx].astype(float) - 4.0)
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

    def get_action_set(self, t, rng=None):
        _rng = rng if rng is not None else self.rng
        k = self.seg_of[t]
        s = self.tau[k]
        e = s + self.segment_lengths[k]
        n = min(self.n_actions, e - s)
        chosen = _rng.choice(e - s, size=n, replace=False) + s
        return self._seg_features[chosen]

    def step(self, action, t):
        eps = self.rng.normal(0.0, self.sigma_eps)
        eps = np.clip(eps, -self.L_eps, self.L_eps)
        return float(action @ self.theta[t]) + eps

    def optimal_reward(self, action_set, t):
        return float(np.max(action_set @ self.theta[t]))

    def segment_projector(self, k):
        B = self.B_list[k]
        return B @ B.T

    def svd_spectrum(self):
        thetas = np.array([self.theta[self.tau[k]] for k in range(self.K)])
        _, svals, _ = np.linalg.svd(thetas, full_matrices=False)
        return svals / svals.sum()
