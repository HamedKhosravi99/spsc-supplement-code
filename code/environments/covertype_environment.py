"""
Covertype semi-synthetic environment for the Low-Rank LDS Bandit.

Uses the UCI Forest Covertype dataset (581K samples, 54 features) to define
real-world action features. Low-rank subspace structure is extracted from
class-conditional statistics. Temporal dynamics follow AR(1) within segments.

This is a standard semi-synthetic benchmark approach: real feature structure
combined with controlled dynamics matching the model assumptions.

Why Covertype
-------------
  - 54 real-valued features (elevation, slope, distances, soil types, etc.)
  - 7 forest cover types create natural clustered structure
  - Large dataset (581K) provides diverse, realistic action features
  - Used in contextual bandit benchmarks: Foster et al. (ICML 2020),
    Bietti et al. (NeurIPS 2021), Agarwal et al. (ICML 2014)
"""

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler


class CovtypeEnvironment:
    """
    Semi-synthetic environment using Covertype features.

    Matches the LowRankLDSEnvironment interface exactly.

    Parameters
    ----------
    d           : ambient dimension (features are PCA-projected to this dim)
    r           : latent rank
    K           : number of segments
    T           : horizon
    sigma_eps   : observation noise std
    spectral_radius : AR(1) parameter
    n_actions   : actions per round
    seed        : RNG seed
    sigma_eta   : innovation noise std
    """

    def __init__(
        self,
        d: int = 54,
        r: int = 3,
        K: int = 8,
        T: int = 10000,
        sigma_eps: float = 0.3,
        spectral_radius: float = 0.99,
        n_actions: int = 80,
        seed: int = 42,
        sigma_eta: float = 0.04,
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

        self.L_x = 1.0
        self.L = 1.0
        self.L_eps = 1.0
        self.S = None

        self._load_and_prepare_data()
        self._build_segments()
        self._build_subspaces()
        self._build_lds_params()
        self._generate_trajectory()

    def _load_and_prepare_data(self):
        """Load Covertype, standardize, and normalize to unit vectors."""
        X_raw, y_raw = fetch_covtype(return_X_y=True)

        # Use first 10 quantitative features (elevation, aspect, slope,
        # distances, hillshade) — these have genuine variance.
        # Columns 0-9 are quantitative; 10-53 are binary soil/wilderness.
        X_quant = X_raw[:, :10].astype(float)

        # Standardize quantitative features
        scaler = StandardScaler()
        X_quant = scaler.fit_transform(X_quant)

        # Build feature matrix based on requested dimensionality
        if self.d <= 10:
            X_full = X_quant
        elif self.d <= 55:
            # Add pairwise interactions of quantitative features (10*9/2 = 45)
            interactions = []
            n_q = X_quant.shape[1]
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    interactions.append(
                        (X_quant[:, i] * X_quant[:, j]).reshape(-1, 1)
                    )
            X_inter = np.hstack(interactions)
            X_full = np.hstack([X_quant, X_inter])  # 10 + 45 = 55
        else:
            interactions = []
            n_q = X_quant.shape[1]
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    interactions.append(
                        (X_quant[:, i] * X_quant[:, j]).reshape(-1, 1)
                    )
            X_inter = np.hstack(interactions)
            X_binary = X_raw[:, 10:].astype(float)
            X_full = np.hstack([X_quant, X_inter, X_binary])  # 10 + 45 + 44 = 99

        # Truncate or pad to exactly d dimensions
        if X_full.shape[1] >= self.d:
            X_full = X_full[:, :self.d]
        else:
            # Pad with random projections of existing features
            n_pad = self.d - X_full.shape[1]
            proj = self.rng.standard_normal((X_full.shape[1], n_pad))
            X_pad = X_full @ proj / np.sqrt(X_full.shape[1])
            X_full = np.hstack([X_full, X_pad])

        # Normalize each sample to unit norm
        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_full = X_full / norms

        # Shuffle and store
        idx = self.rng.permutation(len(X_full))
        self._action_pool = X_full[idx]
        self._labels = y_raw[idx]
        self._class_labels = np.unique(y_raw)

        # Compute class centroids for subspace construction
        self._centroids = {}
        for c in self._class_labels:
            mask = self._labels == c
            self._centroids[c] = X_full[idx][mask].mean(axis=0)

    def _build_segments(self):
        """Divide T rounds into K segments."""
        base = self.T // self.K
        lengths = [base] * self.K
        lengths[-1] += self.T - sum(lengths)
        self.segment_lengths = lengths

        self.tau = [0]
        for l in lengths[:-1]:
            self.tau.append(self.tau[-1] + l)

        self.seg_of = np.zeros(self.T, dtype=int)
        for k, start in enumerate(self.tau):
            end = start + self.segment_lengths[k]
            self.seg_of[start:end] = k

    def _build_subspaces(self):
        """
        Build K distinct rank-r subspaces from the real data structure.

        Uses different subsets of class centroids for each segment,
        then takes SVD to get orthonormal bases. This ensures:
        - Subspaces reflect genuine data structure
        - Different segments have different subspaces
        - Subspace changes are significant (mimicking regime changes)
        """
        self.B_list = []
        centroid_mat = np.array([self._centroids[c] for c in self._class_labels])
        n_classes = len(self._class_labels)

        for k in range(self.K):
            # For each segment, use a different weighted combination of centroids
            # This creates genuinely different subspaces from the real data
            weights = self.rng.dirichlet(np.ones(n_classes))
            # Shift weights to emphasize different classes in different segments
            shift = (k * n_classes // self.K) % n_classes
            weights = np.roll(weights, shift)

            # Create r random directions in the centroid space
            combo_mat = np.zeros((self.r, self.d))
            for j in range(self.r):
                w = self.rng.dirichlet(np.ones(n_classes) * 0.5)
                w = np.roll(w, shift + j)
                combo_mat[j] = w @ centroid_mat
                # Add small perturbation for diversity
                combo_mat[j] += 0.1 * self.rng.standard_normal(self.d)

            # QR decomposition to get orthonormal basis
            Q, R = np.linalg.qr(combo_mat.T)
            B_k = Q[:, :self.r]  # d x r

            # Verify subspace change from previous
            if k > 0:
                P_prev = self.B_list[k - 1] @ self.B_list[k - 1].T
                P_new = B_k @ B_k.T
                gap = np.linalg.norm(P_new - P_prev, ord=2)
                # If subspace too similar, perturb more
                attempts = 0
                while gap < 0.5 and attempts < 10:
                    combo_mat += 0.3 * self.rng.standard_normal(combo_mat.shape)
                    Q, R = np.linalg.qr(combo_mat.T)
                    B_k = Q[:, :self.r]
                    P_new = B_k @ B_k.T
                    gap = np.linalg.norm(P_new - P_prev, ord=2)
                    attempts += 1

            self.B_list.append(B_k)

    def _build_lds_params(self):
        """Build AR(1) dynamics: A_k = rho * I_r, Sigma_eta = sigma_eta * I_r."""
        self.A_list = []
        self.Sigma_eta_list = []
        for k in range(self.K):
            self.A_list.append(self.spectral_radius * np.eye(self.r))
            self.Sigma_eta_list.append(self.sigma_eta * np.eye(self.r))

    def _generate_trajectory(self):
        """Generate theta_t = B_k w_t with AR(1) dynamics within segments."""
        self.theta = np.zeros((self.T, self.d))
        self.w = np.zeros((self.T, self.r))

        t = 0
        for k in range(self.K):
            Bk = self.B_list[k]
            Ak = self.A_list[k]
            Sigma_k = self.Sigma_eta_list[k]
            n_k = self.segment_lengths[k]

            wt = self.rng.multivariate_normal(np.zeros(self.r), Sigma_k)
            for step in range(n_k):
                self.w[t] = wt
                self.theta[t] = Bk @ wt
                t += 1
                if step < n_k - 1:
                    eta = self.rng.multivariate_normal(np.zeros(self.r), Sigma_k)
                    wt = Ak @ wt + eta

        self.S = float(np.max(np.linalg.norm(self.theta, axis=1)))

    # ------------------------------------------------------------------
    # Per-round interface (matches LowRankLDSEnvironment exactly)
    # ------------------------------------------------------------------

    def get_action_set(self, t: int, rng=None):
        """Return n_actions x d matrix of unit-norm Covertype feature vectors."""
        _rng = rng if rng is not None else self.rng
        idx = _rng.choice(len(self._action_pool), size=self.n_actions, replace=False)
        return self._action_pool[idx].copy()

    def step(self, action: np.ndarray, t: int) -> float:
        """Deploy action at round t, return noisy reward."""
        eps = self.rng.normal(0.0, self.sigma_eps)
        eps = np.clip(eps, -self.L_eps, self.L_eps)
        return float(action @ self.theta[t]) + eps

    def optimal_reward(self, action_set: np.ndarray, t: int) -> float:
        """max_{x in action_set} x^T theta_t."""
        return float(np.max(action_set @ self.theta[t]))

    def segment_projector(self, k: int) -> np.ndarray:
        """True segment projector P_k = B_k B_k^T."""
        B = self.B_list[k]
        return B @ B.T

    def predictable_second_moment(self, t: int) -> np.ndarray:
        """Oracle E[theta_t theta_t^T | H_{t-1}]."""
        k = self.seg_of[t]
        Bk = self.B_list[k]
        wt = self.w[t]
        return Bk @ np.outer(wt, wt) @ Bk.T
