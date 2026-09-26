"""
MovieLens *real-data-calibrated* bandit environment.

Uses MovieLens 100K (100,000 ratings, 943 users, 1682 movies, 19 genre features).
All parameters derived from real data:
  - Features: movie genre vectors (19 binary), augmented with SVD latent factors for d > 19
  - theta_k: per-segment OLS fit of features -> ratings
  - Segments: data sorted by user-cluster creates natural non-stationarity
  - Low-rank: user preference patterns live in a low-rank subspace

Data auto-downloaded from GroupLens.
"""

import os, zipfile, io, numpy as np

try:
    import urllib.request
except ImportError:
    pass

from sklearn.preprocessing import StandardScaler


_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".movielens_cache")
_ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def _load_movielens_100k():
    """Download and parse MovieLens 100K. Returns (ratings, movie_genres)."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    ratings_path = os.path.join(_CACHE_DIR, "u.data")
    item_path = os.path.join(_CACHE_DIR, "u.item")

    if not os.path.exists(ratings_path):
        print("Downloading MovieLens 100K...", flush=True)
        resp = urllib.request.urlopen(_ML100K_URL)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name in ["ml-100k/u.data", "ml-100k/u.item"]:
            data = zf.read(name)
            out = os.path.join(_CACHE_DIR, os.path.basename(name))
            with open(out, "wb") as f:
                f.write(data)
        print("  Done.", flush=True)

    # Parse ratings: user_id, item_id, rating, timestamp
    ratings = np.loadtxt(ratings_path, dtype=int)

    # Parse item genres (last 19 columns of u.item, pipe-separated)
    genres = []
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            genre_vec = [int(x) for x in parts[-19:]]
            genres.append(genre_vec)
    genres = np.array(genres, dtype=float)  # (1682, 19)

    return ratings, genres


class MovieLensEnvironment:
    """
    Real-data-calibrated MovieLens bandit.

    Segments created by clustering users by preference patterns, then
    sorting so that each segment represents a distinct user cohort.
    theta_k = per-segment OLS of movie features -> ratings.
    """

    def __init__(
        self,
        d: int = 105,
        r: int = 20,
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
        """Load MovieLens 100K, build features to match target_d."""
        ratings, genres = _load_movielens_100k()

        # Build rating matrix (943 users x 1682 movies)
        n_users = int(ratings[:, 0].max())
        n_items = int(ratings[:, 1].max())
        R_mat = np.zeros((n_users, n_items))
        for row in ratings:
            R_mat[row[0] - 1, row[1] - 1] = row[2]

        # SVD of rating matrix for latent factors
        U, S, Vt = np.linalg.svd(R_mat, full_matrices=False)
        # Movie latent factors (top components)
        n_latent = min(50, n_items)
        movie_latent = (Vt[:n_latent].T * S[:n_latent])  # (1682, n_latent)

        # Base features: 19 genre + latent factors
        d_genre = genres.shape[1]  # 19

        scaler_g = StandardScaler()
        genres_std = scaler_g.fit_transform(genres)

        scaler_l = StandardScaler()
        latent_std = scaler_l.fit_transform(movie_latent)

        X_base = np.hstack([genres_std, latent_std])  # (1682, 19+50=69)
        d_base = X_base.shape[1]

        if target_d <= d_base:
            X_full = X_base[:, :target_d]
        else:
            # Add pairwise interactions of genre features
            interactions = []
            for i in range(d_genre):
                for j in range(i + 1, d_genre):
                    interactions.append(genres_std[:, i] * genres_std[:, j])
            if interactions:
                X_inter = np.column_stack(interactions)  # up to 171
                X_full = np.hstack([X_base, X_inter])[:, :target_d]
            else:
                X_full = X_base[:, :target_d]

        # Normalize rows to unit norm
        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        X_full /= np.maximum(norms, 1e-8)

        # User clusters for non-stationarity: cluster users by their latent factors
        user_latent = U[:, :min(10, n_latent)] * S[:min(10, n_latent)]
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(20, n_users), random_state=42, n_init=5)
        user_clusters = km.fit_predict(user_latent)

        # For each rating, create a (feature, label) pair
        # Feature = movie features, label = rating
        feat_list = []
        label_list = []
        cluster_list = []
        for row in ratings:
            uid, iid, rating = int(row[0]) - 1, int(row[1]) - 1, float(row[2])
            feat_list.append(X_full[iid])
            label_list.append(rating)
            cluster_list.append(user_clusters[uid])

        features = np.array(feat_list)
        labels = np.array(label_list)
        clusters = np.array(cluster_list)

        # Sort by user cluster to create non-stationarity
        sort_idx = np.argsort(clusters, kind='stable')
        self._features = features[sort_idx]
        self._labels = labels[sort_idx] - labels.mean()
        self.d = X_full.shape[1]
        self._n_total = len(self._features)

    def _build_segments(self, segment_size, n_segments):
        self.K = n_segments
        total_needed = segment_size * n_segments
        if total_needed > self._n_total:
            segment_size = self._n_total // n_segments
            total_needed = segment_size * n_segments

        step = max(1, self._n_total // total_needed)
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
