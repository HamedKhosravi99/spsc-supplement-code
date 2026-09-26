"""
MovieLens *fully real* bandit environment.

Uses MovieLens 100K — all rewards are ACTUAL user ratings (1-5), not synthetic.

Setup:
  - Each round t corresponds to a real user from a specific cluster (segment).
  - Action set = feature vectors of movies that user actually rated.
  - step() returns the user's actual rating (centered), not x^T theta + noise.
  - optimal_reward() returns the max actual rating in the action set.
  - theta/subspaces still computed via OLS for oracle methods and regret normalization.

Non-stationarity: users sorted by KMeans clusters on latent preference factors.
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

    ratings = np.loadtxt(ratings_path, dtype=int)

    genres = []
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            genre_vec = [int(x) for x in parts[-19:]]
            genres.append(genre_vec)
    genres = np.array(genres, dtype=float)

    return ratings, genres


class RealMovieLensEnvironment:
    """
    Fully real MovieLens bandit — rewards are actual user ratings.

    At each round t:
      - A user from the current segment's cluster is selected
      - Action set = movies that user actually rated (so reward is known)
      - step() returns centered actual rating
      - optimal_reward() returns max actual rating in action set
    """

    def __init__(
        self,
        d: int = 105,
        r: int = 20,
        n_actions: int = 40,
        segment_size: int = 500,
        n_segments: int = 10,
        seed: int = 42,
        regret: str = "real",
        probes: str = "snap",
    ):
        # regret="real": optimal_reward is the best real rating in the action set (the played movie is scored
        #   by x^T theta in algorithm.py, a different scale).  regret="fitted": the optimum is max_x x^T theta_t.
        # probes="snap": a played vector outside the action set is replaced by the nearest movie and returns its
        #   real rating.  probes="fitted": it returns x^T theta_t plus Gaussian noise of the residual scale.
        self.regret_mode = regret
        self.probe_mode = probes
        self._probe_rng = np.random.default_rng(seed + 12345)
        self.r = r
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self._load_and_prepare_data(d)
        self._build_segments(segment_size, n_segments)
        self._assign_rounds()
        self._build_theta_and_subspaces()

        self.L_x = 1.0
        self.L = 1.0
        self.L_eps = 5.0
        self.spectral_radius = 0.0
        self.sigma_eta = 0.0

        # Per-round state (set by get_action_set, read by step/optimal_reward)
        self._current_ratings = None

    def _load_and_prepare_data(self, target_d):
        ratings_raw, genres = _load_movielens_100k()

        n_users = int(ratings_raw[:, 0].max())
        n_items = int(ratings_raw[:, 1].max())

        # Build per-user rating dicts
        self._user_ratings = {}
        for row in ratings_raw:
            uid, iid, rating = int(row[0]) - 1, int(row[1]) - 1, float(row[2])
            if uid not in self._user_ratings:
                self._user_ratings[uid] = {}
            self._user_ratings[uid][iid] = rating

        self._global_mean = float(ratings_raw[:, 2].mean())

        # Filter users with enough ratings for action sets
        min_ratings = self.n_actions + 5
        self._valid_users = sorted(
            [u for u, r in self._user_ratings.items() if len(r) >= min_ratings]
        )
        assert len(self._valid_users) >= 50, \
            f"Only {len(self._valid_users)} users with {min_ratings}+ ratings"

        # Build movie feature vectors
        R_mat = np.zeros((n_users, n_items))
        for row in ratings_raw:
            R_mat[row[0] - 1, row[1] - 1] = row[2]

        U, S, Vt = np.linalg.svd(R_mat, full_matrices=False)
        n_latent = min(50, n_items)
        movie_latent = (Vt[:n_latent].T * S[:n_latent])
        user_latent = U[:, :min(10, n_latent)] * S[:min(10, n_latent)]

        d_genre = genres.shape[1]
        scaler_g = StandardScaler()
        genres_std = scaler_g.fit_transform(genres)
        scaler_l = StandardScaler()
        latent_std = scaler_l.fit_transform(movie_latent)

        X_base = np.hstack([genres_std, latent_std])
        d_base = X_base.shape[1]

        if target_d <= d_base:
            X_full = X_base[:, :target_d]
        else:
            interactions = []
            for i in range(d_genre):
                for j in range(i + 1, d_genre):
                    interactions.append(genres_std[:, i] * genres_std[:, j])
            if interactions:
                X_inter = np.column_stack(interactions)
                X_full = np.hstack([X_base, X_inter])[:, :target_d]
            else:
                X_full = X_base[:, :target_d]

        norms = np.linalg.norm(X_full, axis=1, keepdims=True)
        X_full /= np.maximum(norms, 1e-8)
        self._movie_features = X_full
        self.d = X_full.shape[1]

        # Cluster valid users for non-stationarity
        from sklearn.cluster import KMeans
        valid_latent = user_latent[self._valid_users]
        n_clusters = min(20, len(self._valid_users))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = km.fit_predict(valid_latent)

        self._user_cluster = {}
        for i, uid in enumerate(self._valid_users):
            self._user_cluster[uid] = labels[i]
        self._n_clusters = n_clusters

        # Sort valid users by cluster
        self._valid_users.sort(key=lambda u: self._user_cluster[u])

    def _build_segments(self, segment_size, n_segments):
        self.K = n_segments
        self.T = segment_size * n_segments
        self.segment_lengths = [segment_size] * self.K
        self.segment_lengths[-1] += self.T - sum(self.segment_lengths)

        self.tau = [0]
        for l in self.segment_lengths[:-1]:
            self.tau.append(self.tau[-1] + l)

        self.seg_of = np.zeros(self.T, dtype=int)
        for k, start in enumerate(self.tau):
            self.seg_of[start:start + self.segment_lengths[k]] = k

    def _assign_rounds(self):
        """Assign a user to each round t, cycling within segment clusters."""
        # Split valid users roughly evenly across segments
        n_per_seg = max(1, len(self._valid_users) // self.K)
        seg_users = []
        for k in range(self.K):
            start_u = k * n_per_seg
            end_u = (k + 1) * n_per_seg if k < self.K - 1 else len(self._valid_users)
            seg_users.append(self._valid_users[start_u:end_u])
            if not seg_users[-1]:
                seg_users[-1] = self._valid_users[:n_per_seg]

        self._round_user = np.zeros(self.T, dtype=int)
        for k in range(self.K):
            s = self.tau[k]
            e = s + self.segment_lengths[k]
            users_k = seg_users[k]
            for t in range(s, e):
                self._round_user[t] = users_k[t % len(users_k)]

    def _build_theta_and_subspaces(self):
        """Fit theta per segment via OLS (for oracle/regret computation)."""
        d = self.d
        self.theta = np.zeros((self.T, d))
        self.B_list = []
        residuals = []

        for k in range(self.K):
            s = self.tau[k]
            e = s + self.segment_lengths[k]

            # Collect (feature, rating) pairs for this segment
            X_list, y_list = [], []
            for t in range(s, e):
                uid = self._round_user[t]
                for mid, rating in self._user_ratings[uid].items():
                    X_list.append(self._movie_features[mid])
                    y_list.append(rating - self._global_mean)

            X_k = np.array(X_list)
            y_k = np.array(y_list)

            # Subsample if too many points
            if len(y_k) > 5000:
                idx = self.rng.choice(len(y_k), 5000, replace=False)
                X_k, y_k = X_k[idx], y_k[idx]

            lam = 1.0
            theta_k = np.linalg.solve(X_k.T @ X_k + lam * np.eye(d), X_k.T @ y_k)
            resid = y_k - X_k @ theta_k
            residuals.extend(resid[:500].tolist())
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

    # ------------------------------------------------------------------
    # Per-round interface — REAL rewards
    # ------------------------------------------------------------------

    def get_action_set(self, t: int, rng=None):
        _rng = rng if rng is not None else self.rng
        uid = self._round_user[t]
        rated_movies = list(self._user_ratings[uid].keys())

        n = min(self.n_actions, len(rated_movies))
        chosen = _rng.choice(rated_movies, size=n, replace=False)

        # Store real ratings for step() and optimal_reward()
        features = self._movie_features[chosen]
        real_ratings = np.array([
            self._user_ratings[uid][mid] - self._global_mean for mid in chosen
        ])
        self._current_features = features
        self._current_ratings = real_ratings

        return features

    def step(self, action: np.ndarray, t: int) -> float:
        """Return the REAL rating for the selected movie."""
        # Match action to one of the current action set entries
        dists = np.linalg.norm(self._current_features - action[np.newaxis, :], axis=1)
        idx = int(np.argmin(dists))
        if self.probe_mode == "fitted" and dists[idx] > 1e-8:
            # a probe outside the action set: answered by the fitted model
            return float(action @ self.theta[t]) + float(self._probe_rng.normal(0.0, self.sigma_eps))
        return float(self._current_ratings[idx])

    def optimal_reward(self, action_set: np.ndarray, t: int) -> float:
        """Max real rating in the current action set (regret='real'), or max fitted reward (regret='fitted')."""
        if self.regret_mode == "fitted":
            return float(np.max(action_set @ self.theta[t]))
        return float(np.max(self._current_ratings))

    def segment_projector(self, k: int) -> np.ndarray:
        B = self.B_list[k]
        return B @ B.T

    def svd_spectrum(self):
        thetas = np.array([self.theta[self.tau[k]] for k in range(self.K)])
        _, svals, _ = np.linalg.svd(thetas, full_matrices=False)
        return svals / svals.sum()
