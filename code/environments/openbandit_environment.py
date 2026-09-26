"""
OpenBandit Pipeline (Saito et al. 2020) real-data bandit environment.

Wraps the publicly-available Japanese fashion e-commerce bandit logs
from ZOZOTOWN.  Unlike the other 'real-data' envs in this repository,
OpenBandit Pipeline ships **real exploration logs**: each (action,
reward, context) triple was actually observed under a logged policy
(random or Thompson-sampling), and the click is genuine 0/1 feedback.

Requires:
    pip install obp

The first time the env is constructed, OBP downloads the small public
sample (~50MB) into ~/.obp_data; subsequent runs use the cache.

Segments: created by bucketing the random-policy timeline into
n_segments equal-time chunks (captures fashion-trend drift across the
log window).
theta_k: per-segment ridge fit of item features -> Bernoulli click rate.
Reward: linear model y = x^T theta_t + eps, with theta_k derived
        from real clicks.

API mirrors RealPendigitsEnvironment so existing experiment scripts
plug in unchanged.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# We don't use obp's OpenBanditDataset loader (it depends on a pandas
# API removed in pandas>=2.0).  We just need the bundled CSV data files,
# which we locate inside the obp install path.
try:
    import obp as _obp
    _OBP_DATA_ROOT = os.path.join(os.path.dirname(_obp.__file__),
                                   "dataset", "obd")
    OBP_AVAILABLE = os.path.isdir(_OBP_DATA_ROOT)
    OBP_IMPORT_ERROR = None if OBP_AVAILABLE else (
        f"obp installed but bundled data dir missing: {_OBP_DATA_ROOT}")
except ImportError as _e:
    _OBP_DATA_ROOT = None
    OBP_AVAILABLE = False
    OBP_IMPORT_ERROR = str(_e)


class OpenBanditEnvironment:
    """OpenBandit Pipeline real-data bandit (ZOZOTOWN fashion logs)."""

    def __init__(
        self,
        d: int = 16,
        r: int = 1,
        n_actions: int = 40,
        segment_size: int = 500,
        n_segments: int = 10,
        seed: int = 42,
        campaign: str = "all",
        behavior_policy: str = "random",
    ):
        if not OBP_AVAILABLE:
            raise ImportError(
                "OpenBandit Pipeline not installed.\n"
                "  Install:  pip install obp\n"
                f"  Original error: {OBP_IMPORT_ERROR}"
            )

        self.r = r
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self._load_obp_data(campaign, behavior_policy, d)
        self._build_segments(segment_size, n_segments)
        self._build_theta_and_subspaces()

        self.L_x = 1.0
        self.L = 1.0
        self.L_eps = 5.0
        self.spectral_radius = 0.0
        self.sigma_eta = 0.0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_obp_data(self, campaign, behavior_policy, target_d):
        """Read the bundled OBD CSVs directly (bypasses obp's loader,
        which is incompatible with pandas>=2.0)."""
        sub = {"all": "all", "men": "men", "women": "women"}[campaign]
        log_csv = os.path.join(_OBP_DATA_ROOT, behavior_policy, sub,
                                f"{sub}.csv")
        item_csv = os.path.join(_OBP_DATA_ROOT, behavior_policy, sub,
                                "item_context.csv")
        if not (os.path.isfile(log_csv) and os.path.isfile(item_csv)):
            raise FileNotFoundError(
                f"OBD CSVs not found for campaign={campaign!r}, "
                f"behavior_policy={behavior_policy!r}.\n"
                f"  log_csv: {log_csv}\n  item_csv: {item_csv}"
            )

        log_df = pd.read_csv(log_csv, index_col=0)
        item_df = pd.read_csv(item_csv, index_col=0)

        actions = log_df["item_id"].to_numpy(dtype=int)
        rewards = log_df["click"].to_numpy(dtype=float)

        # User context: 4 categorical user_feature_* columns (hashed
        # strings) + 80 numeric user-item_affinity_* columns.
        affinity_cols = [c for c in log_df.columns
                         if c.startswith("user-item_affinity_")]
        user_cat_cols = [c for c in log_df.columns
                         if c.startswith("user_feature_")]
        # Hash categorical user features to small one-hot blocks
        BUCKETS = 8
        user_cat_mat = np.zeros((len(log_df), len(user_cat_cols) * BUCKETS),
                                dtype=np.float32)
        for j, col in enumerate(user_cat_cols):
            for i, v in enumerate(log_df[col].to_numpy()):
                bucket = hash((j, str(v))) % BUCKETS
                user_cat_mat[i, j * BUCKETS + bucket] = 1.0
        contexts = np.hstack([
            user_cat_mat,
            log_df[affinity_cols].to_numpy(dtype=np.float32),
        ])

        # Item context: item_feature_0 numeric + 3 categorical strings.
        # Build a (n_items, item_dim) embedding: numeric + hashed cats.
        n_items = item_df["item_id"].max() + 1
        item_cat_cols = [c for c in item_df.columns
                         if c.startswith("item_feature_") and c != "item_feature_0"]
        item_dim = 1 + len(item_cat_cols) * BUCKETS
        action_context = np.zeros((n_items, item_dim), dtype=np.float32)
        for _, row in item_df.iterrows():
            iid = int(row["item_id"])
            action_context[iid, 0] = float(row["item_feature_0"])
            for j, col in enumerate(item_cat_cols):
                bucket = hash((j, str(row[col]))) % BUCKETS
                action_context[iid, 1 + j * BUCKETS + bucket] = 1.0

        # Per-round feature = user context concatenated with arm embedding
        arm_feats = action_context[actions].astype(float)
        full = np.hstack([contexts.astype(float), arm_feats])

        # Standardise (zero-mean, unit-var per column) before dim adjust
        scaler = StandardScaler()
        full = scaler.fit_transform(full)
        d_raw = full.shape[1]

        if target_d <= d_raw:
            X = full[:, :target_d]
        else:
            # Random projection up to target_d
            R = self.rng.standard_normal((d_raw, target_d)) / np.sqrt(d_raw)
            X = full @ R

        # Unit-norm rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.maximum(norms, 1e-8)

        self._features = X
        self._labels = rewards
        self.d = X.shape[1]
        self._n_total = len(self._features)

    # ------------------------------------------------------------------
    # Segmenting
    # ------------------------------------------------------------------
    def _build_segments(self, segment_size, n_segments):
        self.K = n_segments
        total_needed = segment_size * n_segments
        if total_needed > self._n_total:
            segment_size = self._n_total // n_segments
            total_needed = segment_size * n_segments

        # Time-bucket: take every (n_total / total_needed)-th row in
        # time order, preserving temporal drift across segments.
        step = max(self._n_total // total_needed, 1)
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
            self.seg_of[start : start + self.segment_lengths[k]] = k

    # ------------------------------------------------------------------
    # Theta and subspace
    # ------------------------------------------------------------------
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
            theta_k = np.linalg.solve(
                X_k.T @ X_k + lam * np.eye(d), X_k.T @ y_k
            )
            resid = y_k - X_k @ theta_k
            residuals.extend(resid.tolist())
            self.theta[s:e] = theta_k[np.newaxis, :]

            # Top-r subspace via reward-weighted SVD
            n_sub = min(len(y_k), 500)
            w = np.abs(y_k[:n_sub]) + 0.1
            W_diag = np.diag(w)
            X_sub = X_k[:n_sub]
            _, _, Vt = np.linalg.svd(W_diag @ X_sub, full_matrices=False)
            B_k = Vt[: self.r].T
            Q, _ = np.linalg.qr(np.column_stack([theta_k.reshape(-1, 1), B_k]))
            B_k = Q[:, : self.r]
            self.B_list.append(B_k)

        self.sigma_eps = float(np.std(residuals)) if residuals else 0.1
        self.sigma_eps = max(self.sigma_eps, 0.05)
        self.S = float(np.max(np.linalg.norm(self.theta, axis=1)))

    # ------------------------------------------------------------------
    # Per-round bandit interface
    # ------------------------------------------------------------------
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
