"""Collaborative filtering.

Two complementary approaches over the user-item rating matrix:

* Matrix factorization (``method="svd"``): a regularized baseline predictor
  (global mean + user bias + item bias) plus truncated-SVD latent factors on
  the residual matrix. This is the standard, strong CF baseline and gives
  well-calibrated rating predictions.
* Item-item CF (``method="item"``): cosine similarity on mean-centered ratings.

Both expose ``predict`` (rating for one user-item pair) and ``recommend``
(top-k for a user).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import Dataset

RATING_MIN, RATING_MAX = 0.5, 5.0


class _MatrixIndex:
    """Maps userId/movieId to contiguous row/column indices."""

    def __init__(self, ratings: pd.DataFrame):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_pos = {int(u): i for i, u in enumerate(self.user_ids)}
        self.movie_pos = {int(m): i for i, m in enumerate(self.movie_ids)}

    def matrix(self, ratings: pd.DataFrame) -> csr_matrix:
        rows = ratings["userId"].map(self.user_pos).to_numpy()
        cols = ratings["movieId"].map(self.movie_pos).to_numpy()
        data = ratings["rating"].to_numpy(dtype=float)
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )


class CollaborativeRecommender:
    def __init__(
        self,
        method: str = "svd",
        n_factors: int = 50,
        reg: float = 10.0,
    ):
        if method not in {"svd", "item"}:
            raise ValueError("method must be 'svd' or 'item'")
        self.method = method
        self.n_factors = n_factors
        self.reg = reg  # regularization for the bias terms

        self.index: _MatrixIndex | None = None
        self.matrix: csr_matrix | None = None
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        # SVD artifacts
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        # Item-item artifacts
        self.item_sim: np.ndarray | None = None
        self.item_means: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "CollaborativeRecommender":
        self.index = _MatrixIndex(dataset.ratings)
        self.matrix = self.index.matrix(dataset.ratings)
        self._fit_biases()

        if self.method == "svd":
            self._fit_svd()
        else:
            self._fit_item_item()
        return self

    # ---- bias baseline ------------------------------------------------------

    def _fit_biases(self) -> None:
        """Regularized global/user/item biases (Koren's baseline predictor)."""
        dense = self.matrix.toarray()
        mask = dense != 0
        self.global_mean = float(dense[mask].mean())

        # Item bias: damped average deviation from the global mean.
        item_counts = mask.sum(axis=0)
        item_dev = np.where(mask, dense - self.global_mean, 0.0).sum(axis=0)
        self.item_bias = item_dev / (self.reg + item_counts)

        # User bias: damped average deviation after removing item bias.
        resid = np.where(mask, dense - self.global_mean - self.item_bias, 0.0)
        user_counts = mask.sum(axis=1)
        self.user_bias = resid.sum(axis=1) / (self.reg + user_counts)

    def _baseline_matrix(self) -> np.ndarray:
        return (
            self.global_mean
            + self.user_bias[:, None]
            + self.item_bias[None, :]
        )

    # ---- SVD ----------------------------------------------------------------

    def _fit_svd(self) -> None:
        dense = self.matrix.toarray()
        mask = dense != 0
        residual = np.where(mask, dense - self._baseline_matrix(), 0.0)

        k = min(self.n_factors, min(residual.shape) - 1)
        k = max(k, 1)
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.user_factors = svd.fit_transform(residual)
        self.item_factors = svd.components_.T

    # ---- item-item ----------------------------------------------------------

    def _fit_item_item(self) -> None:
        dense = self.matrix.toarray()
        mask = dense != 0
        counts = mask.sum(axis=0)
        self.item_means = np.where(counts > 0, dense.sum(axis=0) / np.maximum(counts, 1), 0.0)
        centered = np.where(mask, dense - self.item_means, 0.0)
        self.item_sim = cosine_similarity(centered.T)
        np.fill_diagonal(self.item_sim, 0.0)

    # ---- prediction ---------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give a movie (clipped to scale)."""
        u = self.index.user_pos.get(int(user_id))
        i = self.index.movie_pos.get(int(movie_id))
        if i is None:  # unknown item -> best available prior
            base = self.global_mean + (self.user_bias[u] if u is not None else 0.0)
            return float(np.clip(base, RATING_MIN, RATING_MAX))
        if u is None:  # cold-start user
            return float(np.clip(self.global_mean + self.item_bias[i], RATING_MIN, RATING_MAX))

        base = self.global_mean + self.user_bias[u] + self.item_bias[i]
        if self.method == "svd":
            base += float(self.user_factors[u] @ self.item_factors[i])
        return float(np.clip(base, RATING_MIN, RATING_MAX))

    def _predict_all_for_user(self, user_pos: int) -> np.ndarray:
        base = self.global_mean + self.user_bias[user_pos] + self.item_bias
        if self.method == "svd":
            base = base + self.user_factors[user_pos] @ self.item_factors.T
            return np.clip(base, RATING_MIN, RATING_MAX)

        # item-item weighted average over the user's rated items
        user_row = self.matrix.getrow(user_pos).toarray().ravel()
        rated_mask = user_row != 0
        centered = np.where(rated_mask, user_row - self.item_means, 0.0)
        numer = self.item_sim @ centered
        denom = np.abs(self.item_sim) @ rated_mask.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            preds = np.where(denom > 0, self.item_means + numer / denom, base)
        return np.clip(preds, RATING_MIN, RATING_MAX)

    def recommend(
        self, user_id: int, user_ratings: pd.DataFrame, k: int = 10, exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        if self.index is None or int(user_id) not in self.index.user_pos:
            return []  # cold-start handled by the hybrid layer
        user_pos = self.index.user_pos[int(user_id)]
        preds = self._predict_all_for_user(user_pos).copy()

        if exclude_seen:
            seen = user_ratings["movieId"].map(self.index.movie_pos).dropna()
            if len(seen):
                preds[seen.astype(int).to_numpy()] = -np.inf

        top = np.argsort(preds)[::-1][:k]
        return [(int(self.index.movie_ids[i]), float(preds[i])) for i in top]
