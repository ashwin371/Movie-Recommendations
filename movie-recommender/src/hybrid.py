"""Hybrid recommender.

Blends normalized scores from the collaborative and content-based models. The
collaborative signal is strongest for users with rating history; the content
signal covers cold-start users and adds diversity. A weighted linear
combination lets you trade the two off with a single ``alpha``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .data_loader import Dataset


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Scale a dict of scores into [0, 1]; flat inputs map to 0."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {mid: 0.0 for mid in scores}
    return {mid: (s - lo) / (hi - lo) for mid, s in scores.items()}


class HybridRecommender:
    def __init__(self, alpha: float = 0.65, cf_method: str = "svd"):
        """``alpha`` weights collaborative filtering; ``1 - alpha`` content."""
        self.alpha = alpha
        self.cf = CollaborativeRecommender(method=cf_method)
        self.cb = ContentBasedRecommender()
        self.titles: dict[int, str] = {}

    def fit(self, dataset: Dataset) -> "HybridRecommender":
        self.cf.fit(dataset)
        self.cb.fit(dataset)
        self.titles = dict(zip(dataset.movies["movieId"], dataset.movies["title"]))
        return self

    def predict_rating(
        self, user_id: int, movie_id: int, user_ratings: pd.DataFrame
    ) -> float:
        """Blend collaborative and content-based rating predictions.

        Where both signals exist we average them (weighted by ``alpha``); if the
        content signal is missing (e.g. no similar rated items) we defer fully to
        collaborative filtering.
        """
        cf_pred = self.cf.predict(user_id, movie_id)
        cb_pred = self.cb.predict_rating(user_ratings, movie_id)
        if cb_pred is None:
            return cf_pred
        return self.alpha * cf_pred + (1 - self.alpha) * cb_pred

    def recommend(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        k: int = 10,
        candidate_pool: int = 100,
    ) -> list[dict]:
        """Return ranked recommendations with per-model score breakdowns."""
        cf_raw = dict(
            self.cf.recommend(user_id, user_ratings, k=candidate_pool)
        )
        cb_raw = dict(self.cb.recommend(user_ratings, k=candidate_pool))

        cf_norm = _minmax(cf_raw)
        cb_norm = _minmax(cb_raw)

        # If one source is empty (e.g. cold-start), fall back to the other.
        alpha = self.alpha
        if not cf_norm:
            alpha = 0.0
        elif not cb_norm:
            alpha = 1.0

        movie_ids = set(cf_norm) | set(cb_norm)
        blended = {
            mid: alpha * cf_norm.get(mid, 0.0) + (1 - alpha) * cb_norm.get(mid, 0.0)
            for mid in movie_ids
        }

        top = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            {
                "movieId": mid,
                "title": self.titles.get(mid, str(mid)),
                "score": round(score, 4),
                "cf_score": round(cf_norm.get(mid, 0.0), 4),
                "cb_score": round(cb_norm.get(mid, 0.0), 4),
            }
            for mid, score in top
        ]
