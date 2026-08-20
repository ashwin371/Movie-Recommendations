"""High-level façade the web app and CLI use.

Loads data from SQL, fits the hybrid model, and exposes a simple query API
plus lightweight search over the movie catalog.
"""

from __future__ import annotations

import pandas as pd

from .data_loader import Dataset, load_from_db
from .hybrid import HybridRecommender


class RecommenderService:
    def __init__(self, alpha: float = 0.65):
        self.dataset: Dataset | None = None
        self.model = HybridRecommender(alpha=alpha)
        self._fitted = False

    def load(self) -> "RecommenderService":
        self.dataset = load_from_db()
        self.model.fit(self.dataset)
        self._fitted = True
        return self

    def _ensure(self) -> None:
        if not self._fitted:
            self.load()

    def recommend_for_user(self, user_id: int, k: int = 10) -> list[dict]:
        self._ensure()
        user_ratings = self.dataset.ratings[self.dataset.ratings["userId"] == user_id]
        return self.model.recommend(user_id, user_ratings, k=k)

    def recommend_from_likes(self, liked_movie_ids: list[int], k: int = 10) -> list[dict]:
        """Cold-start path: recommend from a list of liked movies (rating 5)."""
        self._ensure()
        synthetic = pd.DataFrame(
            {"userId": -1, "movieId": liked_movie_ids, "rating": 5.0, "timestamp": 0}
        )
        # user_id=-1 is unknown to CF, so the hybrid falls back to content.
        return self.model.recommend(-1, synthetic, k=k)

    def search_movies(self, query: str, limit: int = 10) -> list[dict]:
        self._ensure()
        movies = self.dataset.movies
        mask = movies["title"].str.contains(query, case=False, na=False)
        hits = movies[mask].head(limit)
        return hits[["movieId", "title", "genres", "year"]].to_dict("records")

    def user_ids(self) -> list[int]:
        self._ensure()
        return sorted(self.dataset.ratings["userId"].unique().tolist())
