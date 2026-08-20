"""Content-based filtering.

Represents each movie as a TF-IDF vector over its genres and user-supplied
tags, then recommends movies whose vectors are most similar (cosine) to the
ones a user already likes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import Dataset


class ContentBasedRecommender:
    def __init__(self, min_like_rating: float = 4.0):
        # Ratings >= this threshold define a user's "liked" profile.
        self.min_like_rating = min_like_rating
        self.movie_ids: np.ndarray | None = None
        self.index_of: dict[int, int] = {}
        self.tfidf_matrix = None
        self.titles: dict[int, str] = {}

    def _build_corpus(self, dataset: Dataset) -> pd.Series:
        """Combine genres and aggregated tags into one text field per movie."""
        movies = dataset.movies.copy()

        # Genres: "Action|Adventure" -> "Action Adventure"
        genre_text = movies["genres"].fillna("").str.replace("|", " ", regex=False)

        # Tags: join all tags per movie into a single string.
        if not dataset.tags.empty:
            tag_text = (
                dataset.tags.groupby("movieId")["tag"]
                .apply(lambda s: " ".join(str(t) for t in s))
                .rename("tag_text")
            )
            movies = movies.merge(tag_text, on="movieId", how="left")
            movies["tag_text"] = movies["tag_text"].fillna("")
        else:
            movies["tag_text"] = ""

        corpus = (genre_text + " " + movies["tag_text"]).str.strip()
        return movies["movieId"], corpus, movies["title"]

    def fit(self, dataset: Dataset) -> "ContentBasedRecommender":
        movie_ids, corpus, titles = self._build_corpus(dataset)
        self.movie_ids = movie_ids.to_numpy()
        self.index_of = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.titles = dict(zip(movie_ids, titles))

        vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=1)
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        return self

    def _user_profile(self, user_ratings: pd.DataFrame) -> np.ndarray | None:
        """Weighted average of TF-IDF vectors for movies the user liked."""
        liked = user_ratings[user_ratings["rating"] >= self.min_like_rating]
        rows, weights = [], []
        for _, r in liked.iterrows():
            idx = self.index_of.get(int(r["movieId"]))
            if idx is not None:
                rows.append(idx)
                weights.append(r["rating"])
        if not rows:
            return None
        weights = np.asarray(weights).reshape(-1, 1)
        profile = self.tfidf_matrix[rows].multiply(weights).sum(axis=0)
        return np.asarray(profile)

    def predict_rating(self, user_ratings: pd.DataFrame, movie_id: int) -> float | None:
        """Predict a rating from content similarity to the user's rated movies.

        rating(i) = Σ_j sim(i, j) · r_j / Σ_j sim(i, j), over items j the user
        rated whose content is positively similar to item i.
        """
        target = self.index_of.get(int(movie_id))
        if target is None:
            return None

        rows, ratings = [], []
        for _, r in user_ratings.iterrows():
            idx = self.index_of.get(int(r["movieId"]))
            if idx is not None and idx != target:
                rows.append(idx)
                ratings.append(r["rating"])
        if not rows:
            return None

        sims = cosine_similarity(
            self.tfidf_matrix[target], self.tfidf_matrix[rows]
        ).ravel()
        pos = sims > 0
        if not pos.any():
            return float(np.mean(ratings))  # fall back to the user's mean
        weights = sims[pos]
        vals = np.asarray(ratings)[pos]
        return float(np.dot(weights, vals) / weights.sum())

    def recommend(
        self, user_ratings: pd.DataFrame, k: int = 10, exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` (movieId, score) pairs for a user."""
        profile = self._user_profile(user_ratings)
        if profile is None:
            return []

        scores = cosine_similarity(profile, self.tfidf_matrix).ravel()

        if exclude_seen:
            for mid in user_ratings["movieId"]:
                idx = self.index_of.get(int(mid))
                if idx is not None:
                    scores[idx] = -np.inf

        top = np.argsort(scores)[::-1][:k]
        return [(int(self.movie_ids[i]), float(scores[i])) for i in top if scores[i] > 0]
