"""Smoke tests that run on a tiny synthetic dataset (no network needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import Dataset
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender


@pytest.fixture
def tiny_dataset() -> Dataset:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["A (1999)", "B (2001)", "C (2003)", "D (2005)", "E (2007)"],
            "genres": ["Action|Sci-Fi", "Action|Sci-Fi", "Comedy", "Comedy|Romance", "Action"],
            "year": [1999, 2001, 2003, 2005, 2007],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [1, 2, 3, 1, 5, 4, 3, 4, 2],
            "rating": [5.0, 4.5, 2.0, 5.0, 4.0, 1.5, 5.0, 4.5, 2.0],
            "timestamp": [0] * 9,
        }
    )
    tags = pd.DataFrame(columns=["userId", "movieId", "tag"])
    return Dataset(ratings=ratings, movies=movies, tags=tags)


def test_content_based_recommends(tiny_dataset):
    model = ContentBasedRecommender().fit(tiny_dataset)
    user_ratings = tiny_dataset.ratings[tiny_dataset.ratings["userId"] == 1]
    recs = model.recommend(user_ratings, k=3)
    assert all(mid not in set(user_ratings["movieId"]) for mid, _ in recs)


def test_collaborative_svd_recommends(tiny_dataset):
    model = CollaborativeRecommender(method="svd", n_factors=2).fit(tiny_dataset)
    user_ratings = tiny_dataset.ratings[tiny_dataset.ratings["userId"] == 1]
    recs = model.recommend(1, user_ratings, k=3)
    assert isinstance(recs, list)


def test_item_item_recommends(tiny_dataset):
    model = CollaborativeRecommender(method="item").fit(tiny_dataset)
    user_ratings = tiny_dataset.ratings[tiny_dataset.ratings["userId"] == 2]
    recs = model.recommend(2, user_ratings, k=3)
    assert isinstance(recs, list)


def test_hybrid_returns_breakdown(tiny_dataset):
    model = HybridRecommender(cf_method="svd").fit(tiny_dataset)
    user_ratings = tiny_dataset.ratings[tiny_dataset.ratings["userId"] == 1]
    recs = model.recommend(1, user_ratings, k=3)
    assert recs, "hybrid should return recommendations"
    for r in recs:
        assert {"movieId", "title", "score", "cf_score", "cb_score"} <= set(r)


def test_cold_start_falls_back_to_content(tiny_dataset):
    model = HybridRecommender(cf_method="svd").fit(tiny_dataset)
    synthetic = pd.DataFrame(
        {"userId": -1, "movieId": [1, 2], "rating": 5.0, "timestamp": 0}
    )
    recs = model.recommend(-1, synthetic, k=3)
    assert isinstance(recs, list)
