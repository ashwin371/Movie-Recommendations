"""Offline evaluation.

Two complementary views:

1. **Like-classification** (headline): on a held-out set of user ratings,
   predict each rating and label it a "like" if the prediction is >= a
   threshold. Compared against the true likes (rating >= threshold) this gives
   precision and recall over the test set. This is the metric the project's
   headline numbers refer to.

2. **Top-k ranking** (diagnostic): precision@k / recall@k over each user's
   held-out relevant movies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .data_loader import Dataset
from .hybrid import HybridRecommender

LIKE_THRESHOLD = 4.0


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    n_pairs: int

    def __str__(self) -> str:
        return (
            f"precision={self.precision:.3f} recall={self.recall:.3f} "
            f"f1={self.f1:.3f} (n={self.n_pairs})"
        )


@dataclass
class RankMetrics:
    precision_at_k: float
    recall_at_k: float
    k: int
    n_users: int

    def __str__(self) -> str:
        return (
            f"precision@{self.k}={self.precision_at_k:.3f} "
            f"recall@{self.k}={self.recall_at_k:.3f} (n_users={self.n_users})"
        )


def train_test_split(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    min_ratings: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user split into (train, test) rating frames."""
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("userId"):
        if len(group) < min_ratings:
            train_parts.append(group)
            continue
        group = group.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(len(group) * test_frac))
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- #
# Like-classification (headline)
# --------------------------------------------------------------------------- #

def _predict_rating(model, user_id, movie_id, user_ratings) -> float | None:
    if isinstance(model, HybridRecommender):
        return model.predict_rating(user_id, movie_id, user_ratings)
    if isinstance(model, CollaborativeRecommender):
        return model.predict(user_id, movie_id)
    if isinstance(model, ContentBasedRecommender):
        return model.predict_rating(user_ratings, movie_id)
    raise TypeError(f"Unsupported model type: {type(model)}")


def classification_metrics(
    model, train_ratings: pd.DataFrame, test: pd.DataFrame, threshold: float = LIKE_THRESHOLD
) -> ClassMetrics:
    """Precision/recall of the 'user will like this movie' decision."""
    # Group the user's training ratings once per user (used by content/hybrid).
    train_by_user = {uid: g for uid, g in train_ratings.groupby("userId")}

    tp = fp = fn = 0
    for uid, group in test.groupby("userId"):
        user_ratings = train_by_user.get(uid)
        if user_ratings is None or user_ratings.empty:
            continue
        for _, row in group.iterrows():
            pred = _predict_rating(model, int(uid), int(row["movieId"]), user_ratings)
            if pred is None:
                continue
            pred_like = pred >= threshold
            true_like = row["rating"] >= threshold
            if pred_like and true_like:
                tp += 1
            elif pred_like and not true_like:
                fp += 1
            elif not pred_like and true_like:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ClassMetrics(precision, recall, f1, n_pairs=tp + fp + fn)


# --------------------------------------------------------------------------- #
# Top-k ranking (diagnostic)
# --------------------------------------------------------------------------- #

def _relevant_sets(test: pd.DataFrame, threshold: float = LIKE_THRESHOLD) -> dict[int, set[int]]:
    rel = {}
    for uid, group in test.groupby("userId"):
        liked = set(group.loc[group["rating"] >= threshold, "movieId"])
        if liked:
            rel[int(uid)] = liked
    return rel


def ranking_metrics(
    model, train_ratings: pd.DataFrame, test: pd.DataFrame, k: int = 10
) -> RankMetrics:
    relevant = _relevant_sets(test)
    train_by_user = {uid: g for uid, g in train_ratings.groupby("userId")}

    precisions, recalls = [], []
    for uid, rel in relevant.items():
        user_ratings = train_by_user.get(uid)
        if user_ratings is None or user_ratings.empty:
            continue
        if isinstance(model, HybridRecommender):
            recs = [r["movieId"] for r in model.recommend(uid, user_ratings, k=k)]
        elif isinstance(model, CollaborativeRecommender):
            recs = [m for m, _ in model.recommend(uid, user_ratings, k=k)]
        else:  # content-based
            recs = [m for m, _ in model.recommend(user_ratings, k=k)]
        if not recs:
            continue
        hits = len(set(recs[:k]) & rel)
        precisions.append(hits / k)
        recalls.append(hits / len(rel))

    n = len(precisions)
    return RankMetrics(
        precision_at_k=float(np.mean(precisions)) if n else 0.0,
        recall_at_k=float(np.mean(recalls)) if n else 0.0,
        k=k,
        n_users=n,
    )


# --------------------------------------------------------------------------- #
# Full comparison
# --------------------------------------------------------------------------- #

def compare_all(dataset: Dataset, k: int = 10) -> dict:
    train_ratings, test = train_test_split(dataset.ratings)
    train = Dataset(ratings=train_ratings, movies=dataset.movies, tags=dataset.tags)

    print("[eval] Fitting models on the training split...")
    models = {
        "content_based": ContentBasedRecommender().fit(train),
        "collaborative_svd": CollaborativeRecommender(method="svd").fit(train),
        "hybrid": HybridRecommender().fit(train),
    }

    classification = {
        name: classification_metrics(m, train_ratings, test) for name, m in models.items()
    }
    ranking = {name: ranking_metrics(m, train_ratings, test, k=k) for name, m in models.items()}

    best_single = max(
        classification["content_based"].precision,
        classification["collaborative_svd"].precision,
    )
    lift = (
        (classification["hybrid"].precision - best_single) / best_single * 100
        if best_single > 0
        else 0.0
    )

    return {
        "classification": classification,
        "ranking": ranking,
        "hybrid_precision_lift_pct": round(lift, 1),
    }
