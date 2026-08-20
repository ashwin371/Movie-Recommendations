"""Neural Collaborative Filtering (optional, TensorFlow/Keras).

Learns user and item embeddings and passes their concatenation through an MLP
to predict ratings — a neural generalization of matrix factorization. This
module is optional: it imports TensorFlow lazily so the rest of the system
runs without it. Install with:  pip install "tensorflow>=2.15"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import Dataset


class NeuralCFRecommender:
    def __init__(self, n_factors: int = 32, epochs: int = 5, batch_size: int = 256):
        self.n_factors = n_factors
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.user_ids: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self.user_pos: dict[int, int] = {}
        self.movie_pos: dict[int, int] = {}

    def _build(self, n_users: int, n_items: int):
        # Lazy import so TensorFlow is only required when this model is used.
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        user_in = layers.Input(shape=(1,), name="user")
        item_in = layers.Input(shape=(1,), name="item")

        user_emb = layers.Embedding(n_users, self.n_factors)(user_in)
        item_emb = layers.Embedding(n_items, self.n_factors)(item_in)

        x = layers.Concatenate()([layers.Flatten()(user_emb), layers.Flatten()(item_emb)])
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1)(x)

        model = Model([user_in, item_in], out)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, dataset: Dataset) -> "NeuralCFRecommender":
        ratings = dataset.ratings
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_pos = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_pos = {m: i for i, m in enumerate(self.movie_ids)}

        u = ratings["userId"].map(self.user_pos).to_numpy()
        i = ratings["movieId"].map(self.movie_pos).to_numpy()
        y = ratings["rating"].to_numpy(dtype=float)

        self.model = self._build(len(self.user_ids), len(self.movie_ids))
        self.model.fit(
            [u, i], y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_split=0.1,
        )
        return self

    def recommend(
        self, user_id: int, user_ratings: pd.DataFrame, k: int = 10, exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        if self.model is None or user_id not in self.user_pos:
            return []
        u_pos = self.user_pos[user_id]
        item_positions = np.arange(len(self.movie_ids))
        u_arr = np.full_like(item_positions, u_pos)
        preds = self.model.predict([u_arr, item_positions], verbose=0).ravel()

        if exclude_seen:
            for mid in user_ratings["movieId"]:
                pos = self.movie_pos.get(int(mid))
                if pos is not None:
                    preds[pos] = -np.inf

        top = np.argsort(preds)[::-1][:k]
        return [(int(self.movie_ids[p]), float(preds[p])) for p in top]
