"""Download, preprocess, and load the MovieLens dataset.

Ratings and movie metadata are persisted to a local SQLite database so the
rest of the pipeline reads from SQL rather than raw CSVs. This mirrors a
production setup where a relational store backs the recommender.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# MovieLens "latest-small": ~100k ratings, ~9.7k movies, 610 users. ~1 MB.
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "ml-latest-small"
DB_PATH = DATA_DIR / "movies.db"


@dataclass
class Dataset:
    """In-memory view of the preprocessed data."""

    ratings: pd.DataFrame       # columns: userId, movieId, rating, timestamp
    movies: pd.DataFrame        # columns: movieId, title, genres, year
    tags: pd.DataFrame          # columns: userId, movieId, tag


def download(force: bool = False) -> None:
    """Fetch and unzip the MovieLens dataset into ``data/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists() and not force:
        print(f"[data] Dataset already present at {RAW_DIR}")
        return

    print(f"[data] Downloading {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)
    print(f"[data] Extracted to {RAW_DIR}")


def _extract_year(title: str) -> int | None:
    """Pull the release year out of a title like 'Toy Story (1995)'."""
    title = title.strip()
    if title.endswith(")") and "(" in title:
        candidate = title[title.rfind("(") + 1 : -1]
        if candidate.isdigit() and len(candidate) == 4:
            return int(candidate)
    return None


def preprocess() -> Dataset:
    """Load the raw CSVs and clean them into tidy DataFrames."""
    if not RAW_DIR.exists():
        download()

    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    tags_path = RAW_DIR / "tags.csv"
    tags = (
        pd.read_csv(tags_path)[["userId", "movieId", "tag"]]
        if tags_path.exists()
        else pd.DataFrame(columns=["userId", "movieId", "tag"])
    )

    # Genres come as a pipe-delimited string; normalize "(no genres listed)".
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies["year"] = movies["title"].map(_extract_year).astype("Int64")

    # Drop ratings that reference movies not in the metadata (data integrity).
    movies_ids = set(movies["movieId"])
    ratings = ratings[ratings["movieId"].isin(movies_ids)].reset_index(drop=True)

    return Dataset(ratings=ratings, movies=movies, tags=tags)


def build_database(dataset: Dataset | None = None, db_path: Path = DB_PATH) -> Path:
    """Write the preprocessed data to a SQLite database and index it."""
    dataset = dataset or preprocess()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        dataset.ratings.to_sql("ratings", conn, if_exists="replace", index=False)
        dataset.movies.to_sql("movies", conn, if_exists="replace", index=False)
        dataset.tags.to_sql("tags", conn, if_exists="replace", index=False)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_ratings_user  ON ratings(userId);
            CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId);
            CREATE INDEX IF NOT EXISTS idx_movies_movie  ON movies(movieId);
            """
        )
    print(f"[data] SQLite database written to {db_path}")
    return db_path


def load_from_db(db_path: Path = DB_PATH) -> Dataset:
    """Load the dataset back out of SQLite (building it first if needed)."""
    if not db_path.exists():
        build_database(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        movies = pd.read_sql("SELECT * FROM movies", conn)
        tags = pd.read_sql("SELECT * FROM tags", conn)
    return Dataset(ratings=ratings, movies=movies, tags=tags)


if __name__ == "__main__":
    download()
    ds = preprocess()
    build_database(ds)
    print(
        f"[data] {len(ds.ratings):,} ratings | "
        f"{len(ds.movies):,} movies | "
        f"{ds.ratings['userId'].nunique():,} users"
    )
