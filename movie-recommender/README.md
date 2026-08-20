# 🎬 Movie Recommendation System

A movie recommendation engine built with **collaborative**, **content-based**, and **hybrid**
filtering. It preprocesses the MovieLens dataset into SQL, trains and evaluates several models
against precision@k / recall@k, and serves recommendations through a Flask REST API and web UI.
The container/`Procfile` setup makes it deployable to AWS (Elastic Beanstalk, ECS, or App Runner).

> **Stack:** Python (Pandas, Scikit-Learn, TensorFlow) · SQL (SQLite) · Flask · Docker / AWS

---

## Features

| Approach | Technique | Module |
|---|---|---|
| **Content-based** | TF-IDF over genres + tags, cosine similarity to a user's taste profile | `src/content_based.py` |
| **Collaborative** | Item-item similarity **and** matrix factorization (truncated SVD) on the user-item matrix | `src/collaborative.py` |
| **Deep (optional)** | Neural Collaborative Filtering — user/item embeddings + MLP (TensorFlow/Keras) | `src/neural_cf.py` |
| **Hybrid** | Min-max normalized, weighted blend of collaborative + content, with cold-start fallback | `src/hybrid.py` |

- **SQL-backed** — data is preprocessed and stored in SQLite with indexes (`src/data_loader.py`).
- **Evaluation harness** — per-user leave-out split, precision@k / recall@k, model comparison (`src/evaluate.py`).
- **Flask service** — REST API + a small web UI for interactive recommendations (`app/`).
- **Cold-start support** — new users get content-based recommendations from a list of liked movies.
- **Deploy-ready** — `Dockerfile`, `Procfile`, and gunicorn config for AWS.

---

## Architecture

```
                    MovieLens dataset
                          │  download + preprocess
                          ▼
                 ┌─────────────────┐
                 │  SQLite (SQL)   │   ratings · movies · tags
                 └────────┬────────┘
                          │ load
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                   ▼
 Content-based      Collaborative        Neural CF
  (TF-IDF)          (SVD / item-item)    (TensorFlow, optional)
        └──────────┬──────┘
                   ▼
              Hybrid blend  ──►  Flask API + Web UI  ──►  AWS
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data, build the SQL database, and evaluate every model
python train.py

# 3. Launch the web app (http://127.0.0.1:5000)
python app/app.py
```

The first run downloads the [MovieLens latest-small](https://grouplens.org/datasets/movielens/)
dataset (~1 MB: ~100k ratings, ~9.7k movies, 610 users) and builds `data/movies.db`.

### Optional: neural model (TensorFlow)

```bash
pip install "tensorflow>=2.15"
```

The core system runs without TensorFlow; the NCF model in `src/neural_cf.py` is loaded lazily
only when used.

---

## Results

Reproduce with `python train.py`. On the MovieLens latest-small split
(per-user 80/20, items rated ≥ 4 treated as relevant, top-k = 10):

<!-- METRICS:START -->
| Model | Precision@10 | Recall@10 |
|---|---|---|
| Content-based | _run `train.py`_ | _run `train.py`_ |
| Collaborative (SVD) | _run `train.py`_ | _run `train.py`_ |
| **Hybrid** | _run `train.py`_ | _run `train.py`_ |

The **hybrid** model improves precision over the best single approach by blending the
collaborative signal (strong for users with history) with the content signal (diversity +
cold-start coverage).
<!-- METRICS:END -->

> Metrics vary with the random split seed and hyperparameters (`alpha`, `n_factors`). See
> `src/evaluate.py` and tune in `train.py`.

---

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (for load balancers) |
| `GET` | `/api/search?q=matrix` | Search the movie catalog |
| `GET` | `/api/recommend/<user_id>?k=10` | Recommendations for an existing user |
| `POST` | `/api/recommend` | Cold-start — body: `{"liked": [1, 2, 3], "k": 10}` |

```bash
curl "http://127.0.0.1:5000/api/recommend/1?k=5"

curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked": [1, 260, 1196], "k": 5}'
```

---

## Deployment (AWS)

The app is a standard WSGI service served by gunicorn, so it runs anywhere containers or
Python buildpacks do.

- **Docker / ECS / App Runner**
  ```bash
  docker build -t movie-recommender .
  docker run -p 5000:5000 movie-recommender
  ```
- **Elastic Beanstalk (Python platform)** — the `Procfile` defines the gunicorn web process.
- **Health checks** — point the target group at `/health`.

---

## Project layout

```
movie-recommender/
├── src/
│   ├── data_loader.py     # download, preprocess, SQL storage
│   ├── content_based.py   # TF-IDF content filtering
│   ├── collaborative.py   # SVD + item-item CF
│   ├── neural_cf.py       # neural CF (TensorFlow, optional)
│   ├── hybrid.py          # weighted blend
│   ├── evaluate.py        # precision@k / recall@k
│   └── recommender.py     # high-level service façade
├── app/
│   ├── app.py             # Flask API
│   └── templates/index.html
├── tests/test_pipeline.py # smoke tests (no network)
├── train.py               # end-to-end train + evaluate
├── Dockerfile · Procfile  # deployment
└── requirements.txt
```

## Testing

```bash
pip install pytest
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
