"""Flask app serving movie recommendations.

Endpoints:
    GET  /                          -> web UI
    GET  /health                    -> liveness probe (for AWS/ELB)
    GET  /api/search?q=matrix       -> search the movie catalog
    GET  /api/recommend/<user_id>   -> recommendations for a known user
    POST /api/recommend             -> cold-start: {"liked": [1, 2, 3]}
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Make the project root importable when run as `python app/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import RecommenderService  # noqa: E402

app = Flask(__name__)

# Fit the model once at startup (a few seconds on the small dataset).
service = RecommenderService()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})
    return jsonify({"results": service.search_movies(query, limit=10)})


@app.route("/api/recommend/<int:user_id>")
def recommend_user(user_id: int):
    k = request.args.get("k", default=10, type=int)
    recs = service.recommend_for_user(user_id, k=k)
    return jsonify({"user_id": user_id, "recommendations": recs})


@app.route("/api/recommend", methods=["POST"])
def recommend_cold_start():
    payload = request.get_json(silent=True) or {}
    liked = payload.get("liked", [])
    k = int(payload.get("k", 10))
    if not liked:
        return jsonify({"error": "provide a non-empty 'liked' list of movieIds"}), 400
    recs = service.recommend_from_likes([int(m) for m in liked], k=k)
    return jsonify({"recommendations": recs})


if __name__ == "__main__":
    print("[app] Loading data and fitting the hybrid model...")
    service.load()
    print("[app] Ready. Visit http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
