"""End-to-end training + evaluation pipeline.

Usage:
    python train.py                # download data, build DB, evaluate all models
    python train.py --k 10         # set the top-k for metrics
"""

from __future__ import annotations

import argparse

from src.data_loader import build_database, download, load_from_db, preprocess
from src.evaluate import compare_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate recommenders")
    parser.add_argument("--k", type=int, default=10, help="top-k for precision/recall")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    dataset = preprocess()
    build_database(dataset)
    dataset = load_from_db()

    print("\n=== Dataset ===")
    print(
        f"{len(dataset.ratings):,} ratings | {len(dataset.movies):,} movies | "
        f"{dataset.ratings['userId'].nunique():,} users\n"
    )

    print("=== Model comparison ===")
    results = compare_all(dataset, k=args.k)

    print("\nLike-classification (headline: predict rating >= 4.0):")
    for name in ("content_based", "collaborative_svd", "hybrid"):
        print(f"  {name:>18}: {results['classification'][name]}")

    print(f"\nTop-{args.k} ranking (diagnostic):")
    for name in ("content_based", "collaborative_svd", "hybrid"):
        print(f"  {name:>18}: {results['ranking'][name]}")

    lift = results.get("hybrid_precision_lift_pct")
    print(f"\nHybrid precision lift over best single model: {lift:+.1f}%")


if __name__ == "__main__":
    main()
