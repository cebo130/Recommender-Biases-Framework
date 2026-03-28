#!/usr/bin/env python3
"""
Smoke test for the standalone Takealot loader + simple popularity-bias visuals.

Usage:
  python run_takealot_smoke.py
  python run_takealot_smoke.py --data-root "C:\\path\\to\\take-a-lot-dataset"
  python run_takealot_smoke.py --no-plots

Outputs (unless --no-plots): standalone_takealot/outputs/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from standalone_takealot/
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from takealot_loader import (
    load_products,
    load_reviews_raw,
    load_takealot,
    ratings_from_reviews_df,
    resolve_takealot_root,
    split_summary,
)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / x.sum() - (n + 1)) / n)


def _top_share(counts: np.ndarray, frac: float = 0.1) -> float:
    """Share of total mass in the top `frac` fraction of items (by count)."""
    counts = np.sort(counts[counts > 0])[::-1]
    if counts.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * counts.size)))
    return float(counts[:k].sum() / counts.sum())


def popularity_table(ratings: pd.DataFrame) -> pd.DataFrame:
    c = ratings.groupby("item_id", as_index=False).size().rename(columns={"size": "n_ratings"})
    c = c.sort_values("n_ratings", ascending=False).reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Takealot standalone smoke test")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to take-a-lot-dataset folder (overrides TAKEALOT_DATA_DIR and defaults)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG figures")
    args = parser.parse_args()

    root = resolve_takealot_root(args.data_root)
    print(f"Takealot root: {root}")

    products = load_products(root)
    print(f"products.csv rows: {len(products):,}  unique product_id: {products['product_id'].nunique():,}")

    customers_path = root / "customers.csv"
    if customers_path.is_file():
        cust = pd.read_csv(customers_path)
        print(f"customers.csv rows: {len(cust):,}")

    print("\nSplit summary (after user-item dedupe by latest timestamp):")
    summary = split_summary(root)
    print(summary.to_string(index=False))

    train_r, train_feats = load_takealot("train", root, dedupe_user_item=True)
    print(f"\nload_takealot('train'): {len(train_r):,} ratings, {train_feats['item_id'].nunique():,} item feature rows")

    pop = popularity_table(train_r)
    counts = pop["n_ratings"].values
    print("\nTrain item popularity (deduped interactions):")
    print(f"  items with >=1 rating: {len(pop):,}")
    print(f"  Gini (item interaction counts): {_gini(counts):.4f}")
    print(f"  Top 10% of items account for {_top_share(counts, 0.1)*100:.1f}% of interactions")

    if args.no_plots:
        print("\n--no-plots: skipping figures.")
        return

    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Interactions per item (train)")
    ax.set_ylabel("Number of items (log scale)")
    ax.set_title("Takealot — train item popularity histogram")
    fig.tight_layout()
    p1 = out_dir / "takealot_train_popularity_hist.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {p1}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ranks = pop["rank"].values
    ax.loglog(ranks, pop["n_ratings"].values, color="darkred", linewidth=1.2)
    ax.set_xlabel("Item rank by popularity")
    ax.set_ylabel("Interactions (train)")
    ax.set_title("Takealot — train popularity rank curve (log-log)")
    fig.tight_layout()
    p2 = out_dir / "takealot_train_popularity_loglog.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p2}")

    print("\nSmoke test finished OK.")


if __name__ == "__main__":
    main()
