"""
Standalone Takealot dataset loader.

Does not import from the main project `src` package. Set TAKEALOT_DATA_DIR or pass
`data_root` explicitly to point at the folder that contains customers.csv, products.csv,
reviews.csv, and the reviews/ subdirectory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd

SplitName = Literal["train", "validation", "test", "all"]

# This file lives in standalone_takealot/; parent is the project (Thesis_Research) root.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _builtin_data_candidates() -> list[Path]:
    """Paths to try when no explicit root or env is set (cwd first, then repo-relative)."""
    names = ("take-a-lot-dataset", "takealot")
    out: list[Path] = []
    for name in names:
        out.append(Path("data") / name)
    for name in names:
        out.append(_REPO_ROOT / "data" / name)
    return out


def resolve_takealot_root(data_root: Optional[str | Path] = None) -> Path:
    """
    Resolve dataset directory in order:
    1. Explicit `data_root`
    2. Environment variable TAKEALOT_DATA_DIR
    3. ./data/take-a-lot-dataset, ./data/takealot (relative to cwd)
    4. <repo>/data/take-a-lot-dataset, <repo>/data/takealot (if cwd is elsewhere)
    5. ~/Documents/take-a-lot-dataset
    """
    if data_root is not None:
        p = Path(data_root).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Takealot data directory not found: {p}")
        return p

    env = os.environ.get("TAKEALOT_DATA_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"TAKEALOT_DATA_DIR is not a directory: {p}")
        return p

    for rel in _builtin_data_candidates():
        p = rel.resolve()
        if p.is_dir() and (p / "products.csv").is_file():
            return p

    home_docs = Path.home() / "Documents" / "take-a-lot-dataset"
    if home_docs.is_dir():
        return home_docs.resolve()

    raise FileNotFoundError(
        "Could not find Takealot dataset. Pass data_root=..., set TAKEALOT_DATA_DIR, "
        "or place files under data/take-a-lot-dataset or data/takealot (project data folder), "
        "or ~/Documents/take-a-lot-dataset."
    )


def load_customers(data_root: Optional[str | Path] = None) -> pd.DataFrame:
    root = resolve_takealot_root(data_root)
    path = root / "customers.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_products(data_root: Optional[str | Path] = None) -> pd.DataFrame:
    root = resolve_takealot_root(data_root)
    path = root / "products.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_reviews_raw(split: SplitName, data_root: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load review rows as stored in the CSVs (includes review_id, text, timestamp, etc.).

    split:
      train | validation | test  -> reviews/{split}.csv (validation uses validation.csv)
      all                         -> reviews.csv at dataset root
    """
    root = resolve_takealot_root(data_root)
    if split == "all":
        path = root / "reviews.csv"
    else:
        name = "validation.csv" if split == "validation" else f"{split}.csv"
        path = root / "reviews" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, low_memory=False)


def ratings_from_reviews_df(
    reviews: pd.DataFrame,
    dedupe_user_item: bool = True,
) -> pd.DataFrame:
    """
    Map to columns user_id, item_id, rating (float).

    If dedupe_user_item, keep one row per (user, item): last by review_timestamp when present.
    """
    need = {"customer_id", "product_id", "review_rating"}
    missing = need - set(reviews.columns)
    if missing:
        raise ValueError(f"reviews DataFrame missing columns: {missing}")

    work = reviews.copy()
    work["user_id"] = work["customer_id"].astype(str)
    work["item_id"] = work["product_id"].astype(str)
    work["rating"] = pd.to_numeric(work["review_rating"], errors="coerce")
    work = work.dropna(subset=["rating"])

    if dedupe_user_item:
        if "review_timestamp" in work.columns:
            work = work.sort_values("review_timestamp")
        work = work.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    out = work[["user_id", "item_id", "rating"]].reset_index(drop=True)
    out["rating"] = out["rating"].astype(float)
    return out


def item_features_from_products(products: pd.DataFrame) -> pd.DataFrame:
    """item_id + metadata_text (title, brand, department)."""
    p = products.copy()
    p["item_id"] = p["product_id"].astype(str)
    p["metadata_text"] = (
        p["product_title"].fillna("").astype(str)
        + " "
        + p["product_brand"].fillna("").astype(str)
        + " "
        + p["product_department"].fillna("").astype(str)
    ).str.strip()
    return p[["item_id", "metadata_text"]].drop_duplicates(subset=["item_id"]).reset_index(drop=True)


def load_takealot(
    split: SplitName = "train",
    data_root: Optional[str | Path] = None,
    dedupe_user_item: bool = True,
    attach_item_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (ratings, item_features) aligned with the main project convention.

    ratings: columns user_id, item_id, rating
    item_features: columns item_id, metadata_text (from products.csv; items without a row get "")
    """
    root = resolve_takealot_root(data_root)
    products = load_products(root)
    feats = item_features_from_products(products)

    raw = load_reviews_raw(split, root)
    ratings = ratings_from_reviews_df(raw, dedupe_user_item=dedupe_user_item)

    if attach_item_features:
        known = set(feats["item_id"])
        missing_items = ~ratings["item_id"].isin(known)
        if missing_items.any():
            extra = ratings.loc[missing_items, ["item_id"]].drop_duplicates()
            extra["metadata_text"] = ""
            feats = pd.concat([feats, extra], ignore_index=True)
            feats = feats.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)

    return ratings, feats


def split_summary(data_root: Optional[str | Path] = None) -> pd.DataFrame:
    """Row counts and basic stats per split (and full reviews.csv)."""
    rows = []
    for s in ("train", "validation", "test", "all"):
        raw = load_reviews_raw(s, data_root)
        r = ratings_from_reviews_df(raw, dedupe_user_item=True)
        rows.append(
            {
                "split": s,
                "raw_rows": len(raw),
                "deduped_interactions": len(r),
                "users": r["user_id"].nunique(),
                "items": r["item_id"].nunique(),
                "rating_min": r["rating"].min(),
                "rating_max": r["rating"].max(),
            }
        )
    return pd.DataFrame(rows)
