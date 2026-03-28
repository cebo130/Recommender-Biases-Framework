import gc
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd


DatasetName = Literal["ml-100k", "ml-1m", "lastfm-2k", "book-crossing"]


def ensure_output_dirs(base_dir: str | Path = ".") -> Dict[str, Path]:
    root = Path(base_dir)
    figures = root / "outputs" / "figures"
    results = root / "outputs" / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, "results": results}


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        c = df[col]
        if pd.api.types.is_integer_dtype(c):
            df[col] = pd.to_numeric(c, downcast="integer")
        elif pd.api.types.is_float_dtype(c):
            df[col] = pd.to_numeric(c, downcast="float")
    return df


def k_core_filter(
    ratings: pd.DataFrame,
    min_user_interactions: int = 10,
    min_item_interactions: int = 10,
    max_iters: int = 20,
) -> pd.DataFrame:
    filtered = ratings.copy()
    for _ in range(max_iters):
        start_n = len(filtered)
        user_counts = filtered["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["user_id"].isin(keep_users)]

        item_counts = filtered["item_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered["item_id"].isin(keep_items)]

        if len(filtered) == start_n:
            break
    filtered = filtered.reset_index(drop=True)
    gc.collect()
    return filtered


def _load_ml100k(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    ratings = ratings[["user_id", "item_id", "rating"]]

    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python",
    )
    genre_text = items[genre_cols].astype(int).apply(
        lambda row: " ".join([g for g in genre_cols if row[g] == 1]), axis=1
    )
    item_features = pd.DataFrame(
        {"item_id": items["item_id"].astype(str), "metadata_text": genre_text.fillna("").astype(str)}
    )

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = ratings["rating"].astype(float)
    return reduce_memory(ratings), item_features


def _load_ml1m(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        dataset_dir / "ratings.dat",
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    ratings = ratings[["user_id", "item_id", "rating"]]

    movies = pd.read_csv(
        dataset_dir / "movies.dat",
        sep="::",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
        engine="python",
    )
    item_features = pd.DataFrame(
        {
            "item_id": movies["item_id"].astype(str),
            "metadata_text": movies["genres"].fillna("").str.replace("|", " ", regex=False).astype(str),
        }
    )

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = ratings["rating"].astype(float)
    return reduce_memory(ratings), item_features


def _load_lastfm(dataset_dir: Path, implicit_mode: Literal["binary", "1-5"] = "1-5") -> Tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(dataset_dir / "user_artists.dat", sep="\t")
    interactions = interactions.rename(
        columns={"userID": "user_id", "artistID": "item_id", "weight": "play_count"}
    )

    if implicit_mode == "binary":
        interactions["rating"] = 1.0
    else:
        # Quantile binning to map skewed play counts into 1-5.
        q = interactions["play_count"].rank(method="first", pct=True)
        interactions["rating"] = np.clip(np.ceil(q * 5.0), 1, 5).astype(float)

    ratings = interactions[["user_id", "item_id", "rating"]]

    tags = pd.read_csv(dataset_dir / "tags.dat", sep="\t", encoding="latin-1")
    user_tags = pd.read_csv(dataset_dir / "user_taggedartists.dat", sep="\t")
    tags = tags.rename(columns={"tagID": "tag_id", "tagValue": "tag_value"})
    user_tags = user_tags.rename(columns={"artistID": "item_id", "tagID": "tag_id"})

    merged = user_tags.merge(tags[["tag_id", "tag_value"]], on="tag_id", how="left")
    merged["tag_value"] = merged["tag_value"].fillna("").astype(str)
    tag_strings = merged.groupby("item_id", as_index=False)["tag_value"].apply(
        # Keep order while deduplicating tags; avoid pandas unique(list) incompatibility.
        lambda x: " ".join(dict.fromkeys([v for v in x if v]))
    )
    tag_strings = tag_strings.rename(columns={"tag_value": "metadata_text"})

    artists = pd.read_csv(dataset_dir / "artists.dat", sep="\t", encoding="latin-1")
    artists = artists.rename(columns={"id": "item_id", "name": "artist_name"})
    item_features = artists[["item_id", "artist_name"]].merge(tag_strings, on="item_id", how="left")
    item_features["metadata_text"] = (
        item_features["artist_name"].fillna("").astype(str)
        + " "
        + item_features["metadata_text"].fillna("").astype(str)
    ).str.strip()
    item_features = item_features[["item_id", "metadata_text"]]

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = ratings["rating"].astype(float)
    item_features["item_id"] = item_features["item_id"].astype(str)
    return reduce_memory(ratings), item_features


def _load_book_crossing(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(dataset_dir / "Ratings.csv", sep=";", encoding="latin-1")
    ratings = ratings.rename(
        columns={
            "User-ID": "user_id",
            "UserID": "user_id",
            "ISBN": "item_id",
            "Book-Rating": "rating",
            "Rating": "rating",
        }
    )[["user_id", "item_id", "rating"]]

    # Keep explicit feedback to align with RMSE calculations.
    ratings = ratings[ratings["rating"] > 0].copy()

    books = pd.read_csv(dataset_dir / "Books.csv", sep=";", encoding="latin-1", low_memory=False)
    books = books.rename(
        columns={
            "ISBN": "item_id",
            "Book-Author": "author",
            "Author": "author",
            "Publisher": "publisher",
        }
    )
    books["metadata_text"] = (
        books["author"].fillna("").astype(str) + " " + books["publisher"].fillna("").astype(str)
    ).str.strip()
    item_features = books[["item_id", "metadata_text"]]

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = ratings["rating"].astype(float)
    item_features["item_id"] = item_features["item_id"].astype(str)
    return reduce_memory(ratings), item_features


def load_dataset(
    dataset_name: DatasetName,
    data_root: str | Path = "data",
    lastfm_mode: Literal["binary", "1-5"] = "1-5",
    apply_k_core: bool = True,
    min_user_interactions: int = 10,
    min_item_interactions: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if dataset_name == "ml-100k":
        ratings, item_features = _load_ml100k(dataset_dir)
    elif dataset_name == "ml-1m":
        ratings, item_features = _load_ml1m(dataset_dir)
    elif dataset_name == "lastfm-2k":
        ratings, item_features = _load_lastfm(dataset_dir, implicit_mode=lastfm_mode)
    elif dataset_name == "book-crossing":
        ratings, item_features = _load_book_crossing(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    if apply_k_core:
        ratings = k_core_filter(
            ratings,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )
        item_features = item_features[item_features["item_id"].isin(ratings["item_id"].unique())].copy()

    item_features["metadata_text"] = item_features["metadata_text"].fillna("").astype(str)
    item_features = item_features.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    gc.collect()
    return ratings.reset_index(drop=True), item_features

