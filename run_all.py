import argparse
import gc
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
try:
    from surprise import Dataset, KNNBasic, Reader, SVD

    SURPRISE_AVAILABLE = True
except Exception:
    SURPRISE_AVAILABLE = False
    Dataset = None
    Reader = None
    KNNBasic = None
    SVD = None

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(SRC_DIR))

from statistical_tests import (  # noqa: E402
    mann_whitney_pvalue,
    per_user_ndcg_by_segment,
    per_user_rmse_by_segment,
    significance_flag,
    welch_ttest_pvalue,
)
from utils import ensure_output_dirs, load_dataset  # noqa: E402


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


class _FallbackCF:
    def __init__(self):
        self.global_mean = 0.0
        self.user_mean = {}
        self.item_mean = {}
        self.user_bias = {}
        self.item_bias = {}

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean = train_df.groupby("item_id")["rating"].mean().to_dict()
        self.user_bias = {u: m - self.global_mean for u, m in self.user_mean.items()}
        self.item_bias = {i: m - self.global_mean for i, m in self.item_mean.items()}
        return self

    def predict(self, user_id, item_id):
        # Baseline CF fallback: global mean + user bias + item bias.
        pred = self.global_mean + self.user_bias.get(user_id, 0.0) + self.item_bias.get(item_id, 0.0)
        return float(pred)


def popularity_series(df: pd.DataFrame) -> pd.Series:
    return df["item_id"].value_counts().astype(float)


def gini_coefficient(x) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.amin(arr) < 0:
        arr = arr - np.amin(arr)
    arr = arr + 1e-9
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr)))


def ndcg_at_k(recommended_items, relevant_items, k: int = 10) -> float:
    rel_set = set(relevant_items)
    dcg = 0.0
    for i, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in rel_set:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(rel_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = np.sum([1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1)])
    return float(dcg / idcg)


def build_user_seen(train_df: pd.DataFrame) -> dict:
    return train_df.groupby("user_id")["item_id"].apply(set).to_dict()


def surprise_prepare(train_df: pd.DataFrame):
    if not SURPRISE_AVAILABLE:
        return train_df
    reader = Reader(
        rating_scale=(float(train_df["rating"].min()), float(train_df["rating"].max()))
    )
    data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    return data.build_full_trainset()


def recommend_with_surprise(algo, users, user_seen, all_items, k: int = 10) -> dict:
    def _est(pred_obj):
        return pred_obj.est if hasattr(pred_obj, "est") else float(pred_obj)

    recs = {}
    for u in tqdm(users, desc="Surprise Top-K"):
        seen = user_seen.get(u, set())
        candidates = [i for i in all_items if i not in seen]
        scored = [(it, _est(algo.predict(u, it))) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs[u] = [it for it, _ in scored[:k]]
    return recs


def rmse_surprise(algo, test_df: pd.DataFrame) -> float:
    preds = []
    for r in test_df.itertuples(index=False):
        pred_obj = algo.predict(r.user_id, r.item_id)
        preds.append(pred_obj.est if hasattr(pred_obj, "est") else float(pred_obj))
    return float(np.sqrt(mean_squared_error(test_df["rating"].values, preds)))


def est_predict(algo, user_id, item_id) -> float:
    pred_obj = algo.predict(user_id, item_id)
    return pred_obj.est if hasattr(pred_obj, "est") else float(pred_obj)


def recommendation_frequency(recs: dict) -> pd.Series:
    freq = {}
    for items in recs.values():
        for item_id in items:
            freq[item_id] = freq.get(item_id, 0) + 1
    return pd.Series(freq, dtype=float)


def compute_catalog_coverage(recs: dict, all_items) -> float:
    if not recs:
        return 0.0
    rec_items = set([it for items in recs.values() for it in items])
    return float(len(rec_items) / max(1, len(all_items)))


def compute_arp(recs: dict, pop_map: dict) -> float:
    vals = [pop_map.get(it, 0.0) for items in recs.values() for it in items]
    return float(np.mean(vals)) if vals else 0.0


def train_popularity_percentile_map(train_df: pd.DataFrame) -> dict:
    """
    Per-item percentile in [0, 1] where 1 = most interactions in training (head / mainstream).
    """
    vc = train_df["item_id"].value_counts()
    pct = vc.rank(pct=True, ascending=False)
    return pct.to_dict()


def avg_popularity_percentile(recs: dict, pct_map: dict) -> float:
    vals = [pct_map.get(it, np.nan) for items in recs.values() for it in items]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def long_tail_share(recs: dict, pct_map: dict, tail_cutoff: float = 0.2) -> float:
    """
    Fraction of recommendation slots where the item is in the least-popular tail
    of the training catalog (pct_mainstream <= tail_cutoff).
    """
    total = 0
    tail = 0
    for items in recs.values():
        for it in items:
            total += 1
            p = pct_map.get(it, np.nan)
            if np.isfinite(p) and p <= tail_cutoff:
                tail += 1
    return float(tail / total) if total else 0.0


def aggregate_diversity_unique_items(recs: dict) -> int:
    return len({it for items in recs.values() for it in items})


def metadata_token_entropy(recs: dict, item_features: pd.DataFrame) -> float:
    """
    Shannon entropy (bits) over whitespace token counts in recommended items' metadata_text.
    Same pipeline for all datasets; interpret as exposure diversity of metadata tokens.
    """
    meta = dict(zip(item_features["item_id"].astype(str), item_features["metadata_text"].fillna("")))
    c = Counter()
    for items in recs.values():
        for it in items:
            text = str(meta.get(str(it), "")).lower()
            for tok in text.split():
                if tok:
                    c[tok] += 1
    total = sum(c.values())
    if total == 0:
        return float("nan")
    ent = 0.0
    for n in c.values():
        p = n / total
        ent -= p * np.log2(p)
    return float(ent)


def plot_popularity_cdf_compare(
    train_pop: pd.Series,
    model_recs: dict,
    fig_path: Path,
    dataset_name: str,
):
    """ECDF of log10(train count) for training catalog vs multiset of recommended items per model."""
    train_vals = np.log10(train_pop.values.astype(float) + 1.0)
    train_vals = np.sort(train_vals)

    def ecdf(x):
        x = np.sort(np.asarray(x, dtype=float))
        y = np.arange(1, len(x) + 1) / max(1, len(x))
        return x, y

    fig, ax = plt.subplots(figsize=(10, 6))
    x, y = ecdf(train_vals)
    ax.plot(x, y, label="Train catalog (items)", color="black", linewidth=2.2, linestyle="--")

    colors = sns.color_palette("husl", n_colors=max(6, len(model_recs)))
    for i, (name, recs) in enumerate(sorted(model_recs.items())):
        vals = []
        for items in recs.values():
            for it in items:
                vals.append(np.log10(float(train_pop.get(it, 0.0)) + 1.0))
        if not vals:
            continue
        x, y = ecdf(np.array(vals))
        ax.plot(x, y, label=name, color=colors[i % len(colors)], alpha=0.9, linewidth=1.6)

    ax.set_xlabel("log10(train interactions + 1)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Popularity distribution: train vs recommendations ({dataset_name})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def tfidf_item_similarity(item_features: pd.DataFrame):
    tfidf = TfidfVectorizer(min_df=1, max_features=30000)
    mat = tfidf.fit_transform(item_features["metadata_text"])
    sim = cosine_similarity(mat, dense_output=False)
    return sim


def recommend_tfidf(train_df, users, all_items, item_index, sim_matrix, k: int = 10) -> dict:
    user_hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    recs = {}
    for u in tqdm(users, desc="TF-IDF Top-K"):
        seen = set(user_hist.get(u, []))
        seen_idx = [item_index[it] for it in seen if it in item_index]
        if not seen_idx:
            recs[u] = all_items[:k]
            continue
        profile_scores = sim_matrix[seen_idx].mean(axis=0).A1
        cand = []
        for it in all_items:
            if it in seen:
                continue
            idx = item_index.get(it)
            if idx is not None:
                cand.append((it, profile_scores[idx]))
        cand.sort(key=lambda x: x[1], reverse=True)
        recs[u] = [it for it, _ in cand[:k]]
    return recs


def train_user_logistic_models(train_df, item_features, user_min_rows: int = 5) -> dict:
    merged = train_df.merge(item_features, on="item_id", how="left")
    merged["label"] = (merged["rating"] >= merged["rating"].median()).astype(int)
    models = {}
    for user_id, grp in tqdm(merged.groupby("user_id"), desc="LogReg per-user"):
        if grp["label"].nunique() < 2 or len(grp) < user_min_rows:
            continue
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=1, max_features=5000)),
                ("clf", LogisticRegression(max_iter=300)),
            ]
        )
        pipe.fit(grp["metadata_text"].fillna(""), grp["label"])
        models[user_id] = pipe
    return models


def recommend_user_logreg(models, users, user_seen, item_features, all_items, k: int = 10) -> dict:
    item_text = dict(zip(item_features["item_id"], item_features["metadata_text"]))
    recs = {}
    for u in tqdm(users, desc="LogReg Top-K"):
        seen = user_seen.get(u, set())
        candidates = [it for it in all_items if it not in seen]
        model = models.get(u)
        if model is None:
            recs[u] = candidates[:k]
            continue
        X = [item_text.get(it, "") for it in candidates]
        if not X:
            recs[u] = []
            continue
        probs = model.predict_proba(X)[:, 1]
        order = np.argsort(-probs)[:k]
        recs[u] = [candidates[i] for i in order]
    return recs


def train_rf_regressor(train_df):
    df = train_df.copy()
    ue = LabelEncoder()
    ie = LabelEncoder()
    df["u"] = ue.fit_transform(df["user_id"])
    df["i"] = ie.fit_transform(df["item_id"])
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        # Joblib overhead can dominate for small batches; keep stable.
        n_jobs=1,
    )
    rf.fit(df[["u", "i"]], df["rating"])
    return rf, ue, ie


def recommend_rf(rf, ue, ie, users, user_seen, all_items, k: int = 10) -> dict:
    item_to_enc = {it: i for i, it in enumerate(ie.classes_)}
    user_set = set(ue.classes_)
    recs = {}
    for u in tqdm(users, desc="RF Top-K"):
        seen = user_seen.get(u, set())
        candidates = [it for it in all_items if it not in seen and it in item_to_enc]
        if not candidates or u not in user_set:
            recs[u] = all_items[:k]
            continue
        u_enc = ue.transform([u])[0]
        X = pd.DataFrame({"u": [u_enc] * len(candidates), "i": [item_to_enc[it] for it in candidates]})
        preds = rf.predict(X)
        order = np.argsort(-preds)[:k]
        recs[u] = [candidates[i] for i in order]
    return recs


def run_macro_bias(
    dataset_name: str,
    data_root: Path,
    fig_dir: Path,
    res_dir: Path,
    test_size: float,
    *,
    pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    item_features_override: Optional[pd.DataFrame] = None,
):
    print("[1/3] Running macro-bias experiment...")
    if pre_split is not None:
        train_df, test_df = pre_split
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if item_features_override is None:
            raise ValueError("item_features_override is required when pre_split is set")
        item_features = item_features_override.copy()
    else:
        ratings, item_features = load_dataset(dataset_name=dataset_name, data_root=data_root)
        train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    all_items = sorted(train_df["item_id"].unique().tolist())
    users_test = sorted(test_df["user_id"].unique().tolist())
    user_seen = build_user_seen(train_df)
    train_pop = popularity_series(train_df)
    pop_map = train_pop.to_dict()

    model_recs = {}
    model_rmse = {}
    trainset = surprise_prepare(train_df)

    knn = (
        KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False)
        if SURPRISE_AVAILABLE
        else _FallbackCF()
    )
    knn.fit(trainset)
    model_recs["k-NN"] = recommend_with_surprise(knn, users_test, user_seen, all_items, k=10)
    model_rmse["k-NN"] = rmse_surprise(knn, test_df)
    del knn
    gc.collect()

    svd = SVD(random_state=42) if SURPRISE_AVAILABLE else _FallbackCF()
    svd.fit(trainset)
    model_recs["SVD"] = recommend_with_surprise(svd, users_test, user_seen, all_items, k=10)
    model_rmse["SVD"] = rmse_surprise(svd, test_df)
    del svd
    gc.collect()

    try:
        from scipy.sparse import coo_matrix
        from implicit.als import AlternatingLeastSquares

        train_users = sorted(train_df["user_id"].unique().tolist())
        item_to_idx = {it: j for j, it in enumerate(all_items)}

        # Use categorical codes to guarantee contiguous indices.
        user_codes = pd.Categorical(train_df["user_id"], categories=train_users).codes.astype(np.int64)
        item_codes = pd.Categorical(train_df["item_id"], categories=all_items).codes.astype(np.int64)
        if (user_codes < 0).any() or (item_codes < 0).any():
            raise ValueError("ALS indexing failed: found unknown user_id/item_id codes during matrix build.")

        vals = train_df["rating"].astype(float).values
        # implicit 0.7.x expects user×item matrix for fit() and recommend()
        user_item = coo_matrix(
            (vals, (user_codes, item_codes)),
            shape=(len(train_users), len(all_items)),
        ).tocsr()

        als = AlternatingLeastSquares(
            factors=50, iterations=15, regularization=0.01, random_state=42
        )
        als.fit(user_item)
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        user_to_idx = {u: i for i, u in enumerate(train_users)}
        recs_als = {}
        for u in tqdm(users_test, desc="ALS Top-K"):
            uid = user_to_idx.get(u)
            if uid is None:
                recs_als[u] = all_items[:10]
                continue
            # Pass a single-row matrix; userid must be 0 for this batch.
            u_items = user_item[uid]
            rec_ids, _ = als.recommend(
                userid=0, user_items=u_items, N=10, filter_already_liked_items=True
            )
            recs_als[u] = [idx_to_item[i] for i in rec_ids if i in idx_to_item][:10]
        model_recs["ALS"] = recs_als
        model_rmse["ALS"] = np.nan
        del als, user_item
        gc.collect()
    except Exception as e:
        import traceback
        try:
            dbg = {
                "n_items": len(all_items),
                "n_users": int(train_df["user_id"].nunique()),
                "user_codes_min": int(np.min(user_codes)) if "user_codes" in locals() else None,
                "user_codes_max": int(np.max(user_codes)) if "user_codes" in locals() else None,
                "item_codes_min": int(np.min(item_codes)) if "item_codes" in locals() else None,
                "item_codes_max": int(np.max(item_codes)) if "item_codes" in locals() else None,
                "user_item_shape": tuple(user_item.shape) if "user_item" in locals() else None,
                "user_item_shape": tuple(user_item.shape) if "user_item" in locals() else None,
                "user_item_indices_max": int(user_item.indices.max()) if "user_item" in locals() and user_item.nnz else None,
            }
            print(f"ALS skipped due to error: {e} | debug={dbg}")
            print(traceback.format_exc())
        except Exception:
            print(f"ALS skipped due to error: {e}")
        model_recs["ALS"] = {u: all_items[:10] for u in users_test}
        model_rmse["ALS"] = np.nan

    item_features_local = (
        item_features[item_features["item_id"].isin(all_items)].drop_duplicates("item_id").reset_index(drop=True)
    )
    item_index = {it: i for i, it in enumerate(item_features_local["item_id"].tolist())}
    sim = tfidf_item_similarity(item_features_local)
    model_recs["TF-IDF"] = recommend_tfidf(train_df, users_test, all_items, item_index, sim, k=10)
    model_rmse["TF-IDF"] = np.nan
    del sim
    gc.collect()

    logreg_models = train_user_logistic_models(train_df, item_features_local)
    model_recs["LogReg"] = recommend_user_logreg(
        logreg_models, users_test, user_seen, item_features_local, all_items, k=10
    )
    model_rmse["LogReg"] = np.nan
    del logreg_models
    gc.collect()

    rf, ue, ie = train_rf_regressor(train_df)
    model_recs["RandomForest"] = recommend_rf(rf, ue, ie, users_test, user_seen, all_items, k=10)
    model_rmse["RandomForest"] = np.nan
    del rf, ue, ie
    gc.collect()

    pct_map = train_popularity_percentile_map(train_df)

    rows_out = []
    plot_rows = []
    for model_name, recs in model_recs.items():
        rec_freq = recommendation_frequency(recs)
        all_freq = train_pop.copy().rename("train_pop").to_frame()
        all_freq["rec_freq"] = rec_freq
        all_freq["rec_freq"] = all_freq["rec_freq"].fillna(0.0)
        gini = gini_coefficient(all_freq["rec_freq"].values)
        arp = compute_arp(recs, pop_map)
        coverage = compute_catalog_coverage(recs, all_items)
        rho, _ = spearmanr(all_freq["train_pop"].values, all_freq["rec_freq"].values)
        app = avg_popularity_percentile(recs, pct_map)
        lt = long_tail_share(recs, pct_map, tail_cutoff=0.2)
        agg_div = aggregate_diversity_unique_items(recs)
        meta_h = metadata_token_entropy(recs, item_features_local)
        rows_out.append(
            {
                "model": model_name,
                "gini": gini,
                "arp": arp,
                "catalog_coverage": coverage,
                "spearman_pop_vs_recfreq": rho,
                "avg_popularity_percentile": app,
                "long_tail_share_20pct": lt,
                "aggregate_diversity_unique_items": agg_div,
                "metadata_token_entropy_bits": meta_h,
                "rmse": model_rmse.get(model_name, np.nan),
            }
        )
        tmp = all_freq.reset_index().rename(columns={"index": "item_id"})
        tmp["model"] = model_name
        plot_rows.append(tmp)

    metrics_df = pd.DataFrame(rows_out).sort_values("model")
    metrics_df.to_csv(res_dir / f"macro_bias_metrics_{dataset_name}.csv", index=False)

    plot_df = pd.concat(plot_rows, ignore_index=True)
    plot_df = plot_df[(plot_df["train_pop"] > 0) & (plot_df["rec_freq"] > 0)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, model_name in zip(axes, sorted(model_recs.keys())):
        d = plot_df[plot_df["model"] == model_name]
        ax.scatter(d["train_pop"], d["rec_freq"], alpha=0.35, s=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model_name)
        ax.set_xlabel("Global Popularity (train count)")
        ax.set_ylabel("Recommendation Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / f"macro_bias_loglog_{dataset_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ext_cols = [
        "avg_popularity_percentile",
        "long_tail_share_20pct",
        "aggregate_diversity_unique_items",
        "metadata_token_entropy_bits",
    ]
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    for ax, col in zip(axes2, ext_cols):
        dfp = metrics_df.sort_values("model")
        sns.barplot(data=dfp, x="model", y=col, ax=ax)
        ax.set_title(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle(f"Extended macro-bias metrics ({dataset_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / f"macro_bias_extended_metrics_{dataset_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    plot_popularity_cdf_compare(
        train_pop,
        model_recs,
        fig_dir / f"macro_bias_popularity_cdf_{dataset_name}.png",
        dataset_name,
    )
    gc.collect()


def run_user_centric(
    dataset_name: str,
    data_root: Path,
    fig_dir: Path,
    res_dir: Path,
    test_size: float,
    *,
    pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    item_features_override: Optional[pd.DataFrame] = None,
):
    print("[2/3] Running user-centric fairness experiment...")
    if pre_split is not None:
        train_df, test_df = pre_split
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if item_features_override is None:
            raise ValueError("item_features_override is required when pre_split is set")
        item_features = item_features_override.copy()
    else:
        ratings, item_features = load_dataset(dataset_name=dataset_name, data_root=data_root)
        train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    all_items = sorted(train_df["item_id"].unique().tolist())
    users_test = sorted(test_df["user_id"].unique().tolist())
    user_seen = build_user_seen(train_df)

    global_pop = train_df["item_id"].value_counts().to_dict()
    user_rows = []
    for user_id, grp in train_df.groupby("user_id"):
        vals = [global_pop.get(it, 0) for it in grp["item_id"].tolist()]
        user_rows.append({"user_id": user_id, "Mainstreaminess_Score": float(np.mean(vals)) if vals else 0.0})
    user_mainstream_df = pd.DataFrame(user_rows)
    q1 = user_mainstream_df["Mainstreaminess_Score"].quantile(0.33)
    q2 = user_mainstream_df["Mainstreaminess_Score"].quantile(0.67)

    def segment_user(s):
        if s <= q1:
            return "Niche"
        if s >= q2:
            return "Mainstream"
        return "Diverse"

    user_mainstream_df["segment"] = user_mainstream_df["Mainstreaminess_Score"].apply(segment_user)
    segment_map = dict(zip(user_mainstream_df["user_id"], user_mainstream_df["segment"]))

    model_preds = {}
    model_recs = {}
    trainset = surprise_prepare(train_df)
    item_mean = float(train_df["rating"].mean())

    knn = (
        KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False)
        if SURPRISE_AVAILABLE
        else _FallbackCF()
    )
    knn.fit(trainset)
    knn_preds = []
    for r in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="k-NN preds"):
        knn_preds.append((r.user_id, r.item_id, r.rating, est_predict(knn, r.user_id, r.item_id)))
    model_preds["k-NN"] = knn_preds
    model_recs["k-NN"] = recommend_with_surprise(knn, users_test, user_seen, all_items, k=10)
    del knn
    gc.collect()

    try:
        svd = SVD(random_state=42) if SURPRISE_AVAILABLE else _FallbackCF()
        svd.fit(trainset)
        svd_preds = []
        for r in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="SVD preds"):
            svd_preds.append((r.user_id, r.item_id, r.rating, est_predict(svd, r.user_id, r.item_id)))
        model_preds["SVD"] = svd_preds
        model_recs["SVD"] = recommend_with_surprise(svd, users_test, user_seen, all_items, k=10)
        del svd
        gc.collect()
    except Exception as e:
        print(f"SVD failed on sparse data: {e}")
        model_preds["SVD"] = [(r.user_id, r.item_id, r.rating, item_mean) for r in test_df.itertuples(index=False)]
        model_recs["SVD"] = {u: all_items[:10] for u in users_test}

    try:
        from scipy.sparse import coo_matrix
        from implicit.als import AlternatingLeastSquares

        train_users = sorted(train_df["user_id"].unique().tolist())
        item_to_idx = {it: j for j, it in enumerate(all_items)}

        user_codes = pd.Categorical(train_df["user_id"], categories=train_users).codes.astype(np.int64)
        item_codes = pd.Categorical(train_df["item_id"], categories=all_items).codes.astype(np.int64)
        if (user_codes < 0).any() or (item_codes < 0).any():
            raise ValueError("ALS indexing failed: found unknown user_id/item_id codes during matrix build.")

        vals = train_df["rating"].astype(float).values
        user_item = coo_matrix(
            (vals, (user_codes, item_codes)),
            shape=(len(train_users), len(all_items)),
        ).tocsr()
        als = AlternatingLeastSquares(factors=50, iterations=15, regularization=0.01, random_state=42)
        als.fit(user_item)
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        user_to_idx = {u: i for i, u in enumerate(train_users)}
        recs_als = {}
        for u in tqdm(users_test, desc="ALS recs"):
            uid = user_to_idx.get(u)
            if uid is None:
                recs_als[u] = all_items[:10]
                continue
            u_items = user_item[uid]
            rec_ids, _ = als.recommend(
                userid=0, user_items=u_items, N=10, filter_already_liked_items=True
            )
            recs_als[u] = [idx_to_item[i] for i in rec_ids if i in idx_to_item][:10]
        model_recs["ALS"] = recs_als
        model_preds["ALS"] = [(r.user_id, r.item_id, r.rating, item_mean) for r in test_df.itertuples(index=False)]
        del als, user_item
        gc.collect()
    except Exception as e:
        import traceback
        try:
            dbg = {
                "n_items": len(all_items),
                "n_users": int(train_df["user_id"].nunique()),
                "user_codes_min": int(np.min(user_codes)) if "user_codes" in locals() else None,
                "user_codes_max": int(np.max(user_codes)) if "user_codes" in locals() else None,
                "item_codes_min": int(np.min(item_codes)) if "item_codes" in locals() else None,
                "item_codes_max": int(np.max(item_codes)) if "item_codes" in locals() else None,
                "user_item_shape": tuple(user_item.shape) if "user_item" in locals() else None,
                "user_item_shape": tuple(user_item.shape) if "user_item" in locals() else None,
                "user_item_indices_max": int(user_item.indices.max()) if "user_item" in locals() and user_item.nnz else None,
            }
            print(f"ALS skipped: {e} | debug={dbg}")
            print(traceback.format_exc())
        except Exception:
            print(f"ALS skipped: {e}")
        model_recs["ALS"] = {u: all_items[:10] for u in users_test}
        model_preds["ALS"] = [(r.user_id, r.item_id, r.rating, item_mean) for r in test_df.itertuples(index=False)]

    item_features_local = (
        item_features[item_features["item_id"].isin(all_items)].drop_duplicates("item_id").reset_index(drop=True)
    )
    item_index = {it: i for i, it in enumerate(item_features_local["item_id"].tolist())}
    sim = tfidf_item_similarity(item_features_local)
    model_recs["TF-IDF"] = recommend_tfidf(train_df, users_test, all_items, item_index, sim, k=10)
    model_preds["TF-IDF"] = [(r.user_id, r.item_id, r.rating, item_mean) for r in test_df.itertuples(index=False)]
    del sim
    gc.collect()

    logreg_models = train_user_logistic_models(train_df, item_features_local)
    model_recs["LogReg"] = recommend_user_logreg(
        logreg_models, users_test, user_seen, item_features_local, all_items, k=10
    )
    model_preds["LogReg"] = [(r.user_id, r.item_id, r.rating, item_mean) for r in test_df.itertuples(index=False)]
    del logreg_models
    gc.collect()

    rf, ue, ie = train_rf_regressor(train_df)
    model_recs["RandomForest"] = recommend_rf(rf, ue, ie, users_test, user_seen, all_items, k=10)
    item_to_enc = {it: i for i, it in enumerate(ie.classes_)}
    user_set = set(ue.classes_)
    # Vectorized RF predictions (much faster than per-row predict calls).
    df_t = test_df[["user_id", "item_id", "rating"]].copy()
    ok_mask = df_t["user_id"].isin(user_set) & df_t["item_id"].isin(item_to_enc.keys())
    df_t["pred"] = item_mean
    if ok_mask.any():
        u_enc = ue.transform(df_t.loc[ok_mask, "user_id"].astype(str))
        i_enc = df_t.loc[ok_mask, "item_id"].map(item_to_enc).astype(int).values
        X = pd.DataFrame({"u": u_enc, "i": i_enc})
        preds = rf.predict(X)
        df_t.loc[ok_mask, "pred"] = preds

    rf_preds = list(df_t.itertuples(index=False, name=None))
    # tuples: (user_id, item_id, rating, pred)
    model_preds["RandomForest"] = rf_preds
    del rf, ue, ie
    gc.collect()

    def segment_rmse(pred_rows):
        df = pd.DataFrame(pred_rows, columns=["user_id", "item_id", "true", "pred"])
        out = {}
        for seg in ["Niche", "Diverse", "Mainstream"]:
            d = df[df["user_id"].map(segment_map) == seg]
            out[seg] = float(np.sqrt(mean_squared_error(d["true"], d["pred"]))) if len(d) else np.nan
        return out

    rel_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    def segment_ndcg(recs):
        out = {}
        for seg in ["Niche", "Diverse", "Mainstream"]:
            users_seg = [u for u, s in segment_map.items() if s == seg and u in recs]
            vals = [ndcg_at_k(recs[u], rel_by_user.get(u, set()), k=10) for u in users_seg]
            out[seg] = float(np.mean(vals)) if vals else np.nan
        return out

    rows_out = []
    for model_name in model_recs.keys():
        rmses = segment_rmse(model_preds.get(model_name, []))
        ndcgs = segment_ndcg(model_recs[model_name])
        preds = model_preds.get(model_name, [])
        niche_rmse, main_rmse = per_user_rmse_by_segment(preds, segment_map)
        niche_ndcg, main_ndcg = per_user_ndcg_by_segment(
            model_recs[model_name],
            segment_map,
            rel_by_user,
            ndcg_at_k,
            k=10,
        )
        p_rmse_mwu = mann_whitney_pvalue(niche_rmse, main_rmse, alternative="two-sided")
        p_rmse_welch = welch_ttest_pvalue(niche_rmse, main_rmse)
        p_rmse_niche_greater = mann_whitney_pvalue(niche_rmse, main_rmse, alternative="greater")
        p_ndcg_mwu = mann_whitney_pvalue(niche_ndcg, main_ndcg, alternative="two-sided")
        p_ndcg_welch = welch_ttest_pvalue(niche_ndcg, main_ndcg)
        p_ndcg_niche_less = mann_whitney_pvalue(niche_ndcg, main_ndcg, alternative="less")
        rows_out.append(
            {
                "model": model_name,
                "rmse_niche": rmses.get("Niche", np.nan),
                "rmse_diverse": rmses.get("Diverse", np.nan),
                "rmse_mainstream": rmses.get("Mainstream", np.nan),
                "ndcg_niche": ndcgs.get("Niche", np.nan),
                "ndcg_diverse": ndcgs.get("Diverse", np.nan),
                "ndcg_mainstream": ndcgs.get("Mainstream", np.nan),
                "delta_accuracy_rmse_niche_minus_mainstream": rmses.get("Niche", np.nan)
                - rmses.get("Mainstream", np.nan),
                "n_users_niche_rmse": int(len(niche_rmse)),
                "n_users_mainstream_rmse": int(len(main_rmse)),
                "p_mwu_rmse_niche_vs_mainstream": p_rmse_mwu,
                "p_welch_rmse_niche_vs_mainstream": p_rmse_welch,
                "p_mwu_rmse_niche_greater_mainstream": p_rmse_niche_greater,
                "sig_rmse_mwu_005": significance_flag(p_rmse_mwu),
                "n_users_niche_ndcg": int(len(niche_ndcg)),
                "n_users_mainstream_ndcg": int(len(main_ndcg)),
                "p_mwu_ndcg_niche_vs_mainstream": p_ndcg_mwu,
                "p_welch_ndcg_niche_vs_mainstream": p_ndcg_welch,
                "p_mwu_ndcg_niche_less_mainstream": p_ndcg_niche_less,
                "sig_ndcg_mwu_005": significance_flag(p_ndcg_mwu),
            }
        )
    fairness_df = pd.DataFrame(rows_out).sort_values("model")
    fairness_df.to_csv(res_dir / f"user_centric_metrics_{dataset_name}.csv", index=False)

    plot_df = fairness_df.melt(
        id_vars=["model"],
        value_vars=["rmse_niche", "rmse_diverse", "rmse_mainstream"],
        var_name="segment",
        value_name="rmse",
    )
    plot_df["segment"] = plot_df["segment"].str.replace("rmse_", "", regex=False).str.title()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="rmse", hue="segment")
    plt.title(f"RMSE by User Segment ({dataset_name})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_dir / f"user_centric_rmse_segment_{dataset_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def run_feedback_simulation(
    dataset_name: str,
    data_root: Path,
    fig_dir: Path,
    res_dir: Path,
    *,
    ratings_override: Optional[pd.DataFrame] = None,
    item_features_override: Optional[pd.DataFrame] = None,
):
    print("[3/3] Running feedback-loop simulation...")
    if ratings_override is not None:
        if item_features_override is None:
            raise ValueError("item_features_override is required when ratings_override is set")
        ratings = ratings_override.reset_index(drop=True)
        item_features = item_features_override.copy()
    else:
        ratings, item_features = load_dataset(dataset_name=dataset_name, data_root=data_root)
    base_train, _ = train_test_split(ratings, test_size=0.2, random_state=42)
    base_train = base_train.reset_index(drop=True)
    rng = np.random.default_rng(42)
    users = sorted(base_train["user_id"].unique().tolist())

    def aggregate_diversity(recs):
        return len(set([it for items in recs.values() for it in items]))

    def simulate_clicks(recs, rel_scores, pop_map, rel_weight=0.7, pop_weight=0.3):
        rows = []
        pop_vals = np.array(list(pop_map.values()), dtype=float)
        p_min = float(pop_vals.min()) if len(pop_vals) else 0.0
        p_max = float(pop_vals.max()) if len(pop_vals) else 1.0
        for u, items in recs.items():
            for it in items:
                rel = rel_scores.get((u, it), 0.5)
                if p_max > p_min:
                    p_norm = (pop_map.get(it, p_min) - p_min) / (p_max - p_min)
                else:
                    p_norm = 0.0
                click_prob = np.clip(rel_weight * rel + pop_weight * p_norm, 0.0, 1.0)
                if rng.binomial(1, click_prob) == 1:
                    rows.append((u, it, 5.0))
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])

    def train_svd_recommend(train_df, top_k=5):
        trainset = surprise_prepare(train_df)
        svd = SVD(random_state=42) if SURPRISE_AVAILABLE else _FallbackCF()
        svd.fit(trainset)
        seen = build_user_seen(train_df)
        all_items = sorted(train_df["item_id"].unique().tolist())
        recs = {}
        rel_scores = {}
        for u in tqdm(users, desc="SVD recs"):
            candidates = [i for i in all_items if i not in seen.get(u, set())]
            scored = [(it, est_predict(svd, u, it)) for it in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_k]
            recs[u] = [it for it, _ in top]
            rmin = float(train_df["rating"].min())
            rmax = float(train_df["rating"].max())
            denom = (rmax - rmin) + 1e-9
            for it, score in top:
                rel_scores[(u, it)] = float((score - rmin) / denom)
        del svd
        gc.collect()
        return recs, rel_scores

    def train_tfidf_recommend(train_df, top_k=5):
        all_items = sorted(train_df["item_id"].unique().tolist())
        item_features_local = (
            item_features[item_features["item_id"].isin(all_items)].drop_duplicates("item_id").reset_index(drop=True)
        )
        tfidf = TfidfVectorizer(min_df=1, max_features=30000)
        mat = tfidf.fit_transform(item_features_local["metadata_text"])
        sim = cosine_similarity(mat, dense_output=False)
        item_idx = {it: i for i, it in enumerate(item_features_local["item_id"])}
        hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
        recs = {}
        rel_scores = {}
        for u in tqdm(users, desc="TF-IDF recs"):
            seen = set(hist.get(u, []))
            seen_idx = [item_idx[i] for i in seen if i in item_idx]
            if not seen_idx:
                recs[u] = all_items[:top_k]
                for it in recs[u]:
                    rel_scores[(u, it)] = 0.5
                continue
            prof = sim[seen_idx].mean(axis=0).A1
            cand = []
            for it in all_items:
                if it in seen or it not in item_idx:
                    continue
                cand.append((it, prof[item_idx[it]]))
            cand.sort(key=lambda x: x[1], reverse=True)
            top = cand[:top_k]
            recs[u] = [it for it, _ in top]
            for it, score in top:
                rel_scores[(u, it)] = float(score)
        del sim, mat, tfidf
        gc.collect()
        return recs, rel_scores

    svd_train = base_train.copy()
    tfidf_train = base_train.copy()
    records = []

    for t in range(1, 6):
        print(f"  Iteration {t}/5")

        pop_svd = svd_train["item_id"].value_counts().to_dict()
        recs_svd, rel_svd = train_svd_recommend(svd_train, top_k=5)
        clicks_svd = simulate_clicks(recs_svd, rel_svd, pop_svd, rel_weight=0.7, pop_weight=0.3)
        records.append(
            {
                "iteration": t,
                "model": "SVD",
                "aggregate_diversity": aggregate_diversity(recs_svd),
                "arp": compute_arp(recs_svd, pop_svd),
            }
        )
        if len(clicks_svd):
            svd_train = pd.concat([svd_train, clicks_svd], ignore_index=True)
            svd_train = svd_train.drop_duplicates(subset=["user_id", "item_id"], keep="last")
        del recs_svd, rel_svd, clicks_svd
        gc.collect()

        pop_t = tfidf_train["item_id"].value_counts().to_dict()
        recs_t, rel_t = train_tfidf_recommend(tfidf_train, top_k=5)
        clicks_t = simulate_clicks(recs_t, rel_t, pop_t, rel_weight=0.7, pop_weight=0.3)
        records.append(
            {
                "iteration": t,
                "model": "TF-IDF",
                "aggregate_diversity": aggregate_diversity(recs_t),
                "arp": compute_arp(recs_t, pop_t),
            }
        )
        if len(clicks_t):
            tfidf_train = pd.concat([tfidf_train, clicks_t], ignore_index=True)
            tfidf_train = tfidf_train.drop_duplicates(subset=["user_id", "item_id"], keep="last")
        del recs_t, rel_t, clicks_t
        gc.collect()

    sim_df = pd.DataFrame(records)
    sim_df.to_csv(res_dir / f"feedback_simulation_metrics_{dataset_name}.csv", index=False)
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=sim_df, x="iteration", y="aggregate_diversity", hue="model", marker="o")
    plt.title(f"Diversity Decay over Time ({dataset_name})")
    plt.ylabel("Aggregate Diversity")
    plt.xlabel("Timestep")
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(fig_dir / f"feedback_diversity_decay_{dataset_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Run all recommender-system bias experiments.")
    parser.add_argument(
        "--dataset",
        default="ml-100k",
        choices=["ml-100k", "ml-1m", "lastfm-2k", "book-crossing"],
        help="Dataset name under data/",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default 0.2)")
    args = parser.parse_args()

    out = ensure_output_dirs(ROOT)
    fig_dir = out["figures"]
    res_dir = out["results"]
    data_root = ROOT / "data"

    wall_start = datetime.now()
    t0 = time.perf_counter()
    n_phases = 3

    print(f"Dataset: {args.dataset}")
    print("Outputs:")
    print(f"  Figures: {fig_dir}")
    print(f"  Results: {res_dir}")
    print(f"Run started: {wall_start.isoformat(timespec='seconds')}")
    print(f">>> OVERALL PROGRESS: {0.0:.1f}% | initializing three experiments (macro → user-centric → feedback)")

    t_phase = t0
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: {100.0 * 0 / n_phases:.1f}% | [1/3] Macro-bias quantification\n"
        f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
        f"{'=' * 72}"
    )
    run_macro_bias(args.dataset, data_root, fig_dir, res_dir, args.test_size)
    t1 = time.perf_counter()
    print(
        f">>> OVERALL PROGRESS: {100.0 * 1 / n_phases:.1f}% | [1/3] Macro-bias DONE\n"
        f"    Phase duration: {_format_duration(t1 - t_phase)} | Total elapsed: {_format_duration(t1 - t0)}"
    )

    t_phase = t1
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: {100.0 * 1 / n_phases:.1f}% | [2/3] User-centric fairness\n"
        f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
        f"{'=' * 72}"
    )
    run_user_centric(args.dataset, data_root, fig_dir, res_dir, args.test_size)
    t2 = time.perf_counter()
    print(
        f">>> OVERALL PROGRESS: {100.0 * 2 / n_phases:.1f}% | [2/3] User-centric DONE\n"
        f"    Phase duration: {_format_duration(t2 - t_phase)} | Total elapsed: {_format_duration(t2 - t0)}"
    )

    t_phase = t2
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: {100.0 * 2 / n_phases:.1f}% | [3/3] Feedback-loop simulation\n"
        f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
        f"{'=' * 72}"
    )
    run_feedback_simulation(args.dataset, data_root, fig_dir, res_dir)
    t3 = time.perf_counter()
    print(
        f">>> OVERALL PROGRESS: {100.0 * 3 / n_phases:.1f}% | [3/3] Feedback simulation DONE\n"
        f"    Phase duration: {_format_duration(t3 - t_phase)} | Total elapsed: {_format_duration(t3 - t0)}"
    )

    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: 100.0% | ALL EXPERIMENTS COMPLETE\n"
        f"    Total runtime: {_format_duration(t3 - t0)} (wall start {wall_start.isoformat(timespec='seconds')})\n"
        f"{'=' * 72}"
    )


if __name__ == "__main__":
    main()

