"""
Statistical tests for user-centric fairness metrics.

Segment summaries in CSVs are means over users. To test whether Niche vs Mainstream
differ beyond a single train/test split noise, we use per-user observations:
  - per-user RMSE from test-row predictions
  - per-user NDCG@k from recommendations

Comparisons use two independent samples (Niche users vs Mainstream users), not paired
tests. Mann–Whitney U is the default (non-parametric); Welch's t-test is included
for approximately normal per-user metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


def per_user_rmse_by_segment(
    pred_rows: List[Tuple[Any, ...]],
    segment_map: Dict[Any, str],
    segments: Tuple[str, str] = ("Niche", "Mainstream"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-user RMSE arrays for Niche and Mainstream test users."""
    if not pred_rows:
        return np.array([]), np.array([])
    df = pd.DataFrame(pred_rows, columns=["user_id", "item_id", "true", "pred"])
    df["true"] = df["true"].astype(float)
    df["pred"] = df["pred"].astype(float)
    df["sq_err"] = (df["true"] - df["pred"]) ** 2
    niche: List[float] = []
    mainstream: List[float] = []
    for uid, g in df.groupby("user_id"):
        seg = segment_map.get(uid)
        if seg == segments[0]:
            niche.append(float(np.sqrt(np.mean(g["sq_err"].values))))
        elif seg == segments[1]:
            mainstream.append(float(np.sqrt(np.mean(g["sq_err"].values))))
    return np.asarray(niche, dtype=float), np.asarray(mainstream, dtype=float)


def per_user_ndcg_by_segment(
    recs: Dict[Any, List[Any]],
    segment_map: Dict[Any, str],
    relevant_by_user: Dict[Any, set],
    ndcg_fn,
    segments: Tuple[str, str] = ("Niche", "Mainstream"),
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-user NDCG@k arrays for Niche and Mainstream users with recommendations."""
    niche: List[float] = []
    mainstream: List[float] = []
    for u, items in recs.items():
        seg = segment_map.get(u)
        if seg == segments[0]:
            niche.append(float(ndcg_fn(items, relevant_by_user.get(u, set()), k=k)))
        elif seg == segments[1]:
            mainstream.append(float(ndcg_fn(items, relevant_by_user.get(u, set()), k=k)))
    return np.asarray(niche, dtype=float), np.asarray(mainstream, dtype=float)


def _clean(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return a, b


def mann_whitney_pvalue(
    niche: np.ndarray,
    mainstream: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> float:
    """Mann–Whitney U p-value; NaN if either group has fewer than 2 finite values."""
    a, b = _clean(niche, mainstream)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative=alternative).pvalue)


def welch_ttest_pvalue(niche: np.ndarray, mainstream: np.ndarray) -> float:
    """Welch's t-test (unequal variance) on per-user metrics; NaN if too few samples."""
    a, b = _clean(niche, mainstream)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def significance_flag(p: float, alpha: float = 0.05) -> bool:
    return bool(np.isfinite(p) and p < alpha)
