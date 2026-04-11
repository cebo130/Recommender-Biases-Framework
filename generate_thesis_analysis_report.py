#!/usr/bin/env python3
"""
Build a Markdown report for the thesis from completed experiment CSVs under a run's ``results/``:

- Rerank **sensitivity** (``svd_rerank_sensitivity_*.csv``): λ × pool_k vs Gini, long-tail share, Spearman.
- **Same-pool baselines** vs **SVD+Rerank** (``macro_bias_metrics_*.csv``).
- **MovieLens 100K vs 1M** side-by-side on selected models (scale / robustness).
- **Niche vs Mainstream** Mann–Whitney **p-values** (``user_centric_metrics_*.csv``).

Usage::

  python generate_thesis_analysis_report.py --latest
  python generate_thesis_analysis_report.py --run-root More_Outputs/runs/full_thesis_batch
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from utils import default_output_parent  # noqa: E402

POOL_BASELINE_MODELS = [
    "SVD",
    "SVD+Rerank",
    "SVD+PoolPopDesc",
    "SVD+PoolPopAsc",
    "SVD+PoolMMR",
]
SCALE_COMPARE_MODELS = ["SVD", "SVD+Rerank", "k-NN", "TF-IDF", "ALS"]
SENSITIVITY_METRICS = [
    "gini",
    "long_tail_share_20pct",
    "spearman_pop_vs_recfreq",
    "avg_popularity_percentile",
    "catalog_coverage",
]
MACRO_COLS_FOR_TABLE = [
    "model",
    "gini",
    "long_tail_share_20pct",
    "spearman_pop_vs_recfreq",
    "avg_popularity_percentile",
    "catalog_coverage",
]


def _find_latest_results_dir() -> Path | None:
    runs = ROOT / default_output_parent() / "runs"
    if not runs.is_dir():
        return None
    candidates = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        r = d / "results"
        if r.is_dir():
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "_No rows._\n"
    disp = df.copy()
    for c in disp.columns:
        if pd.api.types.is_float_dtype(disp[c]):
            disp[c] = disp[c].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")
    cols = [str(c) for c in disp.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in disp.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in disp.columns) + " |")
    return "\n".join([header, sep] + body) + "\n"


def _sensitivity_section(results: Path, dataset_label: str, filename: str) -> str:
    path = results / filename
    df = _load_csv(path)
    if df is None:
        return f"### {dataset_label}\n\n_File not found: `{filename}`_ (run without `--no-svd-mitigation-extras` for rerank datasets).\n\n"
    cols = [c for c in ["pool_k", "rerank_lambda", "model"] + SENSITIVITY_METRICS if c in df.columns]
    sub = df[cols].sort_values([c for c in ["pool_k", "rerank_lambda"] if c in df.columns])
    lines = [f"### {dataset_label}\n"]
    lines.append(
        "Popularity-penalty rerank only: **higher λ** weights predicted score more and train popularity less "
        "in the pool. **Larger pool_k** widens the candidate set before re-ranking.\n"
    )
    lines.append(_df_to_markdown(sub))
    if "pool_k" in df.columns and "rerank_lambda" in df.columns:
        lines.append("\n**Quick ranges (by pool_k):**\n")
        for pk in sorted(df["pool_k"].dropna().unique()):
            part = df[df["pool_k"] == pk]
            if part.empty:
                continue
            row_bits = []
            for m in ("long_tail_share_20pct", "gini", "spearman_pop_vs_recfreq"):
                if m in part.columns:
                    lo, hi = part[m].min(), part[m].max()
                    row_bits.append(f"`{m}` ∈ [{lo:.4f}, {hi:.4f}]")
            lines.append(f"- **pool_k = {int(pk)}**: " + "; ".join(row_bits) + "\n")
    lines.append("\n")
    return "".join(lines)


def _macro_baseline_section(results: Path, dataset_name: str) -> str:
    path = results / f"macro_bias_metrics_{dataset_name}.csv"
    df = _load_csv(path)
    title = dataset_name.replace("-", " ")
    if df is None:
        return f"### Macro baselines ({title})\n\n_File missing._\n\n"
    sub = df[df["model"].isin(POOL_BASELINE_MODELS)].copy()
    cols = [c for c in MACRO_COLS_FOR_TABLE if c in sub.columns]
    sub = sub[cols].sort_values("model")
    lines = [f"### Same-pool mitigation contrast ({title})\n"]
    lines.append(
        "All pool strategies use the **same SVD top-pool** (pool_k = 50 for baselines and primary rerank). "
        "**SVD+Rerank** uses the popularity-penalty score; **PoolPopDesc / PoolPopAsc** sort by train popularity; "
        "**PoolMMR** trades relevance for diversity in embedding space (TF–IDF cosine).\n\n"
    )
    lines.append(_df_to_markdown(sub))
    lines.append("\n")
    return "".join(lines)


def _scale_100k_vs_1m(results: Path) -> str:
    p100 = results / "macro_bias_metrics_ml-100k.csv"
    p1m = results / "macro_bias_metrics_ml-1m.csv"
    a = _load_csv(p100)
    b = _load_csv(p1m)
    lines = ["### MovieLens 100K vs 1M (scale check)\n"]
    if a is None or b is None:
        lines.append("_Need both `macro_bias_metrics_ml-100k.csv` and `macro_bias_metrics_ml-1m.csv`._\n\n")
        return "".join(lines)
    a = a[a["model"].isin(SCALE_COMPARE_MODELS)].set_index("model")
    b = b[b["model"].isin(SCALE_COMPARE_MODELS)].set_index("model")
    metrics = [c for c in MACRO_COLS_FOR_TABLE if c != "model" and c in a.columns and c in b.columns]
    rows = []
    for mname in SCALE_COMPARE_MODELS:
        if mname not in a.index or mname not in b.index:
            continue
        row = {"model": mname}
        for m in metrics:
            row[f"ml-100k_{m}"] = a.loc[mname, m]
            row[f"ml-1m_{m}"] = b.loc[mname, m]
        rows.append(row)
    if not rows:
        lines.append("_No overlapping models to compare._\n\n")
        return "".join(lines)
    cmp_df = pd.DataFrame(rows)
    lines.append(
        "Use this block for a **short robustness subsection**: same metrics, larger catalog and user base on 1M. "
        "Directions of bias (e.g. Spearman, long-tail share) should be interpreted comparatively, not as identical magnitudes.\n\n"
    )
    lines.append(_df_to_markdown(cmp_df))
    lines.append("\n")
    return "".join(lines)


def _significance_section(results: Path, dataset_name: str) -> str:
    path = results / f"user_centric_metrics_{dataset_name}.csv"
    df = _load_csv(path)
    title = dataset_name.replace("-", " ")
    if df is None:
        return f"### Segment tests ({title})\n\n_File missing._\n\n"
    focus = df[df["model"].isin(POOL_BASELINE_MODELS + ["k-NN", "ALS", "TF-IDF", "RandomForest"])].copy()
    cols = [
        "model",
        "rmse_niche",
        "rmse_mainstream",
        "p_mwu_rmse_niche_vs_mainstream",
        "sig_rmse_mwu_005",
        "ndcg_niche",
        "ndcg_mainstream",
        "p_mwu_ndcg_niche_vs_mainstream",
        "sig_ndcg_mwu_005",
    ]
    cols = [c for c in cols if c in focus.columns]
    focus = focus[cols].sort_values("model")
    lines = [f"### Niche vs Mainstream — tests ({title})\n"]
    lines.append(
        "**Mann–Whitney U** on per-user RMSE and NDCG@10 (Niche vs Mainstream users). "
        "`sig_rmse_mwu_005` / `sig_ndcg_mwu_005` flag p < 0.05 (two-sided) when present in your pipeline output.\n\n"
    )
    lines.append(_df_to_markdown(focus, float_fmt="{:.6g}"))
    p_ex = _p_sentence(df, "SVD+Rerank", "p_mwu_rmse_niche_vs_mainstream")
    lines.append("\n**One-sentence template for the thesis:**\n\n")
    lines.append(
        f"> For **{title}**, differences in per-user RMSE between Niche and Mainstream segments "
        "were assessed with a Mann–Whitney U test; e.g. **SVD+Rerank** yields "
        f"p = {p_ex} for RMSE (see table). Repeat for NDCG using `p_mwu_ndcg_niche_vs_mainstream`.\n\n"
    )
    return "".join(lines)


def _p_sentence(df: pd.DataFrame, model: str, col: str) -> str:
    r = df[df["model"] == model]
    if r.empty or col not in r.columns:
        return "N/A"
    v = r.iloc[0][col]
    if pd.isna(v):
        return "N/A"
    return f"{v:.4g}"


def build_report(results_dir: Path) -> str:
    results_dir = results_dir.resolve()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Thesis analysis — auto-generated from experiment CSVs\n\n",
        f"_Generated: {now}_  \n",
        f"_Results directory: `{results_dir}`_\n\n",
        "---\n\n",
        "## 1. Rerank sensitivity (λ × pool_k)\n\n",
        _sensitivity_section(results_dir, "MovieLens 100K", "svd_rerank_sensitivity_ml-100k.csv"),
        _sensitivity_section(results_dir, "Takealot", "svd_rerank_sensitivity_take-a-lot-dataset.csv"),
        "## 2. Same-pool baselines vs popularity-penalty rerank\n\n",
        _macro_baseline_section(results_dir, "ml-100k"),
        _macro_baseline_section(results_dir, "take-a-lot-dataset"),
        "## 3. Scale: 100K vs 1M\n\n",
        _scale_100k_vs_1m(results_dir),
        "## 4. Fairness segments — statistical tests\n\n",
        _significance_section(results_dir, "ml-100k"),
        _significance_section(results_dir, "ml-1m"),
        "---\n\n",
        "## 5. Takealot (domain note)\n\n",
        "Use **Takealot** results for **domain shift** and **catalog scale** discussion: "
        "full-catalog scoring is costly; interpret exposure metrics in light of sparsity and item cardinality. "
        "Macro and user-centric tables for `take-a-lot-dataset` are in the same `results/` folder.\n\n",
    ]
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate THESIS_ANALYSIS_REPORT.md from results CSVs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to .../results (contains macro_bias_metrics_*.csv).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run folder containing a results/ subdirectory.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=f"Use the newest results/ under {default_output_parent()}/runs/*/results.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path (default: <results-dir>/THESIS_ANALYSIS_REPORT.md).",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        res = args.results_dir.expanduser().resolve()
    elif args.run_root is not None:
        res = (args.run_root.expanduser().resolve() / "results").resolve()
    elif args.latest:
        found = _find_latest_results_dir()
        if found is None:
            print("No results found under More_Outputs/runs/*/results.", file=sys.stderr)
            sys.exit(1)
        res = found
    else:
        parser.error("Pass --results-dir, --run-root, or --latest")

    if not res.is_dir():
        print(f"Not a directory: {res}", file=sys.stderr)
        sys.exit(1)

    out = args.out or (res / "THESIS_ANALYSIS_REPORT.md")
    text = build_report(res)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
