import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import default_output_parent, ensure_output_dirs, safe_run_label  # noqa: E402

FIG_DIR = ROOT / default_output_parent() / "figures"
RES_DIR = ROOT / default_output_parent() / "results"

ALL_DATASETS = ["ml-100k", "ml-1m", "lastfm-2k", "book-crossing"]


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def run_per_dataset(
    dataset: str, test_size: float, run_label: Optional[str], *, no_svd_mitigation_extras: bool = False
):
    cmd = ["python", str(ROOT / "run_all.py"), "--dataset", dataset, "--test-size", str(test_size)]
    if run_label:
        cmd += ["--run-label", run_label]
    if no_svd_mitigation_extras:
        cmd += ["--no-svd-mitigation-extras"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def aggregate_results(datasets: list[str]) -> dict[str, pd.DataFrame]:
    macro_rows = []
    fairness_rows = []
    feedback_rows = []

    for ds in datasets:
        macro_p = RES_DIR / f"macro_bias_metrics_{ds}.csv"
        fairness_p = RES_DIR / f"user_centric_metrics_{ds}.csv"
        feedback_p = RES_DIR / f"feedback_simulation_metrics_{ds}.csv"

        m = _read_if_exists(macro_p)
        if m is not None:
            m["dataset"] = ds
            macro_rows.append(m)

        f = _read_if_exists(fairness_p)
        if f is not None:
            f["dataset"] = ds
            fairness_rows.append(f)

        fb = _read_if_exists(feedback_p)
        if fb is not None:
            fb["dataset"] = ds
            feedback_rows.append(fb)

    out = {}
    if macro_rows:
        out["macro"] = pd.concat(macro_rows, ignore_index=True)
    if fairness_rows:
        out["fairness"] = pd.concat(fairness_rows, ignore_index=True)
    if feedback_rows:
        out["feedback"] = pd.concat(feedback_rows, ignore_index=True)
    return out


def plot_macro_summary(macro: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    metrics = [
        "gini",
        "arp",
        "catalog_coverage",
        "spearman_pop_vs_recfreq",
        "avg_popularity_percentile",
        "long_tail_share_20pct",
        "aggregate_diversity_unique_items",
        "metadata_token_entropy_bits",
    ]
    metrics = [m for m in metrics if m in macro.columns]

    for metric in metrics:
        pivot = macro.pivot_table(index="model", columns="dataset", values=metric, aggfunc="mean")
        plt.figure(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Macro-bias: {metric} (model × dataset)")
        plt.tight_layout()
        out = FIG_DIR / f"combined_macro_{metric}_heatmap.png"
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()


def plot_fairness_summary(fairness: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    # Fairness gap per dataset/model
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=fairness,
        x="dataset",
        y="delta_accuracy_rmse_niche_minus_mainstream",
        hue="model",
    )
    plt.title("User-centric fairness gap (ΔRMSE = RMSE_Niche - RMSE_Mainstream)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = FIG_DIR / "combined_fairness_gap_delta_rmse.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    # RMSE niche vs mainstream, faceted by dataset
    rmse_long = fairness.melt(
        id_vars=["dataset", "model"],
        value_vars=["rmse_niche", "rmse_mainstream"],
        var_name="segment",
        value_name="rmse",
    )
    rmse_long["segment"] = rmse_long["segment"].str.replace("rmse_", "", regex=False).str.title()

    g = sns.catplot(
        data=rmse_long,
        x="model",
        y="rmse",
        hue="segment",
        col="dataset",
        kind="bar",
        col_wrap=2,
        height=4,
        aspect=1.3,
        sharey=False,
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=20)
    g.fig.suptitle("RMSE by segment across datasets", y=1.02)
    g.fig.tight_layout()
    out = FIG_DIR / "combined_rmse_niche_vs_mainstream.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_feedback_summary(feedback: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    # Diversity decay per dataset
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=feedback,
        x="iteration",
        y="aggregate_diversity",
        hue="model",
        style="dataset",
        markers=True,
        dashes=False,
    )
    plt.title("Feedback simulation: diversity decay (all datasets)")
    plt.tight_layout()
    out = FIG_DIR / "combined_feedback_diversity_decay.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run all experiments across all datasets and aggregate results.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to run (default: all).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio passed to run_all.py")
    parser.add_argument(
        "--skip-exec",
        action="store_true",
        help="Skip executing per-dataset runs; only aggregate/plot existing CSVs.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        metavar="NAME",
        help="Per-dataset runs and combined CSVs/plots go under More_Outputs/runs/NAME/.",
    )
    parser.add_argument(
        "--run-label-timestamp",
        action="store_true",
        help="Auto folder More_Outputs/runs/run_YYYYMMDD_HHMMSS/ (overrides --run-label if both given).",
    )
    parser.add_argument(
        "--no-svd-mitigation-extras",
        action="store_true",
        help="Forward to run_all.py: skip SVD pool baselines and rerank sensitivity CSV.",
    )
    args = parser.parse_args()

    global FIG_DIR, RES_DIR
    wall_start = datetime.now()
    run_label: Optional[str] = None
    if args.run_label_timestamp:
        run_label = wall_start.strftime("run_%Y%m%d_%H%M%S")
    elif args.run_label:
        run_label = safe_run_label(args.run_label)

    out = ensure_output_dirs(ROOT, run_label=run_label)
    FIG_DIR = out["figures"]
    RES_DIR = out["results"]
    if run_label:
        out["run_root"].joinpath("RUN_INFO.txt").write_text(
            f"run_label={run_label}\n"
            f"mode=run_all_datasets\n"
            f"datasets={','.join(args.datasets)}\n"
            f"test_size={args.test_size}\n"
            f"skip_exec={args.skip_exec}\n"
            f"started={wall_start.isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    t0 = time.perf_counter()
    datasets = list(args.datasets)
    n_ds = len(datasets)
    # First ~90% = per-dataset runs; last ~10% = combine + plots (heuristic).
    DATASET_FRACTION = 0.90

    print(f"Multi-dataset run started: {wall_start.isoformat(timespec='seconds')}")
    print(f"Datasets ({n_ds}): {', '.join(datasets)}")
    if args.skip_exec:
        print(f">>> OVERALL PROGRESS: {0.0:.1f}% | --skip-exec: skipping per-dataset runs")

    if not args.skip_exec and n_ds > 0:
        for i, ds in enumerate(datasets):
            pct_before = 100.0 * DATASET_FRACTION * (i / max(1, n_ds))
            print(
                f"\n{'=' * 72}\n"
                f">>> OVERALL PROGRESS: {pct_before:.1f}% | dataset {i + 1}/{n_ds}: {ds}\n"
                f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
                f"{'=' * 72}"
            )
            run_per_dataset(
                ds,
                test_size=args.test_size,
                run_label=run_label,
                no_svd_mitigation_extras=args.no_svd_mitigation_extras,
            )
            pct_after = 100.0 * DATASET_FRACTION * ((i + 1) / n_ds)
            print(
                f">>> OVERALL PROGRESS: {pct_after:.1f}% | finished {ds} ({i + 1}/{n_ds})\n"
                f"    Elapsed: {_format_duration(time.perf_counter() - t0)}"
            )

    t_agg = time.perf_counter()
    if args.skip_exec:
        pct_combine = 10.0
    else:
        pct_combine = min(99.0, 100.0 * DATASET_FRACTION + 5.0)
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: {pct_combine:.1f}% | aggregating CSVs + combined plots\n"
        f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
        f"{'=' * 72}"
    )

    combined = aggregate_results(args.datasets)
    if "macro" in combined:
        combined["macro"].to_csv(RES_DIR / "combined_macro_bias_metrics.csv", index=False)
        plot_macro_summary(combined["macro"])
    if "fairness" in combined:
        combined["fairness"].to_csv(RES_DIR / "combined_user_centric_metrics.csv", index=False)
        plot_fairness_summary(combined["fairness"])
    if "feedback" in combined:
        combined["feedback"].to_csv(RES_DIR / "combined_feedback_simulation_metrics.csv", index=False)
        plot_feedback_summary(combined["feedback"])

    t1 = time.perf_counter()
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: 100.0% | ALL DATASETS + COMBINED OUTPUTS COMPLETE\n"
        f"    Aggregation/plots duration: {_format_duration(t1 - t_agg)}\n"
        f"    Total runtime: {_format_duration(t1 - t0)} (since {wall_start.isoformat(timespec='seconds')})\n"
        f"    Combined CSVs: {RES_DIR}\n"
        f"    Combined figures: {FIG_DIR}\n"
        f"{'=' * 72}"
    )


if __name__ == "__main__":
    main()

