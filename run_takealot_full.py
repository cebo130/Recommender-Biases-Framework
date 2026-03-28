#!/usr/bin/env python3
"""
Run the same three experiments as run_all.py (macro bias, user-centric, feedback simulation)
on the Takealot dataset using the official train/test CSV split.

Uses standalone_takealot.takealot_loader for I/O and run_all.run_* for models (6 recommenders
where applicable; feedback sim uses SVD + TF-IDF only, same as run_all).

Usage (from Thesis_Research):
  python run_takealot_full.py
  python run_takealot_full.py --data-dir data/take-a-lot-dataset
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from standalone_takealot.takealot_loader import load_takealot  # noqa: E402

import run_all  # noqa: E402

DATASET_NAME = "take-a-lot-dataset"


def _format_duration(seconds: float) -> str:
    return run_all._format_duration(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full bias experiments on Takealot (official train/test split)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to take-a-lot-dataset folder (default: <repo>/data/take-a-lot-dataset)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Unused for macro/user (official split). Kept for CLI parity with run_all.py.",
    )
    parser.add_argument(
        "--feedback-pool",
        choices=("all", "train"),
        default="all",
        help="Ratings pool for feedback simulation: reviews.csv (all) or train split only.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = ROOT / "data" / "take-a-lot-dataset"
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Takealot data directory not found: {data_dir}")
    if not (data_dir / "products.csv").is_file():
        raise FileNotFoundError(f"Missing products.csv under {data_dir}")

    out = run_all.ensure_output_dirs(ROOT)
    fig_dir = out["figures"]
    res_dir = out["results"]
    data_root = ROOT / "data"

    print(f"Takealot data dir: {data_dir}")
    print("Loading train / test / item features...")
    train_df, item_features = load_takealot("train", data_dir, dedupe_user_item=True)
    test_df, _ = load_takealot("test", data_dir, dedupe_user_item=True, attach_item_features=False)

    if args.feedback_pool == "all":
        feedback_ratings, _ = load_takealot("all", data_dir, dedupe_user_item=True, attach_item_features=False)
    else:
        feedback_ratings = train_df.copy()

    print(f"Train interactions: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Feedback pool ({args.feedback_pool}): {len(feedback_ratings):,} interactions")
    print("Outputs:")
    print(f"  Figures: {fig_dir}")
    print(f"  Results: {res_dir}")

    wall_start = datetime.now()
    t0 = time.perf_counter()
    n_phases = 3

    print(f"Run started: {wall_start.isoformat(timespec='seconds')}")
    print(
        f">>> OVERALL PROGRESS: {0.0:.1f}% | Takealot — three experiments (macro → user-centric → feedback)"
    )

    pre = (train_df, test_df)
    t_phase = t0
    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: {100.0 * 0 / n_phases:.1f}% | [1/3] Macro-bias (official train/test)\n"
        f"    Elapsed: {_format_duration(time.perf_counter() - t0)}\n"
        f"{'=' * 72}"
    )
    run_all.run_macro_bias(
        DATASET_NAME,
        data_root,
        fig_dir,
        res_dir,
        args.test_size,
        pre_split=pre,
        item_features_override=item_features,
    )
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
    run_all.run_user_centric(
        DATASET_NAME,
        data_root,
        fig_dir,
        res_dir,
        args.test_size,
        pre_split=pre,
        item_features_override=item_features,
    )
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
    run_all.run_feedback_simulation(
        DATASET_NAME,
        data_root,
        fig_dir,
        res_dir,
        ratings_override=feedback_ratings,
        item_features_override=item_features,
    )
    t3 = time.perf_counter()
    print(
        f">>> OVERALL PROGRESS: {100.0 * 3 / n_phases:.1f}% | [3/3] Feedback simulation DONE\n"
        f"    Phase duration: {_format_duration(t3 - t_phase)} | Total elapsed: {_format_duration(t3 - t0)}"
    )

    print(
        f"\n{'=' * 72}\n"
        f">>> OVERALL PROGRESS: 100.0% | TAKEALOT RUN COMPLETE\n"
        f"    Total runtime: {_format_duration(t3 - t0)} (wall start {wall_start.isoformat(timespec='seconds')})\n"
        f"{'=' * 72}"
    )


if __name__ == "__main__":
    main()
