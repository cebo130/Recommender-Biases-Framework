#!/usr/bin/env python3
"""
Run the same three experiments as run_all.py (macro bias, user-centric, feedback simulation)
on the Takealot dataset using the official train/test CSV split.

Uses standalone_takealot.takealot_loader for I/O and run_all.run_* for models (6 recommenders
where applicable; feedback sim uses SVD + TF-IDF only, same as run_all).

Usage (from Thesis_Research):
  python run_takealot_full.py
  python run_takealot_full.py --data-dir data/take-a-lot-dataset
  python run_takealot_full.py --output-root debiased_outputs
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
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from standalone_takealot.takealot_loader import load_takealot  # noqa: E402
from utils import ensure_output_dirs, safe_run_label  # noqa: E402

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
    parser.add_argument(
        "--run-label",
        default=None,
        metavar="NAME",
        help="Write outputs under More_Outputs/runs/NAME/figures and .../results.",
    )
    parser.add_argument(
        "--run-label-timestamp",
        action="store_true",
        help="Auto folder More_Outputs/runs/run_YYYYMMDD_HHMMSS/ (overrides --run-label if both given).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write figures/ and results/ directly under DIR (e.g. debiased_outputs). "
        "If set, overrides --run-label paths.",
    )
    parser.add_argument(
        "--no-svd-mitigation-extras",
        action="store_true",
        help="Skip SVD pool baselines, rerank sensitivity CSV, and max-pool=100 cache (faster).",
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

    wall_start = datetime.now()
    run_label = None
    if args.output_root is None:
        if args.run_label_timestamp:
            run_label = wall_start.strftime("run_%Y%m%d_%H%M%S")
        elif args.run_label:
            run_label = safe_run_label(args.run_label)

    if args.output_root is not None:
        run_root = args.output_root.expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        fig_dir = run_root / "figures"
        res_dir = run_root / "results"
        fig_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        out = {"figures": fig_dir, "results": res_dir, "run_root": run_root}
    else:
        out = ensure_output_dirs(ROOT, run_label=run_label)
        fig_dir = out["figures"]
        res_dir = out["results"]
    data_root = ROOT / "data"

    if run_label is not None or args.output_root is not None:
        info_lines = [
            f"dataset=take-a-lot-dataset",
            f"feedback_pool={args.feedback_pool}",
            f"started={wall_start.isoformat(timespec='seconds')}",
        ]
        if run_label is not None:
            info_lines.insert(0, f"run_label={run_label}")
        if args.output_root is not None:
            info_lines.insert(0, f"output_root={out['run_root']}")
        out["run_root"].joinpath("RUN_INFO.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")

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

    t0 = time.perf_counter()
    n_phases = 3

    print(f"Run started: {wall_start.isoformat(timespec='seconds')}")
    print(
        f">>> OVERALL PROGRESS: {0.0:.1f}% | Takealot - three experiments (macro -> user-centric -> feedback)"
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
        mitigations=not args.no_svd_mitigation_extras,
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
        mitigations=not args.no_svd_mitigation_extras,
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
