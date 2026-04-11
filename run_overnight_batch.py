#!/usr/bin/env python3
"""
Run the main experiment drivers **one after another** (same shell session, stop on first failure).

Default order (matches a typical thesis batch):
  1) ``run_all.py`` — single dataset (default ``ml-100k``)
  2) ``run_takealot_full.py`` — Takealot official split
  3) ``run_all_datasets.py`` — MovieLens 1M, Last.fm, Book-Crossing (and again ml-100k)

All steps share the same ``--run-label`` so outputs go under ``More_Outputs/runs/<label>/``
(unless ``THESIS_OUTPUT_PARENT`` overrides the ``More_Outputs`` segment).

**Lambda / pool_k grid:** With mitigation extras enabled (default), each eligible run writes
``svd_rerank_sensitivity_<dataset>.csv`` with **all** combinations of
``RERANK_SENSITIVITY_LAMBDAS`` × ``RERANK_SENSITIVITY_POOL_KS`` in one macro pass (no need to
invoke this script 12 times). Override grids via env or this script's ``--rerank-lambdas`` /
``--rerank-pool-ks``.

Examples::

  python run_overnight_batch.py
  python run_overnight_batch.py --run-label thesis_apr09
  python run_overnight_batch.py --fast
  python run_overnight_batch.py --skip-standalone-run-all
  python run_overnight_batch.py --rerank-lambdas 0.5,0.8 --rerank-pool-ks 30,100

Sleep / lock: disable system sleep for AC power while this runs, or use ``nssm`` / Task Scheduler.

After a successful batch, ``generate_thesis_analysis_report.py`` runs automatically and writes
``THESIS_ANALYSIS_REPORT.md`` into the run's ``results/`` folder (unless ``--no-thesis-report``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from utils import default_output_parent, safe_run_label  # noqa: E402


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential overnight run: run_all.py, run_takealot_full.py, run_all_datasets.py."
    )
    parser.add_argument(
        "--run-label",
        default=None,
        metavar="NAME",
        help="Shared More_Outputs/runs/NAME/ for every step (default: overnight_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--run-label-timestamp",
        action="store_true",
        help="Use run_YYYYMMDD_HHMMSS instead of overnight_* (overrides default label).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Pass --no-svd-mitigation-extras to every step (no sensitivity CSV, no pool baselines).",
    )
    parser.add_argument(
        "--skip-standalone-run-all",
        action="store_true",
        help="Skip step 1 (run_all.py). Saves time; ml-100k still runs inside run_all_datasets.py.",
    )
    parser.add_argument(
        "--dataset",
        default="ml-100k",
        choices=["ml-100k", "ml-1m", "lastfm-2k", "book-crossing"],
        help="Dataset for standalone run_all.py (step 1 only).",
    )
    parser.add_argument(
        "--takealot-data-dir",
        type=Path,
        default=None,
        help="Optional --data-dir for run_takealot_full.py.",
    )
    parser.add_argument(
        "--rerank-lambdas",
        default=None,
        metavar="CSV",
        help="Sets THESIS_RERANK_SENSITIVITY_LAMBDAS for child processes, e.g. 0.5,0.65,0.8,0.95",
    )
    parser.add_argument(
        "--rerank-pool-ks",
        default=None,
        metavar="CSV",
        help="Sets THESIS_RERANK_SENSITIVITY_POOL_KS for child processes, e.g. 30,50,100",
    )
    parser.add_argument(
        "--no-thesis-report",
        action="store_true",
        help="Do not run generate_thesis_analysis_report.py after all steps succeed.",
    )
    args = parser.parse_args()

    if args.run_label_timestamp:
        run_label = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    elif args.run_label:
        run_label = safe_run_label(args.run_label)
    else:
        run_label = datetime.now().strftime("overnight_%Y%m%d_%H%M%S")

    if args.rerank_lambdas:
        os.environ["THESIS_RERANK_SENSITIVITY_LAMBDAS"] = args.rerank_lambdas.strip()
    if args.rerank_pool_ks:
        os.environ["THESIS_RERANK_SENSITIVITY_POOL_KS"] = args.rerank_pool_ks.strip()

    extra: list[str] = []
    if args.fast:
        extra.append("--no-svd-mitigation-extras")

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_standalone_run_all:
        cmd1 = [py, str(ROOT / "run_all.py"), "--dataset", args.dataset, "--run-label", run_label] + extra
        steps.append(("run_all.py", cmd1))

    cmd2 = [py, str(ROOT / "run_takealot_full.py")]
    if args.takealot_data_dir is not None:
        cmd2 += ["--data-dir", str(args.takealot_data_dir.expanduser().resolve())]
    cmd2 += ["--run-label", run_label] + extra
    steps.append(("run_takealot_full.py", cmd2))

    cmd3 = [py, str(ROOT / "run_all_datasets.py"), "--run-label", run_label] + extra
    steps.append(("run_all_datasets.py", cmd3))

    wall_start = datetime.now()
    print(f"Overnight batch started: {wall_start.isoformat(timespec='seconds')}")
    print(f"Shared run_label: {run_label}")
    print(f"Outputs: {ROOT / default_output_parent() / 'runs' / run_label}")
    print(f"Mitigation extras: {'OFF (--fast)' if args.fast else 'ON (sensitivity + baselines)'}")
    if args.rerank_lambdas:
        print(f"THESIS_RERANK_SENSITIVITY_LAMBDAS={os.environ.get('THESIS_RERANK_SENSITIVITY_LAMBDAS')}")
    if args.rerank_pool_ks:
        print(f"THESIS_RERANK_SENSITIVITY_POOL_KS={os.environ.get('THESIS_RERANK_SENSITIVITY_POOL_KS')}")
    print(f"Steps: {len(steps)} sequential jobs, stop on first failure.\n")

    t0 = time.perf_counter()
    for i, (name, cmd) in enumerate(steps, start=1):
        print("=" * 72)
        print(f"[{i}/{len(steps)}] {name}")
        print(" ".join(cmd))
        print("=" * 72)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        print(f"    Done in {_format_duration(time.perf_counter() - t0)} elapsed since batch start\n")

    total = time.perf_counter() - t0
    print("=" * 72)
    print(f"Overnight batch COMPLETE. Total wall time: {_format_duration(total)}")
    run_results = ROOT / default_output_parent() / "runs" / run_label / "results"
    print(f"Results: {run_results}")
    print("=" * 72)

    if not args.no_thesis_report:
        run_root = ROOT / default_output_parent() / "runs" / run_label
        report_cmd = [
            py,
            str(ROOT / "generate_thesis_analysis_report.py"),
            "--run-root",
            str(run_root),
        ]
        print("\nGenerating thesis Markdown report from CSVs...")
        print(" ".join(report_cmd))
        try:
            subprocess.run(report_cmd, cwd=str(ROOT), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: thesis report step failed ({e}); CSVs are still under {run_results}")


if __name__ == "__main__":
    main()
