#!/usr/bin/env python3
"""
Bundle all CSV files under More_Outputs/results/ into one file for sharing (e.g. with an LLM).

Default: Markdown with one section per CSV (readable and easy to cite).
Optional: one concatenated CSV with a leading source_file column (union of columns; NaN where absent).

Usage:
  python3 combine_results.py
  python3 combine_results.py --format csv --out More_Outputs/results/combined_all_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from utils import default_output_parent  # noqa: E402

_DEFAULT_RESULTS = _ROOT / default_output_parent() / "results"


def collect_csvs(folder: Path, skip_names: set[str], skip_combined_prefix: bool) -> list[Path]:
    files: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if p.name in skip_names:
            continue
        if skip_combined_prefix and p.name.startswith("combined_"):
            continue
        files.append(p)
    return files


def write_markdown(paths: list[Path], out_path: Path) -> None:
    lines: list[str] = [
        "# Combined results (CSV bundle)",
        "",
        "Each section is one source file from `More_Outputs/results/` (or your --dir).",
        "",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8")
        lines.append(f"## `{p.name}`")
        lines.append("")
        lines.append("```csv")
        lines.append(text.rstrip("\n"))
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_concat_csv(paths: list[Path], out_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df.insert(0, "source_file", p.name)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle CSV result files into one Markdown or CSV file.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_RESULTS,
        help="Folder containing CSVs (default: <repo>/More_Outputs/results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <dir>/combined_results.md or .csv from --format)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="markdown: sections with fenced CSV (default); csv: one table with source_file column",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip files whose names start with combined_ (avoids duplicating aggregates)",
    )
    args = parser.parse_args()

    folder = args.dir.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    if args.out is None:
        out_path = folder / ("combined_results.md" if args.format == "markdown" else "combined_results_long.csv")
    else:
        out_path = args.out.resolve()

    skip = {out_path.name} if out_path.parent == folder else set()
    paths = collect_csvs(folder, skip, args.skip_combined)
    if not paths:
        raise SystemExit(f"No CSV files found in {folder}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        write_markdown(paths, out_path)
    else:
        write_concat_csv(paths, out_path)

    print(f"Wrote {len(paths)} CSV(s) to {out_path}")


if __name__ == "__main__":
    main()
