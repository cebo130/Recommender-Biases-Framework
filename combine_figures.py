#!/usr/bin/env python3
"""
Combine all raster images under a folder into one grid PNG (e.g. for LLM analysis).

Default folder matches this project: outputs/figures/

Usage:
  python combine_figures.py
  python combine_figures.py --dir outputs/figures --out outputs/figures/all_figures_montage.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp"}


def collect_images(folder: Path, skip_names: set[str]) -> list[Path]:
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if p.name in skip_names:
                continue
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine images in a folder into one montage PNG.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Folder containing images (default: outputs/figures)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <dir>/combined_montage.png)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns (default: ceil(sqrt(n)), capped at 6)",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=6,
        help="Maximum columns when cols is auto (default: 6)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saved PNG (default: 150)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=4.0,
        help="Approximate width/height in inches per grid cell (default: 4)",
    )
    args = parser.parse_args()

    folder = args.dir.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    out_path = args.out
    if out_path is None:
        out_path = folder / "combined_montage.png"
    else:
        out_path = out_path.resolve()

    # Avoid including a previous montage (same folder) when re-running.
    skip = {out_path.name} if out_path.parent == folder else set()
    paths = collect_images(folder, skip)
    if not paths:
        raise SystemExit(f"No raster images found in {folder} (supported: {sorted(IMAGE_SUFFIXES)})")

    n = len(paths)
    if args.cols is not None:
        ncols = max(1, args.cols)
    else:
        ncols = min(args.max_cols, max(1, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)

    fig_w = args.cell_size * ncols
    fig_h = args.cell_size * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n:
            continue
        p = paths[idx]
        try:
            img = mpimg.imread(p)
        except Exception as e:  # noqa: BLE001 — surface load errors clearly
            ax.text(0.5, 0.5, f"Could not load\n{p.name}\n{e}", ha="center", va="center", fontsize=8)
            continue
        ax.imshow(img)
        ax.set_title(p.name, fontsize=8, wrap=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {n} image(s) to {out_path}")


if __name__ == "__main__":
    main()
