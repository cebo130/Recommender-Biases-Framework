import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from utils import default_output_parent  # noqa: E402

NB_DIR = ROOT / "notebooks"
EXEC_DIR = ROOT / default_output_parent() / "executed_notebooks"
EXEC_DIR.mkdir(parents=True, exist_ok=True)


SUPPORTED_DATASETS = {"ml-100k", "ml-1m", "lastfm-2k", "book-crossing"}


def set_dataset_in_notebook(nb, dataset_name: str):
    """
    Update the notebook config cell that defines DATASET_NAME.
    Falls back to inserting a small override cell if not found.
    """
    pattern = re.compile(r"^\s*DATASET_NAME\s*=\s*['\"].*?['\"]\s*$", re.MULTILINE)
    replaced = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if not isinstance(src, str):
            src = "".join(src)
        if "DATASET_NAME" not in src:
            continue
        if pattern.search(src):
            new_src = pattern.sub(f"DATASET_NAME = '{dataset_name}'", src)
            if new_src != src:
                cell["source"] = new_src
                replaced = True
                break

    if not replaced:
        override = nbformat.v4.new_code_cell(
            f"DATASET_NAME = '{dataset_name}'\nprint('Overriding DATASET_NAME ->', DATASET_NAME)\n"
        )
        # Insert near the top (after first markdown title cell if present)
        insert_at = 1 if nb.get("cells") else 0
        nb["cells"].insert(insert_at, override)


def execute_notebook(in_path: Path, out_path: Path, timeout_s: int = 36000):
    nb = nbformat.read(str(in_path), as_version=4)
    ep = ExecutePreprocessor(timeout=timeout_s, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbformat.write(nb, str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Execute experiment notebooks sequentially.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset name under data/",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=36000,
        help="Per-notebook execution timeout (default 10 hours).",
    )
    args = parser.parse_args()

    notebooks = [
        NB_DIR / "01_macro_bias.ipynb",
        NB_DIR / "02_user_centric.ipynb",
        NB_DIR / "03_feedback_simulation.ipynb",
    ]

    for p in notebooks:
        if not p.exists():
            raise FileNotFoundError(f"Notebook not found: {p}")

    for p in notebooks:
        print(f"Executing {p.name} with dataset={args.dataset} ...")
        nb = nbformat.read(str(p), as_version=4)
        set_dataset_in_notebook(nb, args.dataset)

        out_name = p.stem + f"_executed_{args.dataset}.ipynb"
        out_path = EXEC_DIR / out_name

        ep = ExecutePreprocessor(timeout=args.timeout_seconds, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbformat.write(nb, str(out_path))
        print(f"Saved executed notebook: {out_path}")


if __name__ == "__main__":
    main()

