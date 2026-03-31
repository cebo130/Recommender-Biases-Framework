# Thesis Research — Recommender bias, fairness & feedback loops

Python experiments comparing **six recommenders** (k-NN, SVD, ALS, TF‑IDF, per-user logistic regression, Random Forest) on **MovieLens 100K/1M**, **Last.fm 2K**, **Book-Crossing**, and **Takealot**-style review data. The pipeline runs three studies: **macro popularity/exposure bias**, **user-segment RMSE and NDCG@10**, and a **synthetic feedback-loop simulation** (SVD vs TF‑IDF).

| More detail | Link |
|-------------|------|
| Extended methodology, metrics glossary, troubleshooting | [**docs/PROJECT.md**](docs/PROJECT.md) |
| Plain-text folder tree (copy/paste) | [**docs/PROJECT_TREE.txt**](docs/PROJECT_TREE.txt) |
| LaTeX thesis chapter + bibliography | `thesis.tex`, `references.bib` |

---

## Table of contents

- [What this repository runs](#what-this-repository-runs)
- [Project folder tree](#project-folder-tree)
- [Prerequisites](#prerequisites)
- [Installation (step by step)](#installation-step-by-step)
- [Obtain and place datasets](#obtain-and-place-datasets)
- [Reproduce experiments](#reproduce-experiments)
- [Command reference](#command-reference)
- [What gets produced](#what-gets-produced)
- [Jupyter notebooks](#jupyter-notebooks)
- [Build the PDF thesis](#build-the-pdf-thesis)
- [Important caveats](#important-caveats)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What this repository runs

1. **`run_all.py`** — For **one** public dataset (`ml-100k`, `ml-1m`, `lastfm-2k`, `book-crossing`): loads data via `src/utils.py`, applies **k-core** filtering (default: min 10 user and 10 item interactions), splits **80/20 train/test** (fixed seed **42**), then runs in order:
   - **Macro bias:** top-10 recommendations per test user; metrics (Gini, ARP, coverage, Spearman ρ, tail share, aggregate diversity, metadata entropy, RMSE where available).
   - **User-centric:** users split into **Niche / Diverse / Mainstream** tertiles; segment RMSE and NDCG@10.
   - **Feedback simulation:** 5 rounds, top-5 recs, synthetic clicks; tracks diversity and ARP for **SVD** and **TF‑IDF** only.

2. **`run_all_datasets.py`** — Runs `run_all.py` for **all four** public datasets (or a subset), then merges CSVs and writes **combined** plots (heatmaps, fairness gap chart, feedback lines).

3. **`run_takealot_full.py`** — Same three experiments using the **official** `reviews/train.csv` and `reviews/test.csv` split and item text from `products.csv`. Optional `--feedback-pool all|train` controls which ratings feed the feedback simulation.

4. **`run_notebooks.py`** — Executes notebooks `01`–`03` with a chosen dataset name injected into `DATASET_NAME`.

---

## Project folder tree

Layout below matches the intended repository structure. Large dataset blobs are summarized; your clone may omit some `data/` files until you download them.

```
Thesis_Research/
├── README.md                          # This file
├── docs/
│   └── PROJECT.md                     # Long-form documentation
├── requirements.txt                   # Core Python dependencies
├── requirements_optional_windows_build.txt   # scikit-surprise (optional)
├── thesis.tex                         # LaTeX thesis chapter
├── references.bib                     # BibTeX references
├── build_thesis.ps1                   # Windows: locate pdflatex and compile thesis.pdf
│
├── run_all.py                         # Single public dataset: full pipeline
├── run_all_datasets.py                # All public datasets + combined outputs
├── run_takealot_full.py               # Takealot: official split + experiments
├── run_notebooks.py                   # Execute notebooks 01–03 with --dataset
│
├── src/
│   └── utils.py                       # load_dataset(), k_core_filter(), per-dataset loaders
│
├── notebooks/
│   ├── 01_macro_bias.ipynb
│   ├── 02_user_centric.ipynb
│   ├── 03_feedback_simulation.ipynb
│   └── 04_all_datasets.ipynb
│
├── standalone_takealot/
│   ├── takealot_loader.py             # resolve_takealot_root(), load_takealot(), ratings_from_reviews_df()
│   ├── run_takealot_smoke.py          # Quick sanity check (if present)
│   └── explore_takealot.ipynb
│
├── data/                              # Create this; not always committed to git
│   ├── ml-100k/
│   │   ├── u.data                     # ratings (tab-separated)
│   │   ├── u.item                     # items + genres
│   │   └── ...                        # other MovieLens 100K files (u.user, splits, etc.)
│   ├── ml-1m/
│   │   ├── ratings.dat
│   │   └── movies.dat
│   ├── lastfm-2k/
│   │   ├── user_artists.dat
│   │   ├── artists.dat
│   │   ├── tags.dat
│   │   └── user_taggedartists.dat
│   ├── book-crossing/
│   │   ├── Ratings.csv
│   │   └── Books.csv
│   └── take-a-lot-dataset/            # Or set TAKEALOT_DATA_DIR elsewhere
│       ├── products.csv
│       ├── customers.csv
│       ├── reviews.csv
│       └── reviews/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
│
└── outputs/                           # Created when you run experiments
    ├── figures/                       # PNG plots (macro, user-centric, feedback, combined)
    ├── results/                       # CSV metrics per dataset + combined_*.csv
    └── executed_notebooks/            # From run_notebooks.py
```

To regenerate a machine-local tree (optional):

```bash
# Linux / macOS / Git Bash
find . -path ./.git -prune -o -print | head -200

# Or: tree -I '.git' -L 4
```

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** 3.10+ | 3.11 recommended |
| **Disk** | Several GB for MovieLens 1M + figures; Takealot can be large |
| **RAM** | `ml-100k` is modest; `ml-1m` and Takealot need more headroom |
| **LaTeX** (optional) | MiKTeX or TeX Live to compile `thesis.tex` → PDF |

---

## Installation (step by step)

**1. Clone or copy the project**

```bash
cd /path/to
git clone <your-repo-url> Thesis_Research
cd Thesis_Research
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`  
  If execution policy blocks: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- **Linux / macOS:** `source .venv/bin/activate`

**3. Upgrade pip and install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**4. (Recommended) Install scikit-surprise for real k-NN and SVD**

```bash
pip install -r requirements_optional_windows_build.txt
# or: pip install scikit-surprise
```

On Windows, if pip cannot find a wheel, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use a prebuilt wheel from a matching Python version.

**5. Verify imports**

```bash
python -c "import implicit, sklearn, pandas; print('OK')"
python -c "import surprise; print('Surprise OK')"   # should work if step 4 succeeded
```

---

## Obtain and place datasets

### MovieLens 100K / 1M

- Download from [GroupLens MovieLens](https://grouplens.org/datasets/movielens/).
- Extract so that **`u.data`** and **`u.item`** live under `data/ml-100k/`, and **`ratings.dat`**, **`movies.dat`** under `data/ml-1m/`.

### Last.fm 2K (HetRec 2011 style)

- Use the HetRec 2011 / Last.fm dataset files; place **`user_artists.dat`**, **`artists.dat`**, **`tags.dat`**, **`user_taggedartists.dat`** under `data/lastfm-2k/`.

### Book-Crossing

- Obtain **BX-CSV-Dump** (or equivalent); place **`Ratings.csv`** and **`Books.csv`** under `data/book-crossing/`.

### Takealot

- Populate **`data/take-a-lot-dataset/`** with at least **`products.csv`** and review CSVs as used by `standalone_takealot/takealot_loader.py`.
- Alternatively set **`TAKEALOT_DATA_DIR`** to an absolute path containing the same files.
- Related public context (workshop materials): [RecSys2025 Takealot GitHub](https://github.com/stefandominicus-takealot/RecSys2025) — your thesis `references.bib` may cite this; your actual CSV layout must match the loader.

---

## Reproduce experiments

Follow this order the first time you run the project.

### A. Smoke test (fastest)

Use **MovieLens 100K** — smallest public run.

```bash
# From repository root, venv activated
python run_all.py --dataset ml-100k --test-size 0.2
```

**Expected:** Console progress for three phases; no unhandled tracebacks.  
**Artifacts:** `outputs/results/macro_bias_metrics_ml-100k.csv`, `user_centric_metrics_ml-100k.csv`, `feedback_simulation_metrics_ml-100k.csv`, and PNGs under `outputs/figures/`.

### B. Full public benchmark batch

```bash
python run_all_datasets.py
```

This can take **a long time** (especially ML-1M). To only **recombine existing CSVs** without re-running models:

```bash
python run_all_datasets.py --skip-exec
```

To limit datasets:

```bash
python run_all_datasets.py --datasets ml-100k lastfm-2k --test-size 0.2
```

### C. Takealot

```bash
python run_takealot_full.py --data-dir data/take-a-lot-dataset
```

Use `--feedback-pool train` if the simulation should only extend the **training** log, or `--feedback-pool all` (default) to use the full reviews pool where implemented.

### D. Notebooks (optional)

```bash
python run_notebooks.py --dataset ml-100k --timeout-seconds 36000
```

Executed copies appear under **`outputs/executed_notebooks/`**.

---

## Command reference

| Command | Purpose |
|---------|---------|
| `python run_all.py --dataset <name> [--test-size 0.2]` | One dataset; three experiments |
| `python run_all_datasets.py [--datasets ...] [--skip-exec] [--test-size 0.2]` | Batch + combined outputs |
| `python run_takealot_full.py [--data-dir PATH] [--feedback-pool all\|train]` | Takealot experiments |
| `python run_notebooks.py --dataset <name>` | Run notebooks 01–03 |

**`--dataset` values (public):** `ml-100k`, `ml-1m`, `lastfm-2k`, `book-crossing`.

---

## What gets produced

### `outputs/results/`

| File pattern | Content |
|--------------|---------|
| `macro_bias_metrics_<dataset>.csv` | Per model: Gini, ARP, coverage, Spearman, percentiles, tail share, diversity, entropy, RMSE (if computed) |
| `user_centric_metrics_<dataset>.csv` | Per model: RMSE/NDCG by Niche/Diverse/Mainstream, ΔRMSE niche−mainstream |
| `feedback_simulation_metrics_<dataset>.csv` | Per iteration: SVD vs TF‑IDF aggregate diversity and ARP |
| `combined_*.csv` | After `run_all_datasets.py`: stacked metrics across datasets |

Takealot runs use `<dataset>` like **`take-a-lot-dataset`** in filenames.

### `outputs/figures/`

Log–log popularity plots, extended metric bar charts, ECDFs, segment RMSE bars, feedback diversity lines, and **combined** heatmaps (when batch script completes).

---

## Jupyter notebooks

| Notebook | Role |
|----------|------|
| `01_macro_bias.ipynb` | Exploratory / teaching mirror of macro experiment |
| `02_user_centric.ipynb` | User segments |
| `03_feedback_simulation.ipynb` | Feedback loop |
| `04_all_datasets.ipynb` | Multi-dataset / aggregation style workflow |

Set **`DATASET_NAME`** in the notebook config cell, or use **`run_notebooks.py`** to inject it.

---

## Build the PDF thesis

Requires **pdflatex** + **bibtex** (e.g. [MiKTeX](https://miktex.org/download) on Windows).

```bash
pdflatex thesis
bibtex thesis
pdflatex thesis
pdflatex thesis
```

From repo root. On Windows, if `pdflatex` is not on PATH, run **`build_thesis.ps1`**.  
**Overleaf:** upload `thesis.tex` and `references.bib`, set main file to `thesis.tex`.

---

## Important caveats

1. **Surprise:** If `import surprise` fails, **`run_all.py`** uses **`_FallbackCF`** (global mean + biases) instead of real k-NN/SVD — **results are not comparable** to standard Surprise papers.
2. **Segment RMSE:** For **ALS**, **TF‑IDF**, and **per-user LogReg**, segment RMSE uses a **global mean** placeholder; **k-NN**, **SVD**, and **RandomForest** get real per-row predictions. **NDCG@10** still uses proper top‑K lists for all models.
3. **ALS:** On errors, the code may skip to trivial recommendations — check console for “ALS skipped”.
4. **Runtime:** Full `run_all_datasets.py` on ML-1M is heavy; plan overnight or use `--datasets` to subset.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `FileNotFoundError: Dataset directory not found` | Confirm `data/<dataset_name>/` exists and matches `src/utils.py` expectations |
| `No module named 'surprise'` | Install optional requirements; or accept fallback (see caveats) |
| `implicit` / ALS errors | `pip install implicit`; check scipy/numpy versions |
| Empty or missing figures | Ensure the run completed; check `outputs/figures/` permissions |
| Takealot not found | `--data-dir` or `TAKEALOT_DATA_DIR`; folder must contain `products.csv` |
| `pdflatex` not recognized | Install MiKTeX/TeX Live; restart terminal; use `build_thesis.ps1` |

More detail: [**docs/PROJECT.md**](docs/PROJECT.md).

---

## License

Respect **dataset licenses** (MovieLens, Book-Crossing, Last.fm, Takealot). Code and thesis reuse: follow your **institution** and supervisor rules.
