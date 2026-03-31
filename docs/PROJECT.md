# Project documentation — Recommender-system bias, fairness, and feedback loops

This file is the **long-form** companion to the root [README.md](../README.md). It explains goals, layout, data, commands, outputs, notebooks, metrics, methodology, and troubleshooting in full.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [How to run experiments](#how-to-run-experiments)
- [Outputs](#outputs)
- [Notebooks](#notebooks)
- [Models and metrics (reference)](#models-and-metrics-reference)
- [Important methodological notes](#important-methodological-notes)
- [Troubleshooting](#troubleshooting)
- [License and citation](#license-and-citation)

---

## What this project does

### Research questions

1. **Macro bias:** When each model recommends top‑K items to test users, how **concentrated** are recommendations on a few popular items? Metrics include Gini on recommendation-slot frequencies, catalog coverage, Spearman correlation between train popularity and recommendation frequency, average popularity percentiles, long-tail share, aggregate diversity (unique items recommended), and Shannon entropy of metadata tokens in the recommended set. RMSE on test ratings is reported for models that expose rating predictors in this pipeline (e.g. k-NN and SVD in the macro metrics table).

2. **User-centric effects:** Users are split into **Niche / Diverse / Mainstream** tertiles by a **mainstreaminess** score: mean global training popularity of items in that user’s training history. **RMSE** and **NDCG@10** are computed per segment; **ΔRMSE = RMSE_niche − RMSE_mainstream** summarizes whether niche users get worse rating error than mainstream users.

3. **Feedback dynamics:** A **probabilistic click model** blends a relevance signal (from the model) and a normalized popularity signal. Over **five** rounds, training data is augmented with synthetic clicks; **aggregate diversity** (unique items across all recommendation slots) and **ARP** (average training popularity of recommended items) are tracked for **SVD** vs **TF‑IDF** (top‑5 per user per round).

### Models compared

| Family | Implementation |
|--------|------------------|
| User-based collaborative filtering | `surprise.KNNBasic` (cosine, user-based); **fallback** global mean + user/item biases if Surprise is unavailable |
| Matrix factorization | `surprise.SVD` |
| Implicit ALS | `implicit.AlternatingLeastSquares` on a sparse user×item matrix built from training ratings |
| Content-based (TF‑IDF) | scikit-learn TF‑IDF + cosine similarity; user profile = mean vector of consumed items |
| Content-based (per-user) | Per-user TF‑IDF + logistic regression (high vs low rating vs user median); cold users fall back to catalog order |
| ID-based | `RandomForestRegressor` on label-encoded `(user_id, item_id)` |

**Shared code:** `src/utils.py` (loaders, optional k-core, `metadata_text` construction). **Orchestration:** `run_all.py`, `run_all_datasets.py`, `run_takealot_full.py`.

---

## Repository layout

```
Thesis_Research/
├── README.md                 ← short GitHub entry (links here)
├── docs/
│   ├── PROJECT.md            ← this file
│   └── PROJECT_TREE.txt      ← plain-text tree (duplicate of README)
├── requirements.txt
├── requirements_optional_windows_build.txt   # scikit-surprise (optional)
├── run_all.py
├── run_all_datasets.py
├── run_takealot_full.py
├── run_notebooks.py
├── src/
│   └── utils.py
├── notebooks/
│   ├── 01_macro_bias.ipynb
│   ├── 02_user_centric.ipynb
│   ├── 03_feedback_simulation.ipynb
│   └── 04_all_datasets.ipynb
├── standalone_takealot/
│   ├── takealot_loader.py
│   ├── run_takealot_smoke.py
│   └── explore_takealot.ipynb
├── data/
│   ├── ml-100k/
│   ├── ml-1m/
│   ├── lastfm-2k/
│   ├── book-crossing/
│   └── take-a-lot-dataset/
└── outputs/
    ├── figures/
    ├── results/
    └── executed_notebooks/
```

---

## Requirements

- **Python** 3.10+ recommended.
- **Compute:** MovieLens 1M and full Takealot runs can be slow and memory-heavy; start with `ml-100k` to validate the environment.

### Core packages (`requirements.txt`)

| Package | Role |
|---------|------|
| pandas, numpy | Data handling |
| matplotlib, seaborn | Plots |
| scipy | Statistics (e.g. Spearman) |
| scikit-learn | TF‑IDF, logistic regression, random forest, cosine similarity, train/test split, MSE |
| tqdm | Progress |
| implicit | ALS |
| notebook, ipywidgets | Jupyter |

### Optional

- **scikit-surprise** (`requirements_optional_windows_build.txt` on Windows): enables real **k-NN** and **SVD**. If the import fails, `run_all.py` sets `SURPRISE_AVAILABLE = False` and uses **`_FallbackCF`**, which is **not** equivalent to published Surprise benchmarks—always note this in reports.

---

## Installation

```bash
cd Thesis_Research
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_optional_windows_build.txt   # optional: Surprise
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-surprise   # if a wheel exists for your platform
```

---

## Data

### Public benchmarks (`data/<dataset_name>/`)

| Dataset | Typical files |
|---------|----------------|
| **ml-100k** | `u.data`, `u.item`, … |
| **ml-1m** | `ratings.dat`, `movies.dat` |
| **lastfm-2k** | `user_artists.dat`, `tags.dat`, `user_taggedartists.dat`, `artists.dat` |
| **book-crossing** | `Ratings.csv`, `Books.csv` |

**Preprocessing (in code):**

- Optional **k-core** (default: minimum **10** user interactions and **10** item interactions), iterated until stable.
- **Last.fm:** play counts mapped to pseudo-ratings **1–5** via percentile ranks.
- **Book-Crossing:** only **rating > 0** kept for explicit-feedback alignment with RMSE.

### Takealot (`data/take-a-lot-dataset/`)

- `products.csv`, `reviews.csv`, `reviews/train.csv`, `reviews/test.csv`, etc.
- **Loader:** `standalone_takealot/takealot_loader.py`. Resolution order: explicit `--data-dir`, `TAKEALOT_DATA_DIR`, then `data/take-a-lot-dataset` / `data/takealot` relative to cwd or repo.

---

## How to run experiments

### Single public dataset (full pipeline: macro → user-centric → feedback)

```bash
python run_all.py --dataset ml-100k --test-size 0.2
```

`--dataset`: `ml-100k` | `ml-1m` | `lastfm-2k` | `book-crossing`.

### All four public datasets + combined CSVs and summary figures

```bash
python run_all_datasets.py
```

Re-aggregate only (no re-run):

```bash
python run_all_datasets.py --skip-exec
```

Subset:

```bash
python run_all_datasets.py --datasets ml-100k ml-1m
```

### Takealot

Uses **official** train/test CSVs for macro and user-centric experiments; feedback pool configurable.

```bash
python run_takealot_full.py --data-dir data/take-a-lot-dataset
```

- `--feedback-pool all` (default): pool from full `reviews.csv` for the simulation.
- `--feedback-pool train`: pool from training interactions only.

### Batch notebook execution

Executes `01`–`03` with `DATASET_NAME` injected; writes to `outputs/executed_notebooks/`:

```bash
python run_notebooks.py --dataset ml-100k
```

---

## Outputs

| Location | Content |
|----------|---------|
| **`outputs/results/`** | `macro_bias_metrics_<dataset>.csv`, `user_centric_metrics_<dataset>.csv`, `feedback_simulation_metrics_<dataset>.csv` |
| **`outputs/results/`** (batch) | `combined_macro_bias_metrics.csv`, `combined_user_centric_metrics.csv`, `combined_feedback_simulation_metrics.csv` |
| **`outputs/figures/`** | Per-dataset plots (log–log popularity vs rec frequency, extended metrics bars, ECDFs, segment RMSE bars, feedback diversity lines) and **combined** heatmaps / multi-dataset line plots |

Takealot result filenames use the dataset label **`take-a-lot-dataset`**.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_macro_bias.ipynb` | Macro-bias analysis for one `DATASET_NAME` |
| `02_user_centric.ipynb` | Segment RMSE / NDCG |
| `03_feedback_simulation.ipynb` | Feedback-loop experiment |
| `04_all_datasets.ipynb` | Multi-dataset / aggregation-oriented workflow |

Set `DATASET_NAME` in the config cell or rely on `run_notebooks.py`.

---

## Models and metrics (reference)

### Macro bias (`run_macro_bias`)

- **Gini:** inequality of multiset counts of how often each item appears in all users’ top‑K lists.
- **ARP:** mean training interaction count of recommended items (across all slots).
- **Catalog coverage:** fraction of training-catalog items that appear at least once in recommendations.
- **Spearman ρ:** correlation between per-item training popularity and recommendation frequency (zero if never recommended).
- **Average popularity percentile:** mean mainstream percentile of recommended items (higher ⇒ more head-heavy under the training distribution).
- **Long-tail share (20%):** fraction of slots whose item falls in the least mainstream 20% by that percentile definition.
- **Aggregate diversity:** number of distinct items recommended across all users.
- **Metadata token entropy:** Shannon entropy (bits) over lowercase whitespace tokens in recommended items’ `metadata_text`.
- **RMSE:** on held-out test ratings where the implementation supplies per-pair predictions (typically k-NN and SVD in the exported macro table).

### User-centric (`run_user_centric`)

- **Segments:** Niche / Diverse / Mainstream from tertiles of per-user mean training popularity of consumed items.
- **RMSE / NDCG@10** per segment; test relevance for NDCG is the set of items that user rated in the **test** split (binary relevance).

### Feedback simulation (`run_feedback_simulation`)

- **Top-K:** 5 per user per round; **5 rounds**.
- **Click probability:** `clip(0.7 * relevance + 0.3 * pop_norm, [0,1])` with Bernoulli draws; clicked pairs added as rating 5.0; duplicates on `(user, item)` resolved with **keep last**.

---

## Important methodological notes

1. **Segment RMSE for ALS, TF‑IDF, LogReg:** `run_user_centric` assigns the **global training mean** as the prediction for every test row when computing RMSE for these models, while NDCG uses **actual** top‑K lists. Therefore **niche, diverse, and mainstream RMSE are identical** within each of ALS / TF‑IDF / LogReg on a given dataset. **Differentiated segment RMSE** is only meaningful for **k-NN, SVD, and RandomForest** in the current code.

2. **Surprise optional:** Without scikit-surprise, k-NN and SVD paths use **`_FallbackCF`**. Reported metrics are **not** standard collaborative-filtering benchmarks in that mode.

3. **ALS:** Wrapped in try/except; failures may substitute trivial recommendations (e.g. first catalog items). Watch for log lines such as “ALS skipped due to error”.

4. **Complexity:** Surprise-style top‑K scores **all** unseen items per user; large catalogs (e.g. Takealot) are expensive.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `No module named 'surprise'` | `pip install scikit-surprise` or optional requirements file; on Windows you may need MSVC build tools if no wheel. |
| ALS errors or implausible ALS metrics | Check stderr; confirm `implicit` version; inspect matrix indexing logs. |
| Slow runs or memory errors | Use `ml-100k`; avoid running all models on huge catalogs until validated. |
| Takealot not found | `--data-dir` or `TAKEALOT_DATA_DIR`; folder must contain `products.csv`. |

---

## License and citation

- Obey **dataset licenses** (MovieLens, Book-Crossing, Last.fm / HetRec, Takealot).
- Academic and code reuse: follow your **institution** and supervisor guidelines.

For process questions, use your supervisor or institutional policies, not this document.
