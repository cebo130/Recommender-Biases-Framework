# Thesis Research — Recommender bias, fairness & feedback loops

Python experiments comparing **six recommenders** (k-NN, SVD, ALS, TF‑IDF, per-user LogReg, Random Forest) on **MovieLens**, **Last.fm**, **Book-Crossing**, and **Takealot** reviews. Measures **macro popularity bias**, **user-segment RMSE/NDCG**, and a **synthetic feedback simulation** (SVD vs TF‑IDF).

**Full documentation:** [docs/PROJECT.md](docs/PROJECT.md)  
**Thesis write-up:** `thesis.tex` + `references.bib` (see PROJECT.md for LaTeX build)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# optional: pip install -r requirements_optional_windows_build.txt   # scikit-surprise
```

Place datasets under `data/` (see PROJECT.md), then:

```bash
python run_all.py --dataset ml-100k --test-size 0.2
```

All four public benchmarks + combined CSVs/figures:

```bash
python run_all_datasets.py
```

Takealot (official train/test split):

```bash
python run_takealot_full.py --data-dir data/take-a-lot-dataset
```

Results: `outputs/results/*.csv`, figures: `outputs/figures/*.png`.

## Layout (short)

| Path | Role |
|------|------|
| `run_all.py` | One dataset: macro + user-centric + feedback |
| `run_all_datasets.py` | Batch public datasets + `combined_*` outputs |
| `run_takealot_full.py` | Takealot pipeline |
| `src/utils.py` | Loaders, k-core, metadata |
| `notebooks/` | Jupyter workflows `01`–`04` |
| `standalone_takealot/` | Takealot loader |
| `thesis.tex`, `references.bib` | LaTeX chapter |
| `build_thesis.ps1` | Windows helper to find `pdflatex` |

## Important caveats

- **Surprise** (k-NN/SVD) is optional; without it, a **bias-only fallback** is used — not comparable to standard Surprise results.
- **Segment RMSE** for ALS / TF‑IDF / LogReg uses the **global mean** placeholder; only k-NN, SVD, and RandomForest get **real** per-row RMSE by segment. NDCG still uses real top‑K lists.

## License

Respect upstream **dataset licenses**. Thesis/code reuse per your institution.
