# Experiments and thesis analysis (this repo)

## Full unattended batch

From the repository root:

```powershell
Set-Location "c:\Users\29915759\Documents\Thesis_Research"
python run_overnight_batch.py --run-label full_thesis_batch
```

- Runs **without** `--fast`, so you get **SVD pool baselines** (PopDesc, PopAsc, MMR), **`svd_rerank_sensitivity_*.csv`** (full λ × pool_k grid) on **ml-100k** and **take-a-lot-dataset**, plus **`run_all_datasets`** (including **ml-1m**, Last.fm, Book-Crossing).
- Artifacts go under **`More_Outputs/runs/full_thesis_batch/`** (unless `THESIS_OUTPUT_PARENT` overrides).
- **Disable OS sleep** on AC power so the machine does not suspend mid-run.

## Auto-generated analysis draft

When the batch finishes, **`THESIS_ANALYSIS_REPORT.md`** is written to the run’s **`results/`** folder. It includes:

1. Sensitivity tables and short range summaries (Gini, long-tail share, Spearman ρ).
2. Same-pool **SVD+Rerank** vs **PopDesc / PopAsc / MMR** macro metrics.
3. **ml-100k vs ml-1m** comparison table for selected models.
4. **Niche vs Mainstream** Mann–Whitney p-values and a one-sentence template.

Regenerate anytime (e.g. after copying CSVs):

```powershell
python generate_thesis_analysis_report.py --run-root More_Outputs\runs\full_thesis_batch
```

Or pick the newest run automatically:

```powershell
python generate_thesis_analysis_report.py --latest
```

## Using this in the dissertation

- Treat **MovieLens 100K** as the primary narrative (figures and prose).
- Use **1M** as a **scale / robustness** paragraph with the generated comparison table.
- Use **Takealot** for **domain and catalog-scale** discussion; acknowledge computational cost where relevant.
