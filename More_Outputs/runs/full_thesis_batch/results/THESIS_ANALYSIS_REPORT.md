# Thesis analysis — auto-generated from experiment CSVs

_Generated: 2026-04-11 01:51 UTC_  
_Results directory: `C:\Users\29915759\Documents\Thesis_Research\More_Outputs\runs\full_thesis_batch\results`_

---

## 1. Rerank sensitivity (λ × pool_k)

### MovieLens 100K
Popularity-penalty rerank only: **higher λ** weights predicted score more and train popularity less in the pool. **Larger pool_k** widens the candidate set before re-ranking.
| pool_k | rerank_lambda | model | gini | long_tail_share_20pct | spearman_pop_vs_recfreq | avg_popularity_percentile | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.5000 | SVD+Rerank_p30_l0.5 | 0.9536 | 0.3494 | 0.3145 | 0.2820 | 0.2144 |
| 30 | 0.6500 | SVD+Rerank_p30_l0.65 | 0.9532 | 0.4535 | 0.3679 | 0.2358 | 0.2188 |
| 30 | 0.8000 | SVD+Rerank_p30_l0.8 | 0.9534 | 0.5192 | 0.3893 | 0.2094 | 0.2214 |
| 30 | 0.9500 | SVD+Rerank_p30_l0.95 | 0.9544 | 0.5623 | 0.4043 | 0.1921 | 0.2170 |
| 50 | 0.5000 | SVD+Rerank_p50_l0.5 | 0.9538 | 0.3044 | 0.2869 | 0.3046 | 0.2127 |
| 50 | 0.6500 | SVD+Rerank_p50_l0.65 | 0.9529 | 0.4245 | 0.3488 | 0.2505 | 0.2205 |
| 50 | 0.8000 | SVD+Rerank_p50_l0.8 | 0.9527 | 0.5068 | 0.3768 | 0.2166 | 0.2222 |
| 50 | 0.9500 | SVD+Rerank_p50_l0.95 | 0.9537 | 0.5602 | 0.3887 | 0.1954 | 0.2179 |
| 100 | 0.5000 | SVD+Rerank_p100_l0.5 | 0.9543 | 0.2360 | 0.2518 | 0.3381 | 0.2092 |
| 100 | 0.6500 | SVD+Rerank_p100_l0.65 | 0.9527 | 0.3838 | 0.3238 | 0.2696 | 0.2205 |
| 100 | 0.8000 | SVD+Rerank_p100_l0.8 | 0.9525 | 0.4872 | 0.3693 | 0.2255 | 0.2266 |
| 100 | 0.9500 | SVD+Rerank_p100_l0.95 | 0.9533 | 0.5558 | 0.3777 | 0.1982 | 0.2214 |

**Quick ranges (by pool_k):**
- **pool_k = 30**: `long_tail_share_20pct` ∈ [0.3494, 0.5623]; `gini` ∈ [0.9532, 0.9544]; `spearman_pop_vs_recfreq` ∈ [0.3145, 0.4043]
- **pool_k = 50**: `long_tail_share_20pct` ∈ [0.3044, 0.5602]; `gini` ∈ [0.9527, 0.9538]; `spearman_pop_vs_recfreq` ∈ [0.2869, 0.3887]
- **pool_k = 100**: `long_tail_share_20pct` ∈ [0.2360, 0.5558]; `gini` ∈ [0.9525, 0.9543]; `spearman_pop_vs_recfreq` ∈ [0.2518, 0.3777]

### Takealot
Popularity-penalty rerank only: **higher λ** weights predicted score more and train popularity less in the pool. **Larger pool_k** widens the candidate set before re-ranking.
| pool_k | rerank_lambda | model | gini | long_tail_share_20pct | spearman_pop_vs_recfreq | avg_popularity_percentile | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.5000 | SVD+Rerank_p30_l0.5 | 0.9480 | 0.5566 | 0.2416 | 0.2425 | 0.2078 |
| 30 | 0.6500 | SVD+Rerank_p30_l0.65 | 0.9524 | 0.5699 | 0.2491 | 0.2348 | 0.1963 |
| 30 | 0.8000 | SVD+Rerank_p30_l0.8 | 0.9549 | 0.5772 | 0.2516 | 0.2306 | 0.1899 |
| 30 | 0.9500 | SVD+Rerank_p30_l0.95 | 0.9566 | 0.5823 | 0.2538 | 0.2276 | 0.1853 |
| 50 | 0.5000 | SVD+Rerank_p50_l0.5 | 0.9287 | 0.4775 | 0.1802 | 0.3111 | 0.2547 |
| 50 | 0.6500 | SVD+Rerank_p50_l0.65 | 0.9348 | 0.4941 | 0.1845 | 0.3015 | 0.2426 |
| 50 | 0.8000 | SVD+Rerank_p50_l0.8 | 0.9383 | 0.5034 | 0.1859 | 0.2962 | 0.2356 |
| 50 | 0.9500 | SVD+Rerank_p50_l0.95 | 0.9406 | 0.5094 | 0.1856 | 0.2927 | 0.2311 |
| 100 | 0.5000 | SVD+Rerank_p100_l0.5 | 0.9154 | 0.4447 | 0.0801 | 0.3498 | 0.2877 |
| 100 | 0.6500 | SVD+Rerank_p100_l0.65 | 0.9237 | 0.4666 | 0.0790 | 0.3374 | 0.2736 |
| 100 | 0.8000 | SVD+Rerank_p100_l0.8 | 0.9285 | 0.4785 | 0.0753 | 0.3305 | 0.2657 |
| 100 | 0.9500 | SVD+Rerank_p100_l0.95 | 0.9317 | 0.4864 | 0.0720 | 0.3258 | 0.2599 |

**Quick ranges (by pool_k):**
- **pool_k = 30**: `long_tail_share_20pct` ∈ [0.5566, 0.5823]; `gini` ∈ [0.9480, 0.9566]; `spearman_pop_vs_recfreq` ∈ [0.2416, 0.2538]
- **pool_k = 50**: `long_tail_share_20pct` ∈ [0.4775, 0.5094]; `gini` ∈ [0.9287, 0.9406]; `spearman_pop_vs_recfreq` ∈ [0.1802, 0.1859]
- **pool_k = 100**: `long_tail_share_20pct` ∈ [0.4447, 0.4864]; `gini` ∈ [0.9154, 0.9317]; `spearman_pop_vs_recfreq` ∈ [0.0720, 0.0801]

## 2. Same-pool baselines vs popularity-penalty rerank

### Same-pool mitigation contrast (ml 100k)
All pool strategies use the **same SVD top-pool** (pool_k = 50 for baselines and primary rerank). **SVD+Rerank** uses the popularity-penalty score; **PoolPopDesc / PoolPopAsc** sort by train popularity; **PoolMMR** trades relevance for diversity in embedding space (TF–IDF cosine).

| model | gini | long_tail_share_20pct | spearman_pop_vs_recfreq | avg_popularity_percentile | catalog_coverage |
| --- | --- | --- | --- | --- | --- |
| SVD | 0.9552 | 0.5903 | 0.4339 | 0.1809 | 0.2092 |
| SVD+PoolMMR | 0.9497 | 0.5038 | 0.4169 | 0.2244 | 0.2049 |
| SVD+PoolPopAsc | 0.9647 | 0.0000 | -0.1412 | 0.5622 | 0.1450 |
| SVD+PoolPopDesc | 0.9702 | 1.0000 | 0.5295 | 0.0348 | 0.1102 |
| SVD+Rerank | 0.9527 | 0.5068 | 0.3768 | 0.2166 | 0.2222 |

### Same-pool mitigation contrast (take a lot dataset)
All pool strategies use the **same SVD top-pool** (pool_k = 50 for baselines and primary rerank). **SVD+Rerank** uses the popularity-penalty score; **PoolPopDesc / PoolPopAsc** sort by train popularity; **PoolMMR** trades relevance for diversity in embedding space (TF–IDF cosine).

| model | gini | long_tail_share_20pct | spearman_pop_vs_recfreq | avg_popularity_percentile | catalog_coverage |
| --- | --- | --- | --- | --- | --- |
| SVD | 0.9818 | 0.8215 | 0.2908 | 0.1015 | 0.1191 |
| SVD+PoolMMR | 0.9717 | 0.7706 | 0.2694 | 0.1306 | 0.1534 |
| SVD+PoolPopAsc | 0.8401 | 0.1797 | 0.0326 | 0.4709 | 0.3896 |
| SVD+PoolPopDesc | 0.9981 | 1.0000 | 0.2035 | 0.0048 | 0.0092 |
| SVD+Rerank | 0.9383 | 0.5034 | 0.1859 | 0.2962 | 0.2356 |

## 3. Scale: 100K vs 1M

### MovieLens 100K vs 1M (scale check)
Use this block for a **short robustness subsection**: same metrics, larger catalog and user base on 1M. Directions of bias (e.g. Spearman, long-tail share) should be interpreted comparatively, not as identical magnitudes.

| model | ml-100k_gini | ml-1m_gini | ml-100k_long_tail_share_20pct | ml-1m_long_tail_share_20pct | ml-100k_spearman_pop_vs_recfreq | ml-1m_spearman_pop_vs_recfreq | ml-100k_avg_popularity_percentile | ml-1m_avg_popularity_percentile | ml-100k_catalog_coverage | ml-1m_catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVD | 0.9552 | 0.9568 | 0.5903 | 0.6882 | 0.4339 | 0.5180 | 0.1809 | 0.1923 | 0.2092 | 0.2564 |
| k-NN | 0.9800 | 0.9885 | 0.6420 | 0.6156 | 0.3407 | 0.2854 | 0.1651 | 0.2244 | 0.0851 | 0.0709 |
| TF-IDF | 0.9254 | 0.9370 | 0.1556 | 0.2869 | 0.0739 | 0.2576 | 0.5091 | 0.4058 | 0.3220 | 0.3620 |
| ALS | 0.9874 | 0.9958 | 0.9389 | 0.9993 | 0.2557 | 0.2015 | 0.0693 | 0.0432 | 0.0330 | 0.0187 |

## 4. Fairness segments — statistical tests

### Niche vs Mainstream — tests (ml 100k)
**Mann–Whitney U** on per-user RMSE and NDCG@10 (Niche vs Mainstream users). `sig_rmse_mwu_005` / `sig_ndcg_mwu_005` flag p < 0.05 (two-sided) when present in your pipeline output.

| model | rmse_niche | rmse_mainstream | p_mwu_rmse_niche_vs_mainstream | sig_rmse_mwu_005 | ndcg_niche | ndcg_mainstream | p_mwu_ndcg_niche_vs_mainstream | sig_ndcg_mwu_005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALS | 1.13539 | 1.07213 | 0.0036021 | True | 0.170873 | 0.0726599 | 2.60639e-14 | True |
| RandomForest | 1.04419 | 1.02007 | 0.00428988 | True | 0.10438 | 0.0363178 | 2.40442e-15 | True |
| SVD | 0.934604 | 0.929448 | 0.126988 | False | 0.14653 | 0.0435064 | 2.63601e-17 | True |
| SVD+PoolMMR | 0.934604 | 0.929448 | 0.126988 | False | 0.144426 | 0.0391733 | 1.84915e-18 | True |
| SVD+PoolPopAsc | 0.934604 | 0.929448 | 0.126988 | False | 0.0211162 | 0.00270752 | 8.61587e-11 | True |
| SVD+PoolPopDesc | 0.934604 | 0.929448 | 0.126988 | False | 0.247427 | 0.0981687 | 2.26103e-16 | True |
| SVD+Rerank | 0.934604 | 0.929448 | 0.126988 | False | 0.126378 | 0.0360363 | 6.63444e-17 | True |
| TF-IDF | 1.13539 | 1.07213 | 0.0036021 | True | 0.065951 | 0.0233623 | 6.70492e-13 | True |
| k-NN | 1.02349 | 0.975496 | 0.00125351 | True | 0.151427 | 0.0415562 | 2.13944e-17 | True |

**One-sentence template for the thesis:**

> For **ml 100k**, differences in per-user RMSE between Niche and Mainstream segments were assessed with a Mann–Whitney U test; e.g. **SVD+Rerank** yields p = 0.127 for RMSE (see table). Repeat for NDCG using `p_mwu_ndcg_niche_vs_mainstream`.

### Niche vs Mainstream — tests (ml 1m)
**Mann–Whitney U** on per-user RMSE and NDCG@10 (Niche vs Mainstream users). `sig_rmse_mwu_005` / `sig_ndcg_mwu_005` flag p < 0.05 (two-sided) when present in your pipeline output.

| model | rmse_niche | rmse_mainstream | p_mwu_rmse_niche_vs_mainstream | sig_rmse_mwu_005 | ndcg_niche | ndcg_mainstream | p_mwu_ndcg_niche_vs_mainstream | sig_ndcg_mwu_005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALS | 1.14708 | 1.07481 | 5.41271e-11 | True | 0.139604 | 0.050809 | 2.52014e-44 | True |
| RandomForest | 1.01908 | 0.967095 | 2.39882e-20 | True | 0.0604396 | 0.0250711 | 9.58523e-40 | True |
| SVD | 0.874022 | 0.877237 | 1.77497e-05 | True | 0.127449 | 0.0572317 | 4.26371e-34 | True |
| TF-IDF | 1.14708 | 1.07481 | 5.41271e-11 | True | 0.0697283 | 0.0327151 | 1.34545e-35 | True |
| k-NN | 0.997471 | 0.949268 | 9.93852e-16 | True | 0.0791628 | 0.0231017 | 6.71847e-53 | True |

**One-sentence template for the thesis:**

> For **ml 1m**, differences in per-user RMSE between Niche and Mainstream segments were assessed with a Mann–Whitney U test; e.g. **SVD+Rerank** yields p = N/A for RMSE (see table). Repeat for NDCG using `p_mwu_ndcg_niche_vs_mainstream`.

---

## 5. Takealot (domain note)

Use **Takealot** results for **domain shift** and **catalog scale** discussion: full-catalog scoring is costly; interpret exposure metrics in light of sparsity and item cardinality. Macro and user-centric tables for `take-a-lot-dataset` are in the same `results/` folder.

