# Random-split results (10-seed mean ± std)

Means ± sample standard deviation over seeds `{1, 2, 3, 5, 7, 11, 13, 21, 42, 84}` (n=10). Layout mirrors the paper main table; `BD` = `TMD` (both ×100, %).

| Paradigm | Model | Junyi |  |  |  |  |  |  | a0910 |  |  |  |  |  |  | Math1 |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | ACC_old ↑ | ACC_new ↑ | F1_old ↑ | F1_new ↑ | RMSE_old ↓ | RMSE_new ↓ | BD ↓ | ACC_old ↑ | ACC_new ↑ | F1_old ↑ | F1_new ↑ | RMSE_old ↓ | RMSE_new ↓ | BD ↓ | ACC_old ↑ | ACC_new ↑ | F1_old ↑ | F1_new ↑ | RMSE_old ↓ | RMSE_new ↓ | BD ↓ |
| Anchor | G-NCDM | <u>78.40±0.15</u> | — | <u>85.53±0.14</u> | — | <u>38.72±0.05</u> | — | — | <u>73.51±0.15</u> | — | <u>81.35±0.26</u> | — | **42.68±0.12** | — | — | 72.14±0.51 | — | 71.72±1.65 | — | 42.43±0.42 | — | — |
| Ours | CLEAN-Full | <u>78.40±0.15</u> | <u>74.53±0.30</u> | <u>85.53±0.14</u> | <u>81.10±0.13</u> | <u>38.72±0.05</u> | 41.83±0.11 | **0.00±0.00** | <u>73.51±0.15</u> | <u>72.21±0.19</u> | <u>81.35±0.26</u> | <u>79.61±0.20</u> | **42.68±0.12** | <u>43.45±0.11</u> | **0.00±0.00** | 72.14±0.51 | 72.25±4.69 | 71.72±1.65 | 57.53±2.75 | 42.43±0.42 | 45.64±2.39 | **0.00±0.00** |
| Ours | CLEAN-LoRA | <u>78.40±0.15</u> | 73.57±0.64 | <u>85.53±0.14</u> | 80.65±0.39 | <u>38.72±0.05</u> | 42.43±0.44 | **0.00±0.00** | <u>73.51±0.15</u> | 72.12±0.24 | <u>81.35±0.26</u> | 79.55±0.40 | **42.68±0.12** | 43.80±0.23 | **0.00±0.00** | 72.14±0.51 | 68.50±3.00 | 71.72±1.65 | 54.73±2.50 | 42.43±0.42 | 47.73±1.59 | **0.00±0.00** |
| Baselines | Full Replay Oracle | **78.55±0.25** | **75.47±0.22** | **85.74±0.17** | **81.49±0.26** | **38.71±0.12** | **40.86±0.09** | 4.88±1.29 | **73.66±0.12** | **72.99±0.33** | **81.58±0.21** | **80.20±0.45** | <u>42.77±0.10</u> | **43.08±0.12** | 6.92±0.72 | <u>72.20±0.28</u> | <u>75.12±0.28</u> | 71.94±0.85 | <u>66.63±0.98</u> | <u>42.18±0.31</u> | 39.66±0.28 | 6.47±1.60 |
| Baselines | EWC | 71.54±2.03 | 71.58±0.40 | 79.18±2.05 | 78.27±0.52 | 44.83±1.34 | 43.98±0.21 | 8.92±0.20 | 66.87±0.67 | 64.09±0.61 | 74.79±0.91 | 71.84±0.72 | 52.50±0.38 | 53.88±0.41 | 9.01±0.09 | 68.59±0.51 | 73.27±0.86 | 66.01±1.63 | 61.64±1.55 | 50.30±0.45 | 43.11±0.68 | 9.70±0.26 |
| Baselines | DER++ | 76.53±0.28 | 74.27±0.26 | 83.66±0.24 | 80.44±0.25 | 40.00±0.25 | <u>41.61±0.13</u> | <u>3.09±1.14</u> | 69.50±0.44 | 67.66±0.51 | 77.74±0.53 | 76.34±0.40 | 45.08±0.42 | 46.25±0.26 | <u>3.64±0.94</u> | 71.72±0.52 | **75.52±0.22** | <u>72.87±1.36</u> | **67.02±1.14** | 42.56±0.36 | **39.00±0.10** | <u>1.32±1.11</u> |
| Baselines | C-LoRA | 71.26±2.01 | 73.12±0.36 | 79.41±1.74 | 79.32±0.34 | 44.10±1.31 | 42.93±0.21 | 19.69±0.27 | 67.64±0.28 | 64.28±0.46 | 75.79±0.35 | 72.01±0.48 | 52.13±0.18 | 53.94±0.35 | 13.11±0.09 | 68.88±0.51 | 71.98±0.68 | 66.45±1.61 | 59.86±1.08 | 50.28±0.46 | 46.05±0.58 | 10.82±0.27 |
| Baselines | X-DER | 77.59±0.42 | 73.19±0.69 | 85.28±0.21 | 79.73±0.50 | 39.60±0.32 | 42.54±0.24 | 6.36±1.06 | 72.86±0.24 | 69.83±0.44 | 81.18±0.28 | 78.26±0.24 | 43.02±0.21 | 44.79±0.26 | 7.32±1.57 | **72.54±0.21** | 75.12±0.35 | **73.45±1.47** | 66.10±1.02 | **42.15±0.34** | <u>39.07±0.19</u> | 7.24±2.00 |
| Baselines | ICD | 70.01±1.47 | 66.84±4.50 | 70.74±8.29 | 64.20±13.46 | 45.27±1.64 | 47.61±0.44 | 119.02±24.52 | 54.40±0.00 | 65.00±0.00 | 59.20±0.00 | 75.10±0.00 | 49.16±0.00 | 47.28±0.00 | 422.42±0.00 | 68.62±0.00 | 71.10±0.00 | 62.88±0.00 | 51.43±0.00 | 46.82±0.00 | 47.19±0.00 | 95.75±0.00 |

## Notes

1. All values are reported in percentage (%) as **mean ± std** (sample standard deviation, ddof=1) across 10 random seeds.
2. **Bold** = best mean in column; <u>underline</u> = second-best mean (Anchor contributes only to *_last).
3. Higher is better for ACC/F1; lower is better for RMSE/BD (BD = RD).
4. Full-Replay Oracle is a theoretical upper bound (trained/reported, not used as a significance comparator).
5. **Significance (primary endpoint ACC_overall)** — exact two-sided paired sign-flip permutation test, Holm-corrected within each profile over 6 baselines {EWC, DER++, C-LoRA, X-DER, C-LoRA-GNCDM, ICD}, α=0.05, n=10. CLEAN-LoRA and Full-Replay excluded. * = CLEAN-Full significantly better; † = significant but CLEAN-Full worse; n.s. = not significant.

| Dataset / split | EWC | DER++ | C-LoRA | X-DER | C-LoRA-GNCDM | ICD |
|---|---|---|---|---|---|---|
| Junyi random | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) |
| a0910 random | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) |
| Math1 random | * (p_holm=0.02344) | n.s. (p_holm=0.06836) | * (p_holm=0.02344) | † (p_holm=0.02344) | * (p_holm=0.01172) | * (p_holm=0.01953) |
| a0910 user | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) | * (p_holm=0.01172) |

### Verdict

- **Junyi / a0910 random / a0910 user**: CLEAN-Full significantly outperforms all 6 baselines on ACC_overall (all *).
- **Math1 random**: CLEAN-Full significantly better than EWC, C-LoRA, C-LoRA-GNCDM, ICD; n.s. vs DER++; significantly worse than X-DER (†).

Source CSVs: *_random_per_seed_merged.csv, 0910_user_per_seed_merged.csv; tests: ormal_significance_acc_tests_all.csv.
User-split paper table: 	able_a0910_user_10seed_mean.md (also under 0910_user/).
