# Overall ACC (pooled old+new test interactions)

Definition: `ACC_overall = (n_old · ACC_old + n_new · ACC_new) / (n_old + n_new)` on the **test** split, using fixed interaction counts from the random-split bipartition.

| Dataset | n_old | n_new | w_old | w_new |
|---|---:|---:|---:|---:|
| math1 | 10901 | 5935 | 0.647 | 0.353 |
| junyi | 13997 | 6398 | 0.686 | 0.314 |
| a0910 | 37642 | 16836 | 0.691 | 0.309 |

Values are **mean ± std** over 10 seeds, reported in %.
`Balanced_ACC = 0.7 · ACC_old + 0.3 · ACC_new` shown for comparison (fixed weights, not the same as pooled overall).
Anchor has no new-item eval; its ACC_overall = ACC_old only.

| Model | Junyi ACC_overall | Junyi Balanced_ACC | a0910 ACC_overall | a0910 Balanced_ACC | Math1 ACC_overall | Math1 Balanced_ACC |
|---|---|---|---|---|---|---|
| G-NCDM | 78.40±0.15 | — | 73.51±0.15 | — | 72.14±0.51 | — |
| CLEAN-Full | 77.19±0.11 | 77.24±0.11 | 73.11±0.15 | 73.12±0.15 | 72.18±1.62 | 72.17±1.38 |
| CLEAN-LoRA | 76.89±0.19 | 76.95±0.18 | 73.08±0.17 | 73.09±0.17 | 70.85±1.01 | 71.05±0.87 |
| Full Replay Oracle | 77.58±0.18 | 77.62±0.18 | 73.45±0.17 | 73.46±0.17 | 73.23±0.17 | 73.08±0.18 |
| EWC | 71.56±1.41 | 71.56±1.44 | 66.01±0.62 | 66.04±0.62 | 70.24±0.53 | 70.00±0.52 |
| DER++ | 75.82±0.22 | 75.85±0.22 | 68.93±0.28 | 68.94±0.28 | 73.06±0.33 | 72.86±0.36 |
| C-LoRA | 71.84±1.34 | 71.82±1.37 | 66.61±0.23 | 66.64±0.23 | 69.97±0.46 | 69.81±0.46 |
| X-DER | 76.21±0.23 | 76.27±0.24 | 71.92±0.10 | 71.95±0.10 | 73.45±0.16 | 73.32±0.16 |
| ICD | 69.02±0.40 | 69.06±0.32 | 57.67±0.00 | 57.58±0.00 | 69.49±0.00 | 69.36±0.00 |
| C-LoRA-GNCDM | 69.66±5.01 | 69.55±5.11 | 70.45±1.73 | 70.44±1.72 | 65.76±1.02 | 65.32±1.09 |

Files: `acc_overall_10seed_mean.csv`, `acc_overall_10seed_per_seed.csv`, `acc_overall_10seed_means_long.csv`.

## a0910 user_split

Definition same as above, but on **user_split query** interactions (support/query seed=7, frac=0.5): n_old=18677, n_new=8436, w_old=0.689.

| Model | a0910 user ACC_overall | a0910 user Balanced_ACC |
|---|---|---|
| G-NCDM | 70.24±0.54 | — |
| CLEAN-Full | 69.64±0.45 | 69.66±0.45 |
| CLEAN-LoRA | 70.14±0.45 | 70.15±0.45 |
| Full Replay Oracle | 68.99±0.48 | 69.01±0.49 |
| EWC | 67.72±0.63 | 67.75±0.64 |
| DER++ | 66.69±0.21 | 66.71±0.21 |
| C-LoRA | 68.32±0.62 | 68.33±0.62 |
| X-DER | 68.99±0.45 | 69.03±0.46 |
| ICD | 55.09±0.00 | 54.95±0.00 |
| C-LoRA-GNCDM | 63.48±1.16 | 63.45±1.18 |

Source: `a0910_user_per_seed_merged.csv`; profile dir: `a0910_user/`.
