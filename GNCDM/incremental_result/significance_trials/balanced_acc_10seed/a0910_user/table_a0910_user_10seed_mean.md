# a0910 user_split — 10-seed mean ± std + metric significance

Seeds `{1,2,3,5,7,11,13,21,42,84}`.
Primary endpoint: `ACC_overall = (n_old·ACC_old + n_new·ACC_new)/(n_old+n_new)` on **user_split query** interactions (support/query seed=7, frac=0.5): n_old=18677, n_new=8436, w_old=0.689.
`ACC/F1/RMSE_last` = `*_old`. RD column is RD (legacy TMD×100). Values in % as mean±std.

Formal-capable seed count: 10 >= 8.

| Paradigm | Model | ACC_overall ↑ | ACC_last ↑ | ACC_new ↑ | F1_last ↑ | F1_new ↑ | RMSE_last ↓ | RMSE_new ↓ | RD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor | G-NCDM | **70.24±0.54** | **70.24±0.54** | — | <u>79.65±0.69</u> | — | **45.22±0.29** | — | — |
| Ours | CLEAN-Full | 69.64±0.45 | **70.24±0.54** | <u>68.30±0.74*</u> | <u>79.65±0.69</u> | <u>77.65±0.91</u> | **45.22±0.29** | <u>45.42±0.20*</u> | **0.00±0.00** |
| Ours | CLEAN-LoRA | <u>70.14±0.45</u> | **70.24±0.54** | **69.92±0.40*** | <u>79.65±0.69</u> | **78.58±0.23*** | **45.22±0.29** | **44.76±0.22*** | **0.00±0.00** |
| Baselines | Full Replay Oracle | 68.99±0.48 | 69.52±0.77 | 67.82±0.61 | 78.93±0.89 | 77.63±0.55 | 45.64±0.31 | 45.99±0.23 | 3.44±1.04 |
| Baselines | EWC | 67.72±0.63 | 68.73±0.77 | 65.48±0.58 | 76.90±0.61 | 73.06±0.62 | 49.69±0.54 | 51.52±0.32 | 8.47±0.12 |
| Baselines | DER++ | 66.69±0.21 | 67.33±0.28 | 65.29±0.57 | 75.90±0.18 | 72.97±0.58 | 50.09±0.26 | 51.08±0.33 | 11.60±0.06 |
| Baselines | C-LoRA | 68.32±0.62 | 68.81±0.66 | 67.22±0.70 | 77.80±0.64 | 75.60±0.57 | 46.24±0.45 | 47.41±0.41 | 14.38±0.24 |
| Baselines | X-DER | 68.99±0.45 | <u>69.95±0.57</u> | 66.87±0.54 | **79.70±0.68** | 77.15±0.55 | <u>45.33±0.31</u> | 46.39±0.15 | <u>3.34±0.62</u> |
| Baselines | C-LoRA-GNCDM | 63.48±1.16 | 62.90±1.97 | 64.74±2.26 | 71.06±2.49 | 71.81±3.41 | 50.10±1.56 | 47.40±0.74 | 3.77±0.72 |
| Baselines | ICD | 55.09±0.00 | 51.34±0.00 | 63.39±0.00 | 52.62±0.00 | 71.31±0.00 | 50.38±0.00 | 48.89±0.00 | 412.16±0.00 |

## Significance vs CLEAN-Full (`ACC_overall`, Holm α=0.05)

`*` = CLEAN significantly better; `†` = significant but CLEAN worse; n.s. = not significant.

| baseline | Δ mean | wins | losses | p_raw | p_holm | reject |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| EWC | 0.01918 | 10 | 0 | 0.001953 | 0.01172 | * |
| DER++ | 0.02945 | 10 | 0 | 0.001953 | 0.01172 | * |
| C-LoRA | 0.01321 | 9 | 1 | 0.003906 | 0.01172 | * |
| X-DER | 0.006418 | 10 | 0 | 0.001953 | 0.01172 | * |
| C-LoRA-GNCDM | 0.06161 | 10 | 0 | 0.001953 | 0.01172 | * |
| ICD | 0.1455 | 10 | 0 | 0.001953 | 0.01172 | * |

CLEAN-Full mean ACC_overall = 0.6964 (sd=0.0045, n=10).

## Significance vs baselines (`ACC/F1/RMSE` last+new)

Targets: `CLEAN-Full`, `CLEAN-LoRA`. Comparators: `EWC`, `DER++`, `C-LoRA`, `X-DER`, `C-LoRA-GNCDM`, `ICD` (Full-Replay oracle excluded).
Exact two-sided paired sign-flip permutation test; Holm correction **within each metric** over 6 baselines. Formal-capable seeds: 10 >= 8.
`*` on a CLEAN cell = significantly better than **all** 6 baselines on that metric (`reject_holm` and oriented Δ>0).

| target | metric | starred | n_sig_better | n_sig_worse | n.s. |
| --- | --- | --- | ---: | ---: | ---: |
| CLEAN-Full | ACC_last | False | 5 | 0 | 1 |
| CLEAN-Full | ACC_new | True | 6 | 0 | 0 |
| CLEAN-Full | F1_last | False | 5 | 0 | 1 |
| CLEAN-Full | F1_new | False | 5 | 0 | 1 |
| CLEAN-Full | RMSE_last | False | 5 | 0 | 1 |
| CLEAN-Full | RMSE_new | True | 6 | 0 | 0 |
| CLEAN-LoRA | ACC_last | False | 5 | 0 | 1 |
| CLEAN-LoRA | ACC_new | True | 6 | 0 | 0 |
| CLEAN-LoRA | F1_last | False | 5 | 0 | 1 |
| CLEAN-LoRA | F1_new | True | 6 | 0 | 0 |
| CLEAN-LoRA | RMSE_last | False | 5 | 0 | 1 |
| CLEAN-LoRA | RMSE_new | True | 6 | 0 | 0 |

### Pairwise detail

| target | metric | baseline | Δ mean | wins | losses | p_raw | p_holm | reject |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CLEAN-Full | ACC_last | EWC | 0.0151 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_last | DER++ | 0.02915 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_last | C-LoRA | 0.01432 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_last | X-DER | 0.002864 | 8 | 2 | 0.06836 | 0.06836 | n.s. |
| CLEAN-Full | ACC_last | C-LoRA-GNCDM | 0.07336 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_last | ICD | 0.189 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_new | EWC | 0.02822 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_new | DER++ | 0.03013 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_new | C-LoRA | 0.01076 | 9 | 1 | 0.01758 | 0.01758 | * |
| CLEAN-Full | ACC_new | X-DER | 0.01428 | 9 | 1 | 0.003906 | 0.01172 | * |
| CLEAN-Full | ACC_new | C-LoRA-GNCDM | 0.03561 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | ACC_new | ICD | 0.04912 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_last | EWC | 0.02752 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_last | DER++ | 0.03746 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_last | C-LoRA | 0.01846 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_last | X-DER | -0.0005057 | 4 | 6 | 0.7676 | 0.7676 | n.s. |
| CLEAN-Full | F1_last | C-LoRA-GNCDM | 0.08586 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_last | ICD | 0.2703 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_new | EWC | 0.04594 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_new | DER++ | 0.04679 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_new | C-LoRA | 0.02053 | 9 | 1 | 0.003906 | 0.01172 | * |
| CLEAN-Full | F1_new | X-DER | 0.005023 | 8 | 2 | 0.207 | 0.207 | n.s. |
| CLEAN-Full | F1_new | C-LoRA-GNCDM | 0.05838 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | F1_new | ICD | 0.06336 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_last | EWC | 0.04477 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_last | DER++ | 0.04875 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_last | C-LoRA | 0.01021 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_last | X-DER | 0.001134 | 6 | 4 | 0.1738 | 0.1738 | n.s. |
| CLEAN-Full | RMSE_last | C-LoRA-GNCDM | 0.0488 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_last | ICD | 0.05166 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | EWC | 0.06099 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | DER++ | 0.05662 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | C-LoRA | 0.01988 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | X-DER | 0.009717 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | C-LoRA-GNCDM | 0.01982 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-Full | RMSE_new | ICD | 0.03469 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_last | EWC | 0.0151 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_last | DER++ | 0.02915 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_last | C-LoRA | 0.01432 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_last | X-DER | 0.002864 | 8 | 2 | 0.06836 | 0.06836 | n.s. |
| CLEAN-LoRA | ACC_last | C-LoRA-GNCDM | 0.07336 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_last | ICD | 0.189 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | EWC | 0.04445 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | DER++ | 0.04636 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | C-LoRA | 0.02699 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | X-DER | 0.03051 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | C-LoRA-GNCDM | 0.05184 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | ACC_new | ICD | 0.06535 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_last | EWC | 0.02752 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_last | DER++ | 0.03746 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_last | C-LoRA | 0.01846 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_last | X-DER | -0.0005057 | 4 | 6 | 0.7676 | 0.7676 | n.s. |
| CLEAN-LoRA | F1_last | C-LoRA-GNCDM | 0.08586 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_last | ICD | 0.2703 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | EWC | 0.05524 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | DER++ | 0.05609 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | C-LoRA | 0.02983 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | X-DER | 0.01432 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | C-LoRA-GNCDM | 0.06768 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | F1_new | ICD | 0.07266 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_last | EWC | 0.04477 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_last | DER++ | 0.04875 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_last | C-LoRA | 0.01021 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_last | X-DER | 0.001134 | 6 | 4 | 0.1738 | 0.1738 | n.s. |
| CLEAN-LoRA | RMSE_last | C-LoRA-GNCDM | 0.0488 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_last | ICD | 0.05166 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | EWC | 0.06754 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | DER++ | 0.06318 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | C-LoRA | 0.02643 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | X-DER | 0.01627 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | C-LoRA-GNCDM | 0.02637 | 10 | 0 | 0.001953 | 0.01172 | * |
| CLEAN-LoRA | RMSE_new | ICD | 0.04124 | 10 | 0 | 0.001953 | 0.01172 | * |

Files under `d:/CD_continue/GNCDM/incremental_result/significance_trials/acc_overall_10seed_a0910_user`.
