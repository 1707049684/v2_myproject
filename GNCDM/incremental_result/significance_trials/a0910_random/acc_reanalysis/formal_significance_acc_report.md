# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 6 planned hypotheses at alpha=0.05. At least 8 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | EWC | 5 | 0.06901 | 0.06456 | 0.07302 | 0.0625 | 0.375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | DER++ | 5 | 0.04071 | 0.03856 | 0.04261 | 0.0625 | 0.375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | C-LoRA | 5 | 0.06571 | 0.06365 | 0.06791 | 0.0625 | 0.375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | X-DER | 5 | 0.01179 | 0.01051 | 0.01294 | 0.0625 | 0.375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.03434 | 0.02197 | 0.05526 | 0.0625 | 0.375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_ACC | CLEAN-Full | ICD | 5 | 0.1554 | 0.1541 | 0.1566 | 0.0625 | 0.375 | False |
