# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory five-seed analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 7 planned hypotheses at alpha=0.05. At least 9 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | EWC | 5 | 0.07393 | 0.06903 | 0.07853 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | DER++ | 5 | 0.05342 | 0.04937 | 0.05765 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | C-LoRA | 5 | 0.07393 | 0.07156 | 0.07616 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | X-DER | 5 | 0.02498 | 0.02126 | 0.02838 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.02276 | 0.01187 | 0.04177 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | ICD | 5 | 0.1189 | 0.1176 | 0.1202 | 0.0625 | 0.4375 | False |
| a0910 | random | ec6331b45abd69ac | Balanced_AUC | CLEAN-Full | CLEAN-LoRA | 5 | 0.003919 | 0.002558 | 0.00528 | 0.0625 | 0.4375 | False |
