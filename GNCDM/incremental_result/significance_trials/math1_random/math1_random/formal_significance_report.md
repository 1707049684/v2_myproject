# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory five-seed analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 7 planned hypotheses at alpha=0.05. At least 9 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | EWC | 5 | -0.03944 | -0.04786 | -0.0276 | 0.0625 | 0.4375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | DER++ | 5 | -0.06866 | -0.07915 | -0.0536 | 0.0625 | 0.4375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | C-LoRA | 5 | -0.03064 | -0.04059 | -0.01651 | 0.0625 | 0.4375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | X-DER | 5 | -0.07595 | -0.08761 | -0.06033 | 0.0625 | 0.4375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.0553 | 0.04257 | 0.06926 | 0.0625 | 0.4375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | ICD | 5 | -0.005421 | -0.01714 | 0.01048 | 0.5 | 0.5 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_AUC | CLEAN-Full | CLEAN-LoRA | 5 | 0.02175 | 0.003383 | 0.05203 | 0.0625 | 0.4375 | False |
