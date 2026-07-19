# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 6 planned hypotheses at alpha=0.05. At least 8 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | EWC | 5 | 0.02674 | 0.0232 | 0.03083 | 0.0625 | 0.375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | DER++ | 5 | -0.0004939 | -0.003736 | 0.005617 | 1 | 1 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | C-LoRA | 5 | 0.02795 | 0.02582 | 0.03008 | 0.0625 | 0.375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | X-DER | 5 | -0.00643 | -0.01203 | -0.0007347 | 0.125 | 0.375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.07615 | 0.0636 | 0.08537 | 0.0625 | 0.375 | False |
| math1 | random | f4877a6c81b57ed6 | Balanced_ACC | CLEAN-Full | ICD | 5 | 0.03346 | 0.02805 | 0.03901 | 0.0625 | 0.375 | False |
