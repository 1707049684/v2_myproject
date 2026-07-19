# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 6 planned hypotheses at alpha=0.05. At least 8 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | EWC | 5 | 0.05709 | 0.04592 | 0.07473 | 0.0625 | 0.375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | DER++ | 5 | 0.01305 | 0.01238 | 0.01372 | 0.0625 | 0.375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | C-LoRA | 5 | 0.05417 | 0.04229 | 0.06919 | 0.0625 | 0.375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | X-DER | 5 | 0.01068 | 0.008117 | 0.01235 | 0.0625 | 0.375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.04518 | 0.02852 | 0.06581 | 0.0625 | 0.375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_ACC | CLEAN-Full | ICD | 5 | 0.08469 | 0.08366 | 0.08589 | 0.0625 | 0.375 | False |
