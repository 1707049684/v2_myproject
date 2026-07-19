# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory five-seed analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 7 planned hypotheses at alpha=0.05. At least 9 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | EWC | 5 | 0.05009 | 0.04603 | 0.05498 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | DER++ | 5 | 0.007761 | 0.00603 | 0.009514 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | C-LoRA | 5 | 0.05296 | 0.04328 | 0.06743 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | X-DER | 5 | 0.01281 | 0.01026 | 0.01557 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.006813 | 0.004248 | 0.01116 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | ICD | 5 | 0.05559 | 0.05493 | 0.05635 | 0.0625 | 0.4375 | False |
| junyi | random | 3b3caf91efab385f | Balanced_AUC | CLEAN-Full | CLEAN-LoRA | 5 | 0.008926 | 0.004755 | 0.0154 | 0.0625 | 0.4375 | False |
