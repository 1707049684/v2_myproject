# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Exploratory five-seed analysis: 5 paired seeds cannot attain a Holm-adjusted two-sided exact rejection across 7 planned hypotheses at alpha=0.05. At least 9 seeds are required.

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | EWC | 5 | -0.004852 | -0.01145 | 0.005001 | 0.375 | 0.75 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | DER++ | 5 | 0.004715 | 0.0008805 | 0.007374 | 0.125 | 0.4375 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | C-LoRA | 5 | -0.0003918 | -0.004022 | 0.00253 | 0.875 | 0.875 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | X-DER | 5 | 0.01581 | 0.01273 | 0.01945 | 0.0625 | 0.4375 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | C-LoRA-GNCDM | 5 | 0.02574 | 0.02037 | 0.03137 | 0.0625 | 0.4375 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | ICD | 5 | 0.05225 | 0.0503 | 0.05529 | 0.0625 | 0.4375 | False |
| a0910 | user | a224e306d0064143 | Balanced_AUC | CLEAN-Full | CLEAN-LoRA | 5 | -0.007701 | -0.0123 | -0.004187 | 0.0625 | 0.4375 | False |
