# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Formal-capable seed count: 10 >= 8 (Holm family size=6).

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | EWC | 10 | 0.01918 | 0.01421 | 0.02449 | 0.001953 | 0.01172 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | DER++ | 10 | 0.02945 | 0.0267 | 0.03211 | 0.001953 | 0.01172 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA | 10 | 0.01321 | 0.008332 | 0.01815 | 0.003906 | 0.01172 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | X-DER | 10 | 0.006418 | 0.005437 | 0.007436 | 0.001953 | 0.01172 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.06161 | 0.05379 | 0.06888 | 0.001953 | 0.01172 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | ICD | 10 | 0.1455 | 0.1428 | 0.1482 | 0.001953 | 0.01172 | True |
