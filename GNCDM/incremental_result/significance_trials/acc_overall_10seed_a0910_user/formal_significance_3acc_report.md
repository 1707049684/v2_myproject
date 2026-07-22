# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Formal-capable seed count: 10 >= 10 (family=6×3).

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | EWC | 10 | 0.0151 | 0.009193 | 0.021 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | DER++ | 10 | 0.02915 | 0.02614 | 0.03207 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | C-LoRA | 10 | 0.01432 | 0.008807 | 0.02037 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | X-DER | 10 | 0.002864 | 0.0003318 | 0.005354 | 0.06836 | 0.06836 | False |
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.07336 | 0.06113 | 0.08538 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_old | CLEAN-Full | ICD | 10 | 0.189 | 0.1859 | 0.1923 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | EWC | 10 | 0.02822 | 0.02297 | 0.03351 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | DER++ | 10 | 0.03013 | 0.02432 | 0.03626 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | C-LoRA | 10 | 0.01076 | 0.004184 | 0.01722 | 0.01758 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | X-DER | 10 | 0.01428 | 0.009021 | 0.01901 | 0.003906 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.03561 | 0.02187 | 0.05302 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_new | CLEAN-Full | ICD | 10 | 0.04912 | 0.04495 | 0.05355 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | EWC | 10 | 0.01918 | 0.01421 | 0.02449 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | DER++ | 10 | 0.02945 | 0.0267 | 0.03211 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA | 10 | 0.01321 | 0.008332 | 0.01815 | 0.003906 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | X-DER | 10 | 0.006418 | 0.005437 | 0.007436 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.06161 | 0.05379 | 0.06888 | 0.001953 | 0.03516 | True |
| a0910 | user | merged_10seed | ACC_overall | CLEAN-Full | ICD | 10 | 0.1455 | 0.1428 | 0.1482 | 0.001953 | 0.03516 | True |
