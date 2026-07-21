# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Formal-capable seed count: 10 >= 8 (Holm family size=6).

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | EWC | 10 | 0.05633 | 0.04875 | 0.06587 | 0.001953 | 0.01172 | True |
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | DER++ | 10 | 0.01367 | 0.01266 | 0.01467 | 0.001953 | 0.01172 | True |
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA | 10 | 0.05348 | 0.04584 | 0.06167 | 0.001953 | 0.01172 | True |
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | X-DER | 10 | 0.009801 | 0.008174 | 0.01136 | 0.001953 | 0.01172 | True |
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.07529 | 0.04822 | 0.1063 | 0.001953 | 0.01172 | True |
| junyi | random | merged_10seed | ACC_overall | CLEAN-Full | ICD | 10 | 0.08173 | 0.07944 | 0.08406 | 0.001953 | 0.01172 | True |
