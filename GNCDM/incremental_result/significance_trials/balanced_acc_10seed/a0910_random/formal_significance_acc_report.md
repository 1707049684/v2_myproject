# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Formal-capable seed count: 10 >= 8 (Holm family size=6).

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | EWC | 10 | 0.07098 | 0.06751 | 0.07488 | 0.001953 | 0.01172 | True |
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | DER++ | 10 | 0.04183 | 0.04014 | 0.04367 | 0.001953 | 0.01172 | True |
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA | 10 | 0.06506 | 0.0633 | 0.06684 | 0.001953 | 0.01172 | True |
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | X-DER | 10 | 0.01186 | 0.01082 | 0.01291 | 0.001953 | 0.01172 | True |
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.02665 | 0.01872 | 0.03868 | 0.001953 | 0.01172 | True |
| a0910 | random | merged_10seed | ACC_overall | CLEAN-Full | ICD | 10 | 0.1544 | 0.1535 | 0.1552 | 0.001953 | 0.01172 | True |
