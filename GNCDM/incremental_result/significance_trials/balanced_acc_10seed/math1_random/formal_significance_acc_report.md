# Seed-level paired statistical analysis

The test table uses an exact two-sided paired sign-flip permutation test. Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.

## Inference warning

Formal-capable seed count: 10 >= 8 (Holm family size=6).

| dataset | split | protocol | metric | target | baseline | n | oriented_delta_mean | ci95_low | ci95_high | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | EWC | 10 | 0.01935 | 0.008915 | 0.02745 | 0.007812 | 0.02344 | True |
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | DER++ | 10 | -0.008862 | -0.01938 | -0.001022 | 0.06836 | 0.06836 | False |
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA | 10 | 0.02203 | 0.01136 | 0.02981 | 0.005859 | 0.02344 | True |
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | X-DER | 10 | -0.01278 | -0.02312 | -0.004829 | 0.009766 | 0.02344 | True |
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | C-LoRA-GNCDM | 10 | 0.06413 | 0.05116 | 0.07507 | 0.001953 | 0.01172 | True |
| math1 | random | merged_10seed | ACC_overall | CLEAN-Full | ICD | 10 | 0.02681 | 0.01616 | 0.03491 | 0.003906 | 0.01953 | True |
