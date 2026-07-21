# ACC_overall 10-seed paired significance

Metric: `ACC_overall = (n_old·ACC_old + n_new·ACC_new) / (n_old+n_new)` with test-split interaction counts after strict bipartition.
Counts: {'math1': (10901, 5935), 'junyi': (13997, 6398), 'a0910': (37642, 16836)}.
Target: CLEAN-Full. Baselines: EWC, DER++, C-LoRA, X-DER, C-LoRA-GNCDM, ICD.
Excluded from tests: CLEAN-LoRA, Full-Replay.
Seeds: {1,2,3,5,7,11,13,21,42,84}.

## math1_random

Formal-capable seed count: 10 >= 8 (Holm family size=6).

CLEAN-Full mean ACC_overall = 0.7218 (n=10).

| method | n | ACC_old | ACC_new | ACC_overall | sd |
|---|---:|---:|---:|---:|---:|
| X-DER | 10 | 0.7254 | 0.7512 | 0.7345 | 0.0016 |
| Full-Replay | 10 | 0.7220 | 0.7512 | 0.7323 | 0.0017 |
| DER++ | 10 | 0.7172 | 0.7552 | 0.7306 | 0.0033 |
| CLEAN-Full | 10 | 0.7214 | 0.7225 | 0.7218 | 0.0162 |
| CLEAN-LoRA | 10 | 0.7214 | 0.6850 | 0.7085 | 0.0101 |
| EWC | 10 | 0.6859 | 0.7327 | 0.7024 | 0.0053 |
| C-LoRA | 10 | 0.6888 | 0.7198 | 0.6997 | 0.0046 |
| ICD | 10 | 0.6862 | 0.7110 | 0.6949 | 0.0000 |
| C-LoRA-GNCDM | 10 | 0.6277 | 0.7125 | 0.6576 | 0.0102 |

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 10 | 0.01935 | 0.008915 | 0.02745 | 9 | 1 | 0 | 0.007812 | 0.02344 | True |
| DER++ | 10 | -0.008862 | -0.01938 | -0.001022 | 2 | 8 | 0 | 0.06836 | 0.06836 | False |
| C-LoRA | 10 | 0.02203 | 0.01136 | 0.02981 | 9 | 1 | 0 | 0.005859 | 0.02344 | True |
| X-DER | 10 | -0.01278 | -0.02312 | -0.004829 | 1 | 9 | 0 | 0.009766 | 0.02344 | True |
| C-LoRA-GNCDM | 10 | 0.06413 | 0.05116 | 0.07507 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| ICD | 10 | 0.02681 | 0.01616 | 0.03491 | 9 | 1 | 0 | 0.003906 | 0.01953 | True |

## junyi_random

Formal-capable seed count: 10 >= 8 (Holm family size=6).

CLEAN-Full mean ACC_overall = 0.7719 (n=10).

| method | n | ACC_old | ACC_new | ACC_overall | sd |
|---|---:|---:|---:|---:|---:|
| Full-Replay | 10 | 0.7855 | 0.7547 | 0.7758 | 0.0018 |
| CLEAN-Full | 10 | 0.7840 | 0.7453 | 0.7719 | 0.0011 |
| CLEAN-LoRA | 10 | 0.7840 | 0.7357 | 0.7689 | 0.0019 |
| X-DER | 10 | 0.7759 | 0.7319 | 0.7621 | 0.0023 |
| DER++ | 10 | 0.7653 | 0.7427 | 0.7582 | 0.0022 |
| C-LoRA | 10 | 0.7126 | 0.7312 | 0.7184 | 0.0134 |
| EWC | 10 | 0.7154 | 0.7158 | 0.7156 | 0.0141 |
| C-LoRA-GNCDM | 10 | 0.6717 | 0.7511 | 0.6966 | 0.0501 |
| ICD | 10 | 0.7001 | 0.6684 | 0.6902 | 0.0040 |

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 10 | 0.05633 | 0.04875 | 0.06587 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| DER++ | 10 | 0.01367 | 0.01266 | 0.01467 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| C-LoRA | 10 | 0.05348 | 0.04584 | 0.06167 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| X-DER | 10 | 0.009801 | 0.008174 | 0.01136 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| C-LoRA-GNCDM | 10 | 0.07529 | 0.04822 | 0.1063 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| ICD | 10 | 0.08173 | 0.07944 | 0.08406 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |

## a0910_random

Formal-capable seed count: 10 >= 8 (Holm family size=6).

CLEAN-Full mean ACC_overall = 0.7311 (n=10).

| method | n | ACC_old | ACC_new | ACC_overall | sd |
|---|---:|---:|---:|---:|---:|
| Full-Replay | 10 | 0.7366 | 0.7299 | 0.7345 | 0.0017 |
| CLEAN-Full | 10 | 0.7351 | 0.7221 | 0.7311 | 0.0015 |
| CLEAN-LoRA | 10 | 0.7351 | 0.7212 | 0.7308 | 0.0017 |
| X-DER | 10 | 0.7286 | 0.6983 | 0.7192 | 0.0010 |
| C-LoRA-GNCDM | 10 | 0.7011 | 0.7121 | 0.7045 | 0.0173 |
| DER++ | 10 | 0.6950 | 0.6766 | 0.6893 | 0.0028 |
| C-LoRA | 10 | 0.6764 | 0.6428 | 0.6661 | 0.0023 |
| EWC | 10 | 0.6687 | 0.6409 | 0.6601 | 0.0062 |
| ICD | 10 | 0.5440 | 0.6500 | 0.5767 | 0.0000 |

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 10 | 0.07098 | 0.06751 | 0.07488 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| DER++ | 10 | 0.04183 | 0.04014 | 0.04367 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| C-LoRA | 10 | 0.06506 | 0.0633 | 0.06684 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| X-DER | 10 | 0.01186 | 0.01082 | 0.01291 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| C-LoRA-GNCDM | 10 | 0.02665 | 0.01872 | 0.03868 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |
| ICD | 10 | 0.1544 | 0.1535 | 0.1552 | 10 | 0 | 0 | 0.001953 | 0.01172 | True |

