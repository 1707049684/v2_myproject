# Balanced_ACC paired significance (reanalysis)

Primary metric: `Balanced_ACC = 0.7 * ACC_old + 0.3 * ACC_new`.
Target: CLEAN-Full. Baselines: EWC, DER++, C-LoRA, X-DER, C-LoRA-GNCDM, ICD.
CLEAN-LoRA and Full-Replay excluded. Seeds: 1,7,21,42,84.

## Inference status

Five paired seeds yield a minimum two-sided exact p-value of 0.0625, so Holm rejection at alpha=0.05 is unattainable (exploratory_underpowered). Effect sizes, win counts, and CIs remain informative.

## math1_random

CLEAN-Full mean Balanced_ACC = 0.7271 (sd=0.0069, n=5).

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 5 | 0.02674 | 0.0232 | 0.03083 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| DER++ | 5 | -0.0004939 | -0.003736 | 0.005617 | 1 | 4 | 0 | 1 | 1 | False |
| C-LoRA | 5 | 0.02795 | 0.02582 | 0.03008 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| X-DER | 5 | -0.00643 | -0.01203 | -0.0007347 | 1 | 4 | 0 | 0.125 | 0.375 | False |
| C-LoRA-GNCDM | 5 | 0.07615 | 0.0636 | 0.08537 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| ICD | 5 | 0.03346 | 0.02805 | 0.03901 | 5 | 0 | 0 | 0.0625 | 0.375 | False |

## junyi_random

CLEAN-Full mean Balanced_ACC = 0.7722 (sd=0.0014, n=5).

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 5 | 0.05709 | 0.04592 | 0.07473 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| DER++ | 5 | 0.01305 | 0.01238 | 0.01372 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| C-LoRA | 5 | 0.05417 | 0.04229 | 0.06919 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| X-DER | 5 | 0.01068 | 0.008117 | 0.01235 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| C-LoRA-GNCDM | 5 | 0.04518 | 0.02852 | 0.06581 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| ICD | 5 | 0.08469 | 0.08366 | 0.08589 | 5 | 0 | 0 | 0.0625 | 0.375 | False |

## a0910_random

CLEAN-Full mean Balanced_ACC = 0.7312 (sd=0.0016, n=5).

| baseline | n | oriented_delta_mean | ci95_low | ci95_high | wins | losses | ties | p_raw | p_holm | reject_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| EWC | 5 | 0.06901 | 0.06456 | 0.07302 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| DER++ | 5 | 0.04071 | 0.03856 | 0.04261 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| C-LoRA | 5 | 0.06571 | 0.06365 | 0.06791 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| X-DER | 5 | 0.01179 | 0.01051 | 0.01294 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| C-LoRA-GNCDM | 5 | 0.03434 | 0.02197 | 0.05526 | 5 | 0 | 0 | 0.0625 | 0.375 | False |
| ICD | 5 | 0.1554 | 0.1541 | 0.1566 | 5 | 0 | 0 | 0.0625 | 0.375 | False |

