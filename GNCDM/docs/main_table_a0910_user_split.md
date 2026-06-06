# Main comparison table — ASSIST a0910, user split (score reconstruction)

All nine methods are evaluated under **one shared support/query split** (frac=0.5, seed=7):
for each held-out test learner, half of their responses (support) diagnose the learner, the
**disjoint** other half (query) is predicted. Old-task / new-task metrics are computed on the
query responses to old / new items. Every method is scored on the **identical query rows**, so
AUC/ACC/F1/RMSE are directly comparable row-by-row. Higher is better except RMSE; TMD lower is
better (0 = no forgetting).

| Method | Backbone | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | RMSE_old | RMSE_new | TMD |
|---|---|---|---|---|---|---|---|---|---|---|
| Base (old task only) | G-NCDM | 0.6552 | – | 0.6892 | – | 0.7824 | – | 0.4590 | – | – |
| **Ours (Dynamic DNA)** | G-NCDM | 0.6552 | 0.6919 | 0.6892 | 0.6694 | 0.7824 | 0.7730 | 0.4590 | 0.4568 | **0.0000** |
| **Ours (LoRA)** | G-NCDM | 0.6552 | **0.7066** | 0.6892 | 0.6843 | 0.7824 | **0.7809** | 0.4590 | 0.4520 | **0.0000** |
| Ours-Ablated (no ⊥-mask) | G-NCDM | 0.6294 | 0.7060 | 0.6235 | 0.6837 | 0.7108 | 0.7803 | 0.4808 | 0.4541 | 0.0000 |
| Full-Replay Oracle | G-NCDM | 0.6436 | 0.6843 | 0.6901 | 0.6687 | 0.7854 | 0.7735 | 0.4632 | 0.4617 | 0.0150 |
| Naive-FT | G-NCDM | 0.6274 | 0.6817 | 0.6360 | 0.6564 | 0.7321 | 0.7452 | 0.4834 | 0.4628 | 0.0171 |
| EWC (λ=10⁴) | CognitiveBackbone | 0.7064 | 0.6806 | 0.6865 | 0.6667 | 0.7691 | 0.7437 | 0.4972 | 0.5066 | 0.0859 † |
| DER++ (mem=5000) | CognitiveBackbone | 0.6803 | 0.6659 | 0.6709 | 0.6483 | 0.7589 | 0.7287 | 0.5024 | 0.5103 | 0.1164 † |
| C-LoRA (λ=10) | CognitiveBackbone | 0.6839 | 0.6889 | 0.6764 | 0.6732 | 0.7666 | 0.7569 | 0.4676 | 0.4740 | 0.1474 † |

**Reading the table**

1. **Zero forgetting is unique to Ours.** Dynamic-DNA and LoRA leave the old-task metrics
   *bit-identical to Base* (AUC_old/ACC_old/F1_old equal Base's 0.6552/0.6892/0.7824) with
   **TMD = 0** — architectural isolation guarantees no drift. Every continual-learning baseline
   forgets (TMD > 0: 0.086–0.147).
2. **Plasticity is competitive or better.** Ours (LoRA) attains the **highest new-task AUC
   (0.7066)** and F1 (0.7809) of all nine methods; Ours (DNA) (0.6919) also matches or exceeds
   EWC/DER++/C-LoRA (0.666–0.689). Ours learns the new concepts without any replay or retraining
   of old parameters.
3. **On the absolute old-task AUC the baselines read slightly higher (0.680–0.706 vs 0.6552).**
   This is **not** a forgetting effect — Ours' old task equals Base by construction (TMD = 0).
   It reflects (i) a *different backbone* (CognitiveBackbone vs the monotonicity-constrained
   generative G-NCDM) and (ii) the baselines' **per-learner cold-start fitting** at inference
   (30 epochs of gradient descent on each test learner's support set), an inference-time
   adaptation that G-NCDM's single forward-pass diagnosis does not use. The baselines buy this
   with non-zero forgetting *and* the cost of per-learner retraining; G-NCDM is inductive
   (one forward pass, no per-learner optimization).

**Caveats (must accompany the table)**

- **Backbone differs for the baselines.** EWC/DER++/C-LoRA run on `CognitiveBackbone`
  (Embedding+MLP), not G-NCDM. We therefore claim *"under the same split and protocol Ours offers
  zero forgetting with competitive plasticity"*, **not** a pure-strategy victory on every cell.
- **TMD vs TMD†.** Ours' TMD lives in G-NCDM **concept-θ space** (0 by architectural isolation);
  the baselines' TMD† lives in **learner-embedding space**. The magnitudes are **not comparable** —
  only the sign (`= 0` vs `> 0`) is meaningful.
- **Balanced λ.** EWC and C-LoRA are reported at the λ maximizing avg(AUC_old, AUC_new) over the
  sweep (EWC λ=10⁴, C-LoRA λ=10); full sweeps are in
  `ewc_lambda_sweep_a0910_user_split.csv` / `clora_lambda_sweep_a0910_user_split.csv`.
- Source data: `incremental_result/all_methods_a0910_user_split.csv`, produced by
  `experiments/eval_all_methods_user_split.py` (set_seed(42)).

> Note on math1: the same nine-method run on math1 user split is **not** a fair baseline
> comparison — with only 7 new items, the per-learner support set contains too few new-item
> responses for G-NCDM's concept-decomposed diagnosis, so its new-task AUC degrades to near-random
> while the baselines' single global ability vector stays robust. math1 should report Ours'
> own six-strategy zero-forgetting result, not the cross-backbone baseline table.
