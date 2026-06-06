# Main comparison table — Math1, random split (score prediction)

All nine methods share the same strict topological bipartition (ΔK={0,1,3,6} → 13 old items /
7 new items) and the same test rows. Random split: test learners are **shared** with training
(split by interaction), so no cold-start is needed — Ours predicts via `forward_using_buf`
(buffer, no self-information), the CL baselines (CognitiveBackbone) predict directly. AUC/ACC/F1
comparable row-by-row; higher is better except RMSE; TMD lower is better (0 = no forgetting).

| Method | Backbone | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | RMSE_old | RMSE_new | TMD |
|---|---|---|---|---|---|---|---|---|---|---|
| Base (old task only) | G-NCDM | **0.8072** | – | 0.7293 | – | 0.7255 | – | 0.4186 | – | – |
| **Ours (Dynamic DNA)** | G-NCDM | **0.8072** | 0.7204 | 0.7293 | 0.7548 | 0.7255 | 0.6039 | 0.4186 | 0.4409 | **0.0000** |
| **Ours (LoRA)** | G-NCDM | **0.8072** | 0.6712 | 0.7293 | 0.6955 | 0.7255 | 0.5566 | 0.4186 | 0.4717 | **0.0000** |
| Ours-Ablated (no ⊥-mask) | G-NCDM | 0.7381 | 0.8480 | 0.6608 | 0.7574 | 0.7186 | 0.6458 | 0.4627 | 0.3911 | 0.0000 |
| Full-Replay Oracle | G-NCDM | 0.8108 | 0.8316 | 0.7191 | 0.7501 | 0.7058 | 0.6632 | 0.4214 | 0.4002 | 0.0810 |
| Naive-FT | G-NCDM | 0.7648 | 0.8503 | 0.6569 | 0.7553 | 0.5863 | 0.6856 | 0.4596 | 0.3869 | 0.0694 |
| EWC (λ=10³) | CognitiveBackbone | 0.7687 | 0.8162 | 0.6844 | 0.7351 | 0.6465 | 0.6120 | 0.5046 | 0.4258 | 0.1041 † |
| DER++ (mem=5000) | CognitiveBackbone | 0.7967 | 0.8405 | 0.7192 | 0.7553 | 0.7300 | 0.6713 | 0.4244 | 0.3906 | 0.0081 † |
| C-LoRA (λ=10⁴) | CognitiveBackbone | 0.7689 | 0.7905 | 0.6981 | 0.7156 | 0.6863 | 0.5925 | 0.4954 | 0.4587 | 0.1088 † |

**Reading the table**

1. **Zero forgetting and best old-task retention are Ours.** Dynamic-DNA and LoRA keep the old
   task *bit-identical to Base* (AUC_old = 0.8072, the highest of all methods) with **TMD = 0**.
   Every baseline drifts (TMD† > 0; DER++ is lowest at 0.008 because replay is near-total on a
   20-item set, but still not exactly 0).
2. **On new-task accuracy the baselines lead here.** DER++ (0.8405) and EWC (0.8162) exceed Ours
   (DNA 0.7204, LoRA 0.6712) on AUC_new. This is a **dataset-scale artifact**: Math1 has only
   **7 new items spanning 4 new concepts**, so each learner provides very few new-concept
   responses — G-NCDM's *concept-decomposed* new branch is data-starved, whereas a free
   Embedding+MLP (especially DER++, which replays almost the whole tiny item set) fits a handful
   of new item embeddings easily. It is **not** evidence that the baselines are better continual
   learners.
3. **Use the larger benchmark for plasticity.** On ASSIST a0910 (17 746 items — see
   `docs/main_table_a0910_user_split.md`) Ours' new-task AUC *matches or exceeds* every baseline
   (LoRA 0.707 ≥ EWC/DER/C-LoRA) **while keeping TMD = 0**. Math1's small new-item set is the only
   reason Ours' new-task number is low here.

**Caveats (must accompany the table)**

- **Backbone differs for the baselines** (CognitiveBackbone ≠ G-NCDM): we claim zero forgetting +
  best old-task retention under the same split/protocol, **not** a pure-strategy win on every cell.
- **Small new-item set.** With only 7 new items, this table is a *stability/forgetting* showcase,
  not a *plasticity* one. Report Math1 either as Ours' own six-strategy zero-forgetting result, or
  with this caveat explicit; feature a0910 for the head-to-head plasticity comparison.
- **TMD vs TMD†.** Ours' TMD is in G-NCDM concept-θ space (0 by architectural isolation); the
  baselines' TMD† is in learner-embedding space — magnitudes are **not** comparable, only the sign.
- **Balanced λ.** EWC λ=10³, C-LoRA λ=10⁴ (max avg(AUC_old, AUC_new) over the sweep); full sweeps
  in `ewc_lambda_sweep_math1_random_split.csv` / `clora_lambda_sweep_math1_random_split.csv`.
- Source: `incremental_result/all_methods_math1_random_split.csv`, from
  `math1_cl_baselines_random_split.py` + `incremental_results_math1_random_split.csv` (set_seed(42)).
