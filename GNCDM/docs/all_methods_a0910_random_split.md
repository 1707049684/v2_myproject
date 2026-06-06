| Method | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD |
|---|---|---|---|---|---|---|---|---|---|
| Base | 0.7441 | - | 0.4330 | - | 0.7297 | - | 0.8126 | - | - |
| Ours-Ablated | 0.7190 | 0.7397 | 0.4677 | 0.4370 | 0.6872 | 0.7240 | 0.7633 | 0.7999 | 0.0000 |
| Ours (DNA) | 0.7441 | 0.7361 | 0.4330 | 0.4404 | 0.7297 | 0.7156 | 0.8126 | 0.7906 | 0.0000 |
| Ours (LoRA) | 0.7441 | 0.7401 | 0.4330 | 0.4376 | 0.7297 | 0.7232 | 0.8126 | 0.7993 | 0.0000 |
| Full Replay Oracle | 0.7476 | 0.7358 | 0.4322 | 0.4401 | 0.7312 | 0.7230 | 0.8166 | 0.8016 | 0.0216 |
| Naive FT | 0.7007 | 0.7462 | 0.4923 | 0.4386 | 0.6888 | 0.7243 | 0.7755 | 0.7984 | 0.0215 |
| EWC (lambda=10000) | 0.7023 | 0.6690 | 0.5208 | 0.5333 | 0.6753 | 0.6509 | 0.7536 | 0.7294 | 0.0883 |
| DER++ (mem=5000) | 0.7126 | 0.6792 | 0.4465 | 0.4622 | 0.6988 | 0.6750 | 0.7833 | 0.7676 | 0.0241 |
| C-LoRA (lambda=10000) | 0.6999 | 0.6674 | 0.5201 | 0.5339 | 0.6769 | 0.6425 | 0.7555 | 0.7187 | 0.1320 |

*TMD note*: Ours (DNA/LoRA/Ablated) operate in G-NCDM concept-theta space (TMD=0 by architectural isolation). EWC/DER++/C-LoRA run on the `CognitiveBackbone` (Embedding+MLP); their TMD lives in embedding space and is **not** comparable in magnitude — only its sign (>0, i.e. non-zero forgetting) is meaningful. Balanced points shown: EWC/C-LoRA at lambda=10000, DER++ at mem=5000. All rows are a0910 random_split, set_seed(42).
