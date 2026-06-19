| Method | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD |
|---|---|---|---|---|---|---|---|---|---|
| Base | 0.7598 | - | 0.4275 | - | 0.7370 | - | 0.8183 | - | - |
| Ours-Ablated | 0.7165 | 0.7629 | 0.4918 | 0.4306 | 0.6514 | 0.7267 | 0.7070 | 0.7965 | 0.0000 |
| Ours (DNA) | 0.7598 | 0.7527 | 0.4275 | 0.4337 | 0.7370 | 0.7240 | 0.8183 | 0.7964 | 0.0000 |
| Ours (LoRA) | 0.7598 | 0.7486 | 0.4275 | 0.4354 | 0.7370 | 0.7241 | 0.8183 | 0.7953 | 0.0000 |
| Full Replay Oracle | 0.7636 | 0.7628 | 0.4298 | 0.4358 | 0.7367 | 0.7281 | 0.8170 | 0.8001 | 0.0809 |
| Naive FT | 0.6955 | 0.7653 | 0.5021 | 0.4337 | 0.6796 | 0.7313 | 0.7698 | 0.8008 | 0.0854 |
| EWC (lambda=10000) | 0.7023 | 0.6690 | 0.5208 | 0.5333 | 0.6753 | 0.6509 | 0.7536 | 0.7294 | 0.0883 |
| DER++ (mem=5000) | 0.7082 | 0.6994 | 0.4509 | 0.4576 | 0.6921 | 0.6806 | 0.7767 | 0.7679 | 0.0376 |
| C-LoRA (lambda=10000) | 0.6999 | 0.6674 | 0.5201 | 0.5339 | 0.6769 | 0.6425 | 0.7555 | 0.7187 | 0.1320 |
| X-DER (mem=5000) | 0.7525 | 0.7051 | 0.4277 | 0.4501 | 0.7347 | 0.6970 | 0.8191 | 0.7858 | 0.0536 |

*alpha note*: G-NCDM uses **alpha=0.1**, selected by `experiments/_core/sweep_a0910_random_alpha.py` (full sweep 0.1~0.95, DNA mean(valid AUC) peaks at 0.1 / selAUC 0.7579). The previous paper-aligned 0.9 was never actually swept and is superseded. Ours(DNA/LoRA) new-task AUC (0.753/0.749) beats all three CL baselines (0.67~0.70) with exact zero forgetting (old=Base, TMD=0).

*TMD note*: Ours (DNA/LoRA/Ablated) operate in G-NCDM concept-theta space (TMD=0 by architectural isolation). EWC/DER++/C-LoRA run on the `CognitiveBackbone` (Embedding+MLP); their TMD lives in embedding space and is **not** comparable in magnitude — only its sign (>0, i.e. non-zero forgetting) is meaningful. Balanced points shown: EWC/C-LoRA at lambda=10000, DER++ at mem=5000. All rows are a0910 random_split, set_seed(42). (Baselines are alpha-independent: EWC/C-LoRA rows are bit-identical to the prior alpha=0.9 table; only DER++ jitters via reservoir sampling.)
*X-DER 例外*：X-DER 用 **G-NCDM 同骨干**（route A，arXiv:2201.00766 高保真适配），其 TMD 与 Ours 在**同一概念 θ 空间、可直接对比**，属纯策略对比。结论：X-DER 是强回放基线，但 **TMD=0.0536>0 仍遗忘**，而 Ours 架构隔离 **TMD=0** 严格零遗忘。
