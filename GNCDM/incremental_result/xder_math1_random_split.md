| Method | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD |
|---|---|---|---|---|---|---|---|---|---|
| X-DER (mem=5000) | 0.8121 | 0.8445 | 0.4208 | 0.3913 | 0.7269 | 0.7542 | 0.7303 | 0.6648 | 0.0804 |

*口径*:math1 random_split,G-NCDM 骨干,buf 无泄漏预测,与 Ours 主表逐行可比。
*X-DER 损失*:L_BCE(new)+α·L_KD(logit)+β·L_BCE_buf+λ·L_Future;memory revision(γ clamp)。
*超参*:mem=5000, α=0.5, β=0.5, λ=0.5, γ=0.75, alpha=0.2, epochs=25。
*TMD*:与 Ours 同在 G-NCDM 概念 θ 空间(calculate_tmd 取前 K_old 列),可与 Ours 行直接比。
*L_Future*:CDM 无类槽,以 ΔK 潜通道反激活 mean(relu(θ_ΔK)^2) 再诠释 X-DER future-prep,论文须注明为 CDM 适配而非原类头机制。
