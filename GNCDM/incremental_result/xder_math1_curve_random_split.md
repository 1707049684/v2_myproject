| Method | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | RD |
|---|---|---|---|---|---|---|---|---|---|
| X-DER (mem=5000) | 0.8153 | 0.8468 | 0.4198 | 0.3899 | 0.7318 | 0.7582 | 0.7334 | 0.6623 | 0.0734 |

*口径*:math1_curve random_split,G-NCDM 骨干,buf 无泄漏预测,与 Ours 主表逐行可比。
*X-DER 损失*:L_BCE(new)+α·L_KD(logit)+β·L_BCE_buf+λ·L_Future;memory revision(γ clamp)。
*超参*:mem=5000, α=0.5, β=0.5, λ=0.5, γ=0.75, alpha=0.2, epochs=25。
*RD*:与 Ours 同在 G-NCDM 概念 θ 空间(calculate_rd 取前 K_old 列),可与 Ours 行直接比。
*L_Future*:CDM 无类槽,以 ΔK 潜通道反激活 mean(relu(θ_ΔK)^2) 再诠释 X-DER future-prep,论文须注明为 CDM 适配而非原类头机制。
