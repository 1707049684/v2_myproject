| Method | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | RD |
|---|---|---|---|---|---|---|---|---|---|
| X-DER (mem=5000) | 0.6596 | 0.6754 | 0.4509 | 0.4635 | 0.6963 | 0.6655 | 0.7969 | 0.7770 | 0.0285 |

*口径*：a0910 user_split，G-NCDM 骨干，support/query 留出（frac=0.5, seed=7），与 eval_all_methods_user_split 同口径。
*X-DER 损失*：L_BCE(new)+α·L_KD(logit)+β·L_BCE_buf+λ·L_Future；memory revision(γ clamp)。
*超参*：mem=5000, α=0.5, β=0.5, λ=0.5, γ=0.75, alpha=0.6, epochs=15。
*RD*：与 Ours 同在 G-NCDM 概念 θ 空间（populate_buffers 取训练用户），可与 Ours 行直接比。
