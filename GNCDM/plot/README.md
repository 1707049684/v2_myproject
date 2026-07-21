# GNCDM/plot — 画图脚本目录

**约定：所有 matplotlib / 论文配图 / epoch 曲线等可视化脚本统一放在此目录，不要放在 `experiments/`。**

`experiments/` 只放增量学习主实验入口（`run_incremental_*`）与 `_core/` 管线库；画图脚本依赖实验库时，在脚本头部把 `GNCDM/`、`experiments/`、`experiments/_core/` 加入 `sys.path`。

## 运行方式

```bash
cd GNCDM/plot
python plot_epoch_curve_math1.py
python plot_epoch_curve_gncdm_math1.py
python plot_epoch_curve_avalanche_math1.py   # 需 avalanche-lib（_scratch/clbase-venv）
python plot_epoch_curve_final_math1.py       # 合并上述 CSV，出终版图
```

产物 CSV/PNG 仍写入 `GNCDM/incremental_result/`（与主实验结果同目录）。

## 当前脚本

| 脚本 | 说明 |
|------|------|
| `plot_epoch_curve_math1.py` | 4 策略效率曲线（Ours-DNA/LoRA/Full-Replay/Naive-FT） |
| `plot_epoch_curve_gncdm_math1.py` | GNCDM 骨干 5 条曲线 + X-DER + C-LoRA-GNCDM |
| `plot_epoch_curve_avalanche_math1.py` | EWC / DER++（avalanche 环境） |
| `plot_epoch_curve_final_math1.py` | 合并 7 模型终版图（纵轴 ACC_new） |
| `plot_epoch_curve_final_math1_old.py` | 合并 7 模型终版图（纵轴 ACC_old，旧任务保持） |
| `plot_alpha_sensitivity_random.py` | Math1/junyi/a0910 random_split 的 α 敏感性（sel_DNA_validACC） |
| `plot_umap_math1_gncdm.py` | Math1 θ UMAP：默认三面板 G-NCDM \| CLEAN-Full \| CLEAN-LoRA（`--compare all3`） |
| `plot_tsne_math1_gncdm.py` | Math1 θ t-SNE：同三面板（复用 umap_cache_math1 的 θ） |
| `plot_aligned_umap_math1_drift.py` | Math1 旧概念 θ 的 AlignedUMAP 两阶段（t=0 Base vs t=1 CLEAN） |
| `plot_aligned_umap_math1_baselines.py` | 同口径：Full-Replay / X-DER / C-LoRA-GNCDM（EWC/DER/ICD 无概念 θ，不可画）；`--method grid` 出 2×2 拼图 |
| `plot_old_pred_dist_ablation.py` | CLEAN 聚合消融：旧题 \(\hat y\) 密度（user_split）；`--dataset math1|a0910` |
| `plot_aligned_umap_math1_pred_ablation.py` | Math1 **random_split**：旧题 \(\hat y\) AlignedUMAP 消融；另出 **轨迹箭头×3**（`*_traj`）与 **残差 UMAP(Δŷ)**（`*_delta`）放大对比 |
| `plot_aligned_umap_a0910_pred_ablation.py` | ASSIST a0910 **random_split**：同上消融（α=0.1，`N_OLD_PRED=256` 子采样旧题维）；需 GPU |

**ICD 不在这张图里**：ICD 是单遍流式方法，新题阶段是否更新参数由 `turning_point()`
按分布漂移量门控决定（实测本次 25 个 chunk 全部未触发，零样本泛化），没有"随 epoch 反复
训练收敛"这个过程，跟本图 x 轴（重复训练进度）定义的东西不是一回事，放进来会误导读者。
图上留了一行脚注说明这点。ICD 仍出现在 `all_methods_math1_random_split.csv` 的最终指标
总表里（那里比的是各方法跑完自己协议后的结果，不涉及训练进度，比较成立）。
ICD 曲线数据脚本 `experiments/run_icd_math1_curve.py` 保留作旁证，但不接入本图。
