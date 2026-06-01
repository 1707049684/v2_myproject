# 任务规划：G-NCDM 增量学习 — 修复旧任务性能退化 + Math1 调优

## 目标（已根据诊断更新）
**先修复根因，再调优。** 诊断发现 ACC_old≈0.62 退化**不是 Dynamic DNA 的问题**，而是
`run_incremental_real.py` 给生成式模型喂了**全零作答日志**，导致 θ/ψ 恒定、模型不区分学习者
（详见 findings.md 根因记录）。修复数据管线后，旧任务指标应回到论文量级（~0.75），再做超参调优。
脚本：`GNCDM/experiments/run_incremental_real.py`，结果写 `incremental_result/incremental_results.csv`，固定 `set_seed(42)`，用 GPU。

## 成功标准（草拟，待用户确认数值）
- [ ] Ours(DNA) 与 Ours(LoRA) 的**新题 AUC** 相比当前 baseline 提升（目标值待定）
- [ ] **旧题 AUC 不下降**、**TMD 不增大**（保旧知识）
- [ ] Ours 明显优于 Naive-FT，且接近 Oracle（差距 < 待定阈值）
- [ ] 最优配置可复现（seed=42 重跑一致）

## 阶段

### 阶段 1：诊断根因 ✅ 完成
- [x] 查看已记录结果 → 发现 DNA old 指标 = Base old 指标，退化不在 DNA
- [x] 勘察 forward 路径 → 确认所有 forward 喂 `torch.zeros` 作答日志（根因，见 findings.md）

### 阶段 2：修复数据管线（喂真实作答）⏳ 进行中
- [x] 最小验证：base 喂真实 vs 全零日志 → ACC 0.91 vs 0.38，根因坐实（verify_base_realfeed.py）
- [ ] 让实验 Dataset 提供真实作答向量（复用 core/train.py 的 IDCDataset，或内联构建 log_mat）
- [ ] 训练 + 评测的 forward 改用真实 `user_log/item_log`（base 与各增量策略 6 处都改）
- [ ] **避免泄漏**：评测诊断输入用训练集作答 / 走 forward_using_buf（验证里的 0.91 含泄漏，真实应 ~0.75）
- [ ] 处理增量阶段 log 维度与扩展后 item 空间对齐

### 阶段 3：重跑 + 定位真根因 ✅ 完成
- [x] 修复版无泄漏重跑 6 策略（incremental_results_fixed.csv）+ 完整模型校准（AUC 0.76）
- [x] 用户提示 Ablated 也应只训 'new' → 实测 expand_topology 后可训练 14 参数，含 4 个聚合矩阵未被 'new' 过滤捕获
- [x] **真根因**：DNA 漏训聚合矩阵新列 → 新题学不动(0.53)；Ablated 训整个聚合矩阵 → 破坏旧列(old 0.47)
- [x] 正解：只训聚合矩阵新列 `[:, 7:]`，冻结旧列（详见 findings.md 修复方向 A/B/C）

### 阶段 4：实现按列分离的聚合矩阵训练（待用户确认方案）
- [ ] 选定方案（A 梯度 mask / B 侧分支化 / C 优化器+backward hook 清零旧列梯度）
- [ ] 在 DNA（及 LoRA）训练里实现「只更新聚合矩阵新列」
- [ ] 重跑验证：DNA 同时保旧(≈0.60)且学新(向 0.85 靠)，TMD 仍小

### 阶段 4：超参调优（修复确认后才有意义）
- [ ] 可调超参：`lr`、`n_epoch`、LoRA `rank`、微方差 scale
- [ ] **决策点**：增量训练 `nn.BCELoss` → `TopologyAwareDecoupledLoss`（论文卖点，当前未接入）
- [ ] 单因子优先扫描，结果追加对比表，选最优配置并复现验证

## 待澄清的问题
1. 成功标准的**具体数值**？（如「新题 AUC ≥ 0.X」「TMD ≤ Y」）——也可先跑 baseline 再定。
2. 是否同意把「BCELoss → TopologyAwareDecoupledLoss」纳入本轮实验？
3. 有无算力 / 时间约束（决定扫描规模）？是否有 GPU 可用，还是 CPU 跑？

## 关键决策记录
- （待填）损失函数选择、最终超参配置 …
