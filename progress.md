# 会话日志

## 2026-06-01 — 会话 1：规划启动
- 通过 /init 完成 Claude Code 配置：CLAUDE.md、两个 skill（run-incremental / train-gncdm）、ruff 格式化 hook、pyproject.toml、.gitignore、tests/（3 个冒烟测试，已通过）。
- 已 `git push -u v2 main`，main 跟踪 v2/main。
- 建立论文软链接 `docs/paper.pdf` → Zotero PDF，通读全文（IEEE TLT 2026）。
- 创建规划文件 task_plan.md / findings.md / progress.md。
- **当前状态**：等待用户确认本次规划的具体目标（见 task_plan.md「待澄清」）。

- 目标确认：**Math1 数据集 + 超参调优**提升 Ours(DNA/LoRA)。
- 勘察 run_incremental_real.py：列出可调超参清单；**发现增量训练用 BCELoss 而非 TopologyAwareDecoupledLoss**（写入 findings.md）。
- task_plan.md 细化为 4 阶段（建 baseline → 定搜索空间 → 扫描 → 分析定稿），当前进行阶段 1。

- 用户补充：已跑过 baseline（见 incremental_results.csv）；GPU 可用；同意纳入损失切换。
- 用户报告：ACC_old 降到 ~0.63、F1_old ~0.68（论文 ~0.75），怀疑 Dynamic DNA 写错。
- **诊断完成（阶段1）**：
  - 数据事实：DNA 的 old 指标 = Base 的 old 指标（0.619/0.681），TMD≈0.0005 → **退化不在 DNA**。
  - 根因：实验所有 forward 喂 `torch.zeros` 作答日志（内联 IDCDataset 不返回作答向量），
    导致 θ/ψ 恒定、模型不区分学习者，性能封顶在 ~0.62。详见 findings.md 根因记录。
- task_plan 重排为：阶段1 诊断✅ → 阶段2 修复数据管线（喂真实作答）→ 阶段3 重跑校验 → 阶段4 调优。

- **验证完成**：写 verify_base_realfeed.py，同一 base 模型仅换评测输入：真实日志 ACC=0.9123 / AUC=0.9784，全零日志 ACC=0.3807。根因 100% 坐实（详见 findings.md 验证结果）。注意 0.91 含信息泄漏（评测 log 含待预测题答案），修复后真实水平应 ~0.75。
- 顺手把 ruff 格式化 hook 命令改成 cwd 无关的内联 `python -c`（原相对路径在非根目录 cwd 下会失效）；需重载配置（/hooks 或重启）才生效。

- **修复版完成 + 重要更正**（建 run_incremental_real_fixed.py：真实日志训练 + buffer 无泄漏评测）：
  - 更正：verify 的 0.91 是泄漏虚高，不算数。无泄漏下 Base(13题)≈0.60，和原来 0.62 几乎一样。
  - 校准 calib_full_model.py：完整 20 题 AUC 0.76 / ACC 0.68 → item 越多越准；离 0.75 的差距是子集+训练预算，非 bug。
  - **结论1**：Dynamic DNA 不造成旧任务退化（两版 old≈Base），用户疑虑排除。
  - **结论2（新发现）**：无泄漏下 DNA/LoRA 新题 AUC≈0.53（近随机），Ablated/Oracle/NFT≈0.85。
    冻结旧参+只训 new+微方差 1e-3 可能太受限，新分支学不动新题——这才是核心研究问题。

- **正交掩码实现**：DNA 优化器加入聚合矩阵 weight + backward hook 冻结旧列 → 新题 0.57→0.86，但旧题掉到 0.455。
- **定位根因**：Q 实测旧题重度用到"新概念"，按索引切知识无效。用户拍板「按知识切题」严格拓扑二分。
- **二分分析**（analyze_q_bipartition.py）：索引默认严格规则下旧题=0；ΔK={0,1,3,6} 给 13旧/7新（匹配原比例且数学有效）。用户确认。
- **修复3 完成**：实现 strict_bipartition+重排接入脚本，重跑：
  - Base 跳回 0.681/0.765（不再被阉割）。
  - **DNA：旧任务=Base 逐位相同、TMD=0（零遗忘），新任务 0.734 ≈Oracle** → 设计达成！
  - Ablated 破坏旧(0.586)、NFT 灾难遗忘(TMD 0.089)、LoRA 保旧但学新弱(0.638)。叙事完全成立。

### 下一步（待用户定）
- LoRA 新题偏弱：低秩路径(A/B agg)未被 dense 正交掩码覆盖，需单独处理。
- 可选：接非对称混态流(1:3/1:4 对比)、换 TopologyAwareDecoupledLoss。
- 是否把修复回灌/替换原 run_incremental_real.py。
