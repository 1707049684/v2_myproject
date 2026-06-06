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

## 2026-06-06 — 会话：第二十五轮（帕累托图 + 主对比表）
- 完成第二十四轮待办：`plot_pareto_frontier.py`（读已提交 CSV，可复现）产出
  - `incremental_result/cl_baseline_sweeps_a0910_random_split.csv`（EWC/C-LoRA 各 6 点 + DER 1 点，落盘可复现）
  - 主表：`incremental_result/main_comparison_a0910_random_split.csv`（数据）+ `docs/main_comparison_a0910_random_split.md`（人读版，带脚注）
  - `docs/pareto_frontier_a0910_random_split.png`（稳定性-可塑性前沿，Ours 右上支配 CL 基线；只存 png）
- 红线守住：坐标轴只用 AUC（不同骨干不当纯策略比）；TMD 仅文字标注、不上同一数值轴。
- findings.md 待办全部清空（剩 3 项均为可选/待用户执行，非阻塞）。
- **当前状态**：未提交。等待用户决定是否 commit + push v2。

## 2026-06-06 — 会话：第二十六轮（三 CL 基线 user_split 合一脚本）
- 删除 `GNCDM/plot_pareto_frontier.py`（用户认为无用）。
- 新建 **`GNCDM/a0910_cl_baselines_user_split.py`**：一次跑完 EWC/DER++/C-LoRA 三基线的 a0910 **user_split**。
- 拍板冷启动协议 = **Recon-mirror**（对齐 Ours/CDAE 重构口径）+ EWC/C-LoRA 完整 6 点 λ 扫描 + DER 单点。
  - 原因：user_split test 用户与训练互斥（train∩test=0），transductive 骨干须冷启动；对比实验须全员同口径，本论文 user_split=重构，故基线也用重构。
  - `coldstart_recon_eval`：冻结 item_emb+MLP(+LoRA)，给 test 用户用其全部作答梯度拟合 student_emb，再 old/new 分别重构。
- avalanche 改惰性导入（本地无 avalanche 也能跑 C-LoRA）；训练固定 epoch（user_split valid 也互斥）。
- 本地冒烟测试 C-LoRA 全链路通过（数据维度对、冷启动 AUC 有意义）；EWC/DER 需 GPU 服务器装 avalanche 实跑。ruff 通过。
- 输出 4 个 csv 到 `incremental_result/`（含 `common_cl_baselines_a0910_user_split.csv` 合并总表）。

### 更正（同日）：Recon-mirror 记忆泄漏 → 改 Support/Query 留出
- 用户服务器实跑发现基线 ACC/AUC 高过 Ours 20 点（EWC AUC 0.93 等）→ 根因：在全部作答上拟合 student_emb 又考同一批题，背下自己标签（协议缺陷，非 bug）。
- 改 `coldstart_recon_eval` + `load_a0910_user_split` 为 support/query 留出（按用户切 0.5，support 拟合、query 评测、不相交）。冒烟验证无泄漏、数字回落到 AUC 0.68~0.71（低于 Ours，合理）。
- 🔴 遗留 TODO：Ours user_split 仍用全向量 recon，须也改成 support/query 才能与基线同表对比（用户要求本轮只改基线、已提醒）。服务器需重跑该脚本，旧 0.9+ csv 作废。

## 2026-06-06 — 会话：第二十七轮（Ours user_split 也改 support/query，独立脚本）
- 用户服务器重跑基线得健康数字（EWC 0.706/0.681、DER 0.680/0.666、C-LoRA λ=10 0.684/0.689，全低于 Ours，反超消失）。
- 用户拍板「改 Ours，但若有损文件风险就单独做脚本」→ 新建 **`GNCDM/experiments/eval_ours_supportquery_user_split.py`**：import 主脚本全部函数、零侵入，只换评测为 support/query。
- 零侵入关键：复用 `evaluate_recon`，把 log_mat 换成仅 support 构建、eval_df 换成 query 行 → user_log 不含被预测项、无泄漏；训练逻辑完全不动。
- 本地 math1 冒烟通过：DNA/LoRA AUC_old=Base=0.6929、TMD=0；Ablated/NFT 旧崩。叙事不变。ruff（除 E402 同 runner 惯例）通过。
- 服务器待跑该脚本得 a0910 Ours support/query 数字，再与基线总表同口径合表 = 论文 user_split 主对比表。

## 2026-06-06 — 会话：第二十八轮（九方法统一脚本 + math1 冷启动隐患）
- 用户要求把 6 Ours + 3 基线放一个脚本、math1 也跑 → 新建 **`GNCDM/experiments/eval_all_methods_user_split.py`**：一份 support/query 切一次共用，9 方法同 query 行评测，一张合并表。默认 math1，`RUN_A0910=True` 加跑 a0910。
- ⚠️ math1 冒烟（epoch 不对等）发现 C-LoRA 冷启动反超 Ours（非泄漏，是小题空间+推理期逐用户拟合的优势）；a0910 实跑无此问题。需服务器跑满 epoch 核实，再定 math1 baseline 的呈现方式（见 findings 第二十八轮）。

## 2026-06-06 — 会话：第二十九轮（实跑结果 + 分析 + 论文主表）
- 服务器实跑两表（comparison_all_methods_{math1,a0910}_user_split.md）已分析（findings 第二十九轮）：
  - **a0910（主对比成立）**：Ours DNA/LoRA 旧=Base、TMD=0；LoRA new AUC 0.707 ≥ 全部基线；唯一瑕疵=基线绝对 AUC_old 略高（骨干+冷启动差异，非遗忘）。
  - **math1（协议退化）**：仅 7 新题→support 新题作答太少→Ours new 近随机(LoRA 0.506)、基线两轴反超。建议 math1 不进基线主表。
- 4 个结果文件复制进 `GNCDM/incremental_result/`；新建论文主表 `GNCDM/docs/main_table_a0910_user_split.md`（分组+Backbone 列+加粗+完整 caveat）。

## 2026-06-06 — 会话：第三十轮（incremental_result 清理 + 总表统一命名）
- 删 `common_cl_baselines_a0910_random_split.csv`（冗余，已并入 random 总表）。
- 删旧口径（全向量 recon）的 `incremental_results_{a0910,math1}_user_split.csv`（被新 support/query 总表取代；可由 run_incremental_*.py 重新生成）。
- **总表统一命名 `all_methods_{dataset}_{split}.{csv,md}`**（去掉 `comparison_`/`main_comparison_`）：
  - user_split 两表 + random_split 总表（原 main_comparison）都改名；docs 的 random md 同步改名；`eval_all_methods_user_split.py` 的 write_tables 输出名同步改为 `all_methods_{split}`。
  - `docs/main_table_a0910_user_split.md`（论文主表，独立命名保留）的 source 路径已更新指向 `all_methods_a0910_user_split.csv`。
- 两个边界文件（cl_baseline_sweeps / base_alpha_sweep）按用户要求保留。
- 当前 `incremental_result/` 留：all_methods_{a0910_random,a0910_user,math1_user}（总表）+ incremental_results_{a0910,math1}_random_split（Ours RQ2 预测口径原始输出）+ 两个 sweep。

## 2026-06-06 — 会话：第三十一轮（math1 random 三基线 + 合并总表脚本）
- math1 random_split 此前无三基线结果 → 新建 **`GNCDM/math1_cl_baselines_random_split.py`**：EWC/DER/C-LoRA 一次跑完（random 无需冷启动，直接预测）+ 自动合并 Ours csv → `all_methods_math1_random_split.{csv,md}`。
- 本地冒烟（C-LoRA+合表）通过，ruff 通过。EWC/DER 需 avalanche → 服务器跑 `cd GNCDM && python math1_cl_baselines_random_split.py`。
- ⚠️ 冒烟 C-LoRA 在 math1 又偏高，但 random 无冷启动是干净口径，可能是真实信号；待服务器跑满 epoch 核实。
- 服务器实跑结果到（all_methods_math1_random_split.md，已复制进 repo）：**保旧 Ours 最强**（DNA/LoRA=Base=0.807、TMD=0），但**学新 Ours 弱**（DNA 0.720/LoRA 0.671 < 三基线 0.79~0.84）——真实信号（7 新题致 G-NCDM 新分支欠数据）。跨四设置定论：**主对比用 a0910（Ours 全胜）；math1 不作 baseline 可塑性主对比**（见 findings 第三十一轮）。
- **当前状态**：未提交。

## 2026-06-06 — 会话：第三十二轮（脚本整合：只留覆盖型）
- 把 random 基线泛化为 **`GNCDM/cl_baselines_random_split.py`**（覆盖 math1+a0910，三基线+合并九方法总表，RUN_A0910 开关），补上 a0910 random 的统一覆盖缺口。
- 删除被覆盖的单脚本：a0910_{clora,der,ewc}_baseline.py、math1_der_baseline.py、math1_cl_baselines_random_split.py、a0910_cl_baselines_user_split.py、experiments/eval_ours_supportquery_user_split.py。
- CL 基线脚本最终只剩两覆盖型：`cl_baselines_random_split.py`（random）+ `experiments/eval_all_methods_user_split.py`（user）；`a0910_gncdm_clora_baseline.py`（方案二负结果）待用户定去留。
- ⚠️ 教训：冒烟测试勿用真实 cfg 写真实输出路径（本轮误覆盖+删 all_methods_math1_random_split，已从 WPS 恢复）。
- **当前状态**：未提交。

## 2026-06-06 — 会话：第三十三~三十四轮（方案二救活 + 机制文档）
- 方案二 `a0910_gncdm_clora_baseline.py` → 重命名 **`gncdm_clora_baseline.py`**（参数化 math1/a0910，命令行选数据集），三处最佳努力修复（归一化惩罚/细扫 λ∈(0,1)/解冻新概念聚合列）全部生效：AUC_new 从 ~0.5 救回 0.72~0.77，干净权衡曲线。
- a0910 实测：C-LoRA 最佳 TMD=0.0142>0、AUC_old 0.740<Base 0.744（微遗忘），Ours 旧=0.744/TMD=0。方案二同骨干、TMD 同空间 → 比方案一更硬的对照。
- 用户拍板两个变体都报；新建 `GNCDM/docs/CLoRA_vs_Ours_LoRA.md` 讲机制区别 + Ours 优势。
- **当前状态**：未提交。

## 2026-06-06 — 会话：第三十五轮（接入解耦损失，受控测试）
- 用户拍板「接入解耦损失先跑测试看有无效果」→ 新建 **`GNCDM/experiments/eval_decoupled_loss_math1.py`**（零侵入，import 主脚本全部函数，仅加 `train_decoupled` 混态批 + TopologyAwareDecoupledLoss 循环）。math1 random_split，5 策略受控对比（3/4/5 同 oracle 全参，只差损失+数据流）。
- **结果（findings 第三十五轮）**：解耦损失**有效但有限**——TMD 0.020（比 NFT 0.064/Replay 0.076 低 3~4×）、AUC_new 0.852 最高（可塑性最佳）；**但 ACC_old 0.686 全场最差**，因 L_old 只蒸馏 θ、不管 ψ/agg/ncd。Ours-DNA 架构隔离两端通吃(TMD=0 且旧=Base)，**严格碾压软损失**。→ 强化论文主卖点，解耦损失宜定位为 ablation。
- 结果落 `incremental_result/decoupled_loss_test_math1_random_split.csv`。**当前状态**：未提交，待用户定解耦损失去留/是否扩展蒸馏到 ψ。

## 2026-06-06 — 会话：第三十六轮（扩展蒸馏 θ→θ+ψ→θ+ψ+resp）
- 用户选「扩展蒸馏到 ψ/agg」→ 给 `eval_decoupled_loss_math1.py` 加 `train_decoupled_ext`（零侵入，不改 loss.py），三档消融。
- **结论（findings 第三十六轮）**：①特征级蒸馏（θ、θ+ψ）救不回 ACC_old（仍 ~0.68，下游 agg+ncd 不受约束）；②**响应级蒸馏（旧题预测 KD）才是钥匙**——θ+ψ+resp 使 AUC_old=0.807=Base、ACC_old 0.725≈Base、TMD 0.019；③完整解耦损失是 DNA 外一个有竞争力工作点：旧≈Base **且可塑性更高**（AUC_new 0.829 vs DNA 0.720），以放弃精确零遗忘换学新能力；④DNA 仍独占精确零遗忘（TMD=0/旧=Base 逐位）。二者互补构成 stability-plasticity 谱系两端。
- CSV 更新为 7 行。**当前状态**：未提交。待用户定是否把 θ+ψ+resp 提升为正式软变体并接主实验。
- **补（同轮）**：用户选「a0910 上复现谱系」→ 脚本参数化并重命名 `experiments/eval_decoupled_loss.py`（`DATASETS` 配置 + 命令行选数据集，a0910 用 auto_new_concepts）。本机重跑 math1 无回归，ruff/py_compile 通过。服务器待跑 `python experiments/eval_decoupled_loss.py a0910`（GPU）。

## 2026-06-06 — 会话：第三十七轮（a0910 实跑 + 论文归属澄清）
- 用户服务器跑出 a0910 结果（存 `incremental_result/decoupled_loss_test_a0910_random_split.csv`）。**复现**：特征蒸馏不足、响应蒸馏救回旧精度（θ+ψ+resp AUC_old=0.742=Base）。**推翻**：math1 上「软损失可塑性 > DNA」是小数据假象——a0910 上两者打平（0.735≈0.736）。→ **真实大数据集 DNA 严格占优**（同可塑性 + 精确 TMD=0 + 更简单），解耦损失宜作 ablation。详见 findings 第三十七轮。
- **澄清用户疑问**：`TopologyAwareDecoupledLoss` **不是原论文（Toward Fair…，arXiv:2507.09831）的**——原论文用交叉熵、只覆盖新学习者；该损失是本项目增量扩展（git 同在"第一版上传我的项目代码"、在 incremental/），原论文没有、此前也未接主实验。
- 用户要求出文档 → 新建 **`GNCDM/docs/DecoupledLoss_vs_BCE.md`**：详解解耦损失机制（时空权重 + L_old 蒸馏 + L_new BCE）+ math1/a0910 实证 + 四条"为何不提升"原因（只蒸馏 θ 损旧精度／须补 response-KD／TMD 压不到 0／a0910 可塑性也只打平）+ 结论选 BCE。
- 结论已定 → 用户要求删除原始 CSV：已删 `incremental_result/decoupled_loss_test_{math1,a0910}_random_split.csv`（数据已固化在 `docs/DecoupledLoss_vs_BCE.md` + findings 第三十六/三十七轮表格）。
- **用户工作习惯（已存记忆 [[keep-auxiliary-scripts-out-of-repo]]）**：主实验用不到的工具型/探索脚本不留 repo，结论入文档后即删、需要时对话里重建。据此删除 `experiments/eval_decoupled_loss.py`（结论见 `docs/DecoupledLoss_vs_BCE.md` + 记忆 [[decoupled-loss-conclusion]]，可按需重建）。
- 清理：合并 `clora_gncdm_lambda_sweep_{math1,a0910}_random_split.csv` → 单文件 `clora_gncdm_lambda_sweep_random_split.csv`（加 `dataset` 列，16 行），删两源文件。注：`gncdm_clora_baseline.py` 重跑仍按数据集分别生成、需再合并。
- 用户自行整理论文表：`main_table_*` → 去 "main_" 改名 `docs/table_{a0910_user,math1_random}_split.md`（math1 表移进 docs/ 与 a0910 并列）。
- 收尾清理（用户确认）：① 删 `incremental_results_math1_random_split.csv`（冗余，六策略已在 `all_methods_math1_random_split.csv`）；② 清除 `haha`/`hello` 杂项出版本控制；③ `CLoRA_vs_Ours_LoRA.md` 结果引用改精确文件名。
- **当前状态**：未提交（建议下一步统一 commit + push v2）。
