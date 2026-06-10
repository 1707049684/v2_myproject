# C-LoRA（基线）vs. Ours (LoRA)：机制区别与 Ours 的优势

> 配套文档：`DNA_vs_LoRA.md` 讲 **Ours 内部** Dynamic-DNA 与 LoRA 两种「学新分支」的差异；
> 本文讲 **Ours (LoRA)** 与**持续学习基线 C-LoRA** 的差异——二者都用「低秩适配」，但**保旧的
> 机制根本不同**：C-LoRA 用*软正交惩罚近似*不遗忘，Ours 用*硬架构隔离精确*零遗忘。

---

## 一句话区别

| | 保旧（不遗忘）的机制 | 学新的位置 | 旧任务结果 | TMD |
|---|---|---|---|---|
| **C-LoRA**（baseline） | **软**：冻结基座 + 在**共享层**挂 LoRA，加正交惩罚 `λ·‖W_base·ΔWᵀ‖²` 把增量*推向*旧权重零空间 | 写进**共享诊断层**的 ΔW（与旧知识同一权重空间） | **近似**不变（仍漂移） | **> 0**（再调也压不到 0） |
| **Ours (LoRA)** | **硬**：`_freeze_parameters()` 冻结全部旧参 + 旧聚合矩阵新列**零填充** | 写进**独立的新分支**低秩适配器（与旧路结构隔离） | **逐位 = Base** | **= 0**（数学保证） |

---

## 1. C-LoRA 基线（我们复现的两个变体）

C-LoRA（Continual LoRA + 权重级软正交惩罚）的范式是「**冻结基座 + 小幅低秩增量 + 正交约束**」：
增量阶段只训 LoRA 因子 `ΔW = scaling·(B@A)`，损失为

```
L_total = L_CE + λ_ortho · L_ortho ,
L_ortho = Σ_layers ‖ W_base.detach() · ΔWᵀ ‖²_F / ‖W_base‖²_F      （按层归一化）
```

正交惩罚把新学的 `ΔW` *推向*冻结基座权重 `W_base` 的（左）零空间，**软性地**减少对旧任务决策面
的干扰。`λ_ortho` 越大越偏保旧、越小越偏学新——存在**稳定性-可塑性权衡**，需扫描 λ。

我们实现了两个变体（论文两个都报）：
- **方案一 `cl_baselines_random_split.py` / `eval_all_methods_user_split.py`（CognitiveBackbone）**：
  LoRA 挂在通用 Embedding+MLP 骨干上。优点是「被广泛认可的通用 CL 基线」；缺点是骨干 ≠ G-NCDM，
  TMD 只能在 embedding 空间度量、**量级不可与 Ours 概念 θ 的 TMD 直接比**。
- **方案二 `gncdm_clora_baseline.py`（真·G-NCDM 骨干）**：LoRA 直接挂在 G-NCDM 的 GDF(f_nn/g_nn)
  与 IRF(ncd) 上，**与 Ours 同骨干**，TMD 在**概念 θ 空间、可与 Ours 直接比**——这是它的核心价值。

**关键点（无论哪个变体）**：C-LoRA 把新知识写进**共享层**的 `ΔW`。即使正交惩罚把 `ΔW` 推向旧权重
零空间，它仍与旧知识共用同一前向通路 → 旧任务表征**必然被扰动**（只是被软性压小）→ **TMD > 0**。

---

## 2. Ours (LoRA)：`expand_topology_lora`（见 `core/model.py`）

Ours 的低秩变体保旧靠的是**结构**，不是惩罚：

1. **硬冻结旧参**：`_freeze_parameters()` 把所有既有参数 `requires_grad=False`（梯度恒为 0，
   不是"被正则化变小"，是**根本不更新**）。
2. **独立新分支学新概念**：为新知识 ΔK 新建**专属**低秩适配器，与旧路**结构隔离**：
   - 诊断（GDF）：`A_new_f@B_new_f`、`A_new_g@B_new_g`（微方差初始化 `*1e-3`）；
   - 聚合（新概念列）：`W_theta_new = |A_theta_agg @ B_theta_agg|`、`W_psi_new = |A_psi_agg @ B_psi_agg|`
     —— 这是**独立的低秩矩阵**，**不是**往旧 `theta_agg_mat` 里加 ΔW。
3. **旧聚合矩阵的新列零填充**（`torch.zeros`）：旧概念的聚合通路完全不变。
4. **预测时按列分离**（`predict_response`）：旧概念 θ 走冻结旧路、新概念 θ 走新分支，互不干扰。

结果：旧任务的 θ/ψ 与重构输出**逐位等于 Base** → **TMD = 0（精确，非近似）**。新知识全部落在
一个**结构上独立**的低秩分支里，对旧权重零扰动。

---

## 3. 核心区别（为什么"都用低秩"却天差地别）

| 维度 | C-LoRA（baseline） | Ours (LoRA) |
|---|---|---|
| 增量写在哪 | 共享层的 `ΔW`（旧知识同一权重空间） | **独立新分支**（结构隔离） |
| 旧参处理 | 冻结**值**，但其前向输出被 `ΔW` 改变 | 冻结**且旁路**：旧路输出完全不变 |
| 保旧靠什么 | **软正交惩罚**（超参 λ 控制强度） | **架构隔离 + 零填充**（无超参，硬保证） |
| 新概念维度 | 需额外解冻聚合新列才能学（否则学不动） | 设计上就有专属新聚合低秩矩阵 |
| 稳定性-可塑性 | **权衡**：λ 大保旧弱学新、λ 小反之，**无单一 λ 两全** | **无权衡**：旧=Base 与 学新 同时达成 |
| 遗忘度 TMD | **> 0**，软约束只能逼近 | **= 0**，数学保证 |
| 是否需调 λ | 是（要扫描） | 否 |

---

## 4. Ours 的优势（含实验证据）

### 优势一：**精确零遗忘**，而非"逼近"
软正交惩罚是把干扰**压小**，不是**消除**——`ΔW` 与 `W_base` 的乘积只能趋近、无法恒等于 0。
而 Ours 的旧路被结构旁路，输出逐位不变。**同骨干、同 TMD 空间**（方案二）下的铁证：

| a0910 random（同 G-NCDM 骨干） | AUC_old | AUC_new | TMD(concept-θ) |
|---|---|---|---|
| Base（旧任务上界参照） | 0.7441 | – | – |
| **Ours (LoRA)** | **0.7441（=Base）** | 0.7401 | **0** |
| **Ours (DNA)** | **0.7441（=Base）** | 0.7361 | **0** |
| G-NCDM + C-LoRA（λ=10，最偏保旧） | 0.7398 | 0.7211 | 0.0142 |
| G-NCDM + C-LoRA（λ=0.5，均衡） | 0.7257 | 0.7392 | 0.0220 |
| G-NCDM + C-LoRA（λ=0，无约束） | 0.6387 | 0.7409 | 0.0298 |

→ C-LoRA 把 λ 调到最大，TMD 也只到 **0.0142（仍 > 0）**、AUC_old **0.740 < Base 0.744**（仍有微遗忘）；
**没有任何单一 λ 能同时达到 旧=Base 且 TMD=0**。Ours 两项都**恒**满足。

math1 上差距更醒目（小数据、共享层漂移更明显）：C-LoRA 的 TMD 高达 **0.17~0.28**、AUC_old 掉到
**0.67~0.70**（Base/Ours 为 0.807）；Ours 仍 **0.807 / TMD=0**。

### 优势二：**没有稳定性-可塑性窘境，无需调 λ**
C-LoRA 必须在「保旧」与「学新」之间用 λ 取舍（扫描曲线见上：λ↑ 则 old↑/new↓）。Ours 因为旧路被
冻结旁路、新知识进独立分支，**保旧（=Base）与学新互不掣肘**，无需任何权衡超参。

### 优势三：新知识**结构可定位、可解释**
Ours 的新概念诊断/聚合都在带 `new` 名字的独立低秩参数里，物理隔离、便于分析与扩展；C-LoRA 的
新知识弥散在共享层的 `ΔW` 中，与旧知识纠缠。

### 优势四：C-LoRA"学新略好"恰恰反衬 Ours 的设计正确
方案二中 C-LoRA 在低 λ 下 AUC_new 偶尔略高于 Ours（如 a0910 λ=0.1 新 0.743 ≈ Ours 0.740；
math1 λ=0.01 新 0.769 > Ours 0.720）——这是因为它**改了共享诊断层 f_nn/g_nn**，对新题更灵活，
但**代价就是漂移旧概念（TMD>0、AUC_old 掉）**。这正说明："想用共享层多学一点新的，就必然牺牲旧的"；
**唯有 Ours 的"冻旧 + 专用新分支"能两全**。

---

## 5. 写论文的红线（务必遵守）

- **方案一**（CognitiveBackbone）：骨干 ≠ G-NCDM，其 TMD 在 **embedding 空间**，量级**不可**与 Ours
  概念 θ 的 TMD 比，只能说"TMD>0、未达零遗忘"；AUC/ACC/F1/RMSE 同划分同口径可比。
- **方案二**（G-NCDM 骨干）：**同骨干、TMD 同概念 θ 空间**，可与 Ours **逐项直接对比**（推荐作主对照）。
- C-LoRA 是**调校到位的强基线**（a0910 上仅微遗忘、AUC 接近 Ours），不是稻草人——Ours 的卖点是
  **精确 TMD=0 + 略高保旧 + 无需 λ 权衡**，而非碾压式领先。诚实陈述反而增强可信度。

---

## 6. 复现入口

- Ours（含 LoRA/DNA 六策略）：`experiments/run_incremental_{math1,a0910}_{random,user}_split.py`（核心库在 `experiments/_core/`）。
- C-LoRA 方案一 + 三基线合表：`cl_baselines_random_split.py`（random）、
  `experiments/_core/eval_all_methods_user_split.py`（user，support/query）。
- C-LoRA 方案二（G-NCDM 骨干）：`gncdm_clora_baseline.py`（`python gncdm_clora_baseline.py a0910`）。
- 结果：`incremental_result/all_methods_*`、`clora_gncdm_lambda_sweep_random_split.csv`（math1+a0910 合一，含 `dataset` 列）。
