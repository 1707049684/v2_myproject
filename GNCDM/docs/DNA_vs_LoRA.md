# Ours(DNA) vs Ours(LoRA):机制差异与大/小数据集表现翻转

> 适用于 G-NCDM 增量学习主实验(`run_incremental_math1.py` / `run_incremental_a0910.py`)。
> 代码出处:`GNCDM/core/model.py`,方法 `expand_topology`(DNA)、`expand_topology_lora`(LoRA)、`diagnose_theta`/`diagnose_psi`(前向)。

## TL;DR

- **保旧机制完全相同**:两者都靠"冻结全部旧参数 + buffer 零填充扩列 + 聚合矩阵旧列原样拷贝",所以旧任务输出**逐位不变**,`TMD = 0`(零遗忘)。`Ours(DNA/LoRA)` 旧任务恒等于 `Base` 是架构隔离的必然结果,不是 bug。
- **唯一区别在"学新概念的新分支怎么长"**:
  - **DNA** = 长出一个**稠密全连接子网**(lateral neural splitting)。
  - **LoRA** = 长出一个**低秩适配器**(rank-`r` 分解,`r` 默认 4)。
- **关键后果**:DNA 诊断 ψ 的新分支输入维度 = `n_user`,参数量 `O(n_user · ΔK)` **随用户规模膨胀**;LoRA 的对应分支复用旧网络隐层(维度 `n_know`),参数量 `O(r · (n_know + ΔK))`,**与用户数解耦**。
- **因此**:数据集越大、用户越多 → DNA 新分支越臃肿、越易过拟合稀疏的新概念信号 → **大数据集(a0910)上 LoRA 学新更好**;小数据集(math1)上 DNA 的额外容量反而能多拟合一点,**排名翻转**。

---

## 1. 共享的"零遗忘"骨架(两者一致)

`expand_topology` 与 `expand_topology_lora` 开头都执行:

```python
self._freeze_parameters()                       # 冻结所有旧参数
new_theta_buf[:, :self.n_know] = self.Theta_buf  # buffer 零填充扩列,旧列原样保留
new_psi_buf[:self.n_item, :self.n_know] = self.Psi_buf
# 聚合矩阵:旧列拷贝,只在新列上做微方差初始化
new_theta_agg_weight.data[:, :self.n_know] = self.theta_agg_mat.weight.data
```

新分支一律用 **微方差初始化**(`* 1e-3`,LoRA 注释 "MV-NN-LoRA",DNA 用 `_initialize_new_params_with_micro_variance`),保证扩展瞬间对旧任务输出零扰动、同时打破"零梯度死亡"。

**结论**:旧题、旧概念那部分的预测路径在两种方法下完全相同 → 旧任务指标逐位相等、`TMD = 0`。**保旧不构成两者的差异**。

---

## 2. 不同的新分支结构

诊断模块有两条:`f_nn` 生成学生能力 θ、`g_nn` 生成题目属性 ψ。增量时各自侧向生长一条"看新概念"的分支,再 `concat` 到旧输出后面。

### 2.1 θ 分支(`diagnose_theta`,`model.py:558-569`)

| | DNA | LoRA |
|---|---|---|
| 新分支输入 | 新题作答 `user_log[:, M_old:]`(维度 `ΔM`) | 同样是新题作答 `x_new`(维度 `ΔM`) |
| 结构 | 2 层稠密 `Linear(ΔM,ΔK)→Sigmoid→Linear(ΔK,ΔK)` | 单层低秩 `theta_new = σ(x_new · \|A_f B_f\|)`,`A_f∈ℝ^{ΔM×r}`、`B_f∈ℝ^{r×ΔK}` |
| 新参数量 | `ΔM·ΔK + ΔK²` | `r·(ΔM + ΔK)` |

> LoRA 在 θ 分支**故意只看新题原始作答 `x_new`**,不接旧隐层 `h_old`(代码注释:"彻底斩断对旧网络 h_old 的污染"),进一步降低旧→新串扰。

### 2.2 ψ 分支(`diagnose_psi`,`model.py:594-607`)—— **这里是大数据集差距的根源**

| | DNA | LoRA |
|---|---|---|
| 新分支输入 | `item_log`,首层 `Linear(n_user, ΔK)` → **输入维度 = n_user** | 复用冻结旧网络的隐层特征 `h_old_g = g_nn[0..3](item_log)`,**维度 = n_know** |
| 结构 | 3 层稠密 `Linear(n_user,ΔK)→…→Linear(ΔK,ΔK)` | 单层低秩 `psi_new = σ(h_old_g · \|A_g B_g\|)`,`A_g∈ℝ^{n_know×r}`、`B_g∈ℝ^{r×ΔK}` |
| 新参数量 | **`n_user·ΔK` + 2·ΔK²** | `r·(n_know + ΔK)` |

**核心**:DNA 的 ψ 新分支首层是 `nn.Linear(self.n_user, delta_K)`,参数量正比于 **用户数**;LoRA 把这块换成"在旧隐层(`n_know` 维)上叠 rank-`r` 适配器",**彻底不含 `n_user` 项**。

### 2.3 聚合矩阵新列

| | DNA | LoRA |
|---|---|---|
| `theta_agg` 新列 | 整列稠密 `user_dim·ΔK` | `r·(user_dim + ΔK)`(`A_theta_agg`/`B_theta_agg`) |
| `psi_agg` 新列 | 整列稠密 `item_dim·ΔK` | `r·(item_dim + ΔK)`(`A_psi_agg`/`B_psi_agg`) |

### 2.4 共享的残差 / α 混合

两者最终都做 `θ = θ_concat·(1-α) + residual_concat·α`(`model.py:584`),残差路径(基于 Q 矩阵的先验)与 α 完全相同 → **混合机制不是差异来源**。

---

## 3. 为什么大数据集上 LoRA 更好

设新概念相关的有效训练信号量近似固定(只有新题 / 新概念那部分作答带监督),记其规模为 `S`。

- **DNA ψ 新分支自由参数 ≈ `n_user · ΔK`**,随数据集规模(用户数)线性膨胀。
  - a0910 用户数远大于 math1 → 新分支参数爆炸,而监督信号 `S` 没同比增加 → **参数多、数据稀 → 过拟合 / 优化困难**,从 `~1e-6` 量级硬学一大片权重,新任务泛化受损。
- **LoRA ψ 新分支自由参数 ≈ `r·(n_know + ΔK)`**(`r=4`),**与 `n_user` 解耦**,是个天然强正则的小适配器,且复用了旧网络已学好的隐层表征 → 新概念数据稀也压得住,泛化更稳。

**一句话**:数据集越大、用户越多,DNA 的 `n_user×ΔK` 稠密新分支越吃亏;LoRA 的低秩适配器参数与规模解耦,所以大数据集上学新更稳更好,而保旧两者一样都是零遗忘。

---

## 4. 实证(a0910 random_split,findings 第十九/二十轮)

| 策略 | AUC_old | AUC_new | ACC_old | ACC_new | TMD |
|---|---|---|---|---|---|
| Ours (DNA) | 0.744 | 0.736 | 0.730 | 0.716 | **0** |
| **Ours (LoRA)** | 0.744 | **0.740** | 0.730 | **0.723** | **0** |
| Full Replay Oracle | 0.748 | 0.736 | 0.731 | 0.723 | 0.022 |

- **保旧**:DNA 与 LoRA 完全相同(`AUC_old=0.744`、`TMD=0`)——印证第 1 节。
- **学新**:大数据集上 **LoRA 略优**(new AUC 0.740 vs 0.736、ACC 0.723 vs 0.716)。

**小数据集 math1 排名翻转**:random 划分下 `DNA > LoRA`。小数据上 DNA 的额外容量没被规模惩罚,反而能多拟合一点;LoRA 的 rank 约束此时成了天花板。

---

## 5. 实践建议

### 5.0 `rank` 到底是什么

LoRA 不直接学一个大权重 `W`(形状 `d_in × d_out`),而是拆成两个瘦矩阵相乘:

```python
# model.py:348-349 / 602  (ψ 分支)
A_new_g : n_know × r     # 把输入从 n_know 压到 r 维(瓶颈)
B_new_g : r × ΔK         # 再从 r 维还原到 ΔK 维
W_new_g = |A_new_g · B_new_g|   # 等效得到 n_know × ΔK 的权重
```

中间那个细腰维度 `r` 就是 **rank**(默认 4)。由线性代数,`A·B` 的秩 ≤ `r`,即 LoRA **强制新分支权重秩 ≤ r**——等于规定"新概念与旧表征之间只允许用 `r` 个独立方向去表达"。

| | `r` 小(如 2) | `r` 大(如 32) |
|---|---|---|
| 新参数量 `r·(n_know+ΔK)` | 少 | 多 |
| 表达能力 / 容量 | 弱 | 强 |
| 过拟合风险 | 低(强正则) | 高 |
| 极限 | — | `r→min(d_in,d_out)` 时退化为≈无约束稠密层(即 DNA) |

> 直觉:DNA 的新分支≈"满秩稠密层"(`r` 拉满,参数随 `n_user` 膨胀);LoRA 用小 `r` 把参数压到极小且与 `n_user` 解耦,所以大数据集上更稳、更不易过拟合。

### 5.1 选择建议

- **大数据集 / 用户数多**(如 a0910、真实题库):优先 **Ours(LoRA)**,新任务泛化更好且显存/参数开销低。
- **小数据集**(如 math1):两者皆可,DNA 学新可能略占优;若追求一致口径仍可统一用 LoRA。
- **保旧诉求**:任选,二者都是零遗忘(`TMD=0`)。
- **可调超参**:LoRA 的 `rank`(默认 4)。大数据上适当增大 `rank` 可在"泛化"与"容量"间权衡;但不要回退到稠密 `n_user` 输入,那会丢掉与规模解耦的优势。

---

## 附:论文叙事红线

1. `Ours(DNA/LoRA)` 旧任务恒等于 `Base`、`TMD=0` 是**架构隔离的预期结果**,不是退化、不是 bug。
2. DNA↔LoRA 的差异**仅在学新分支**;保旧机制相同,不要把"LoRA 更好"误述为"LoRA 保旧更好"——它是**学新更好**。
3. 大/小数据集排名翻转有清晰机制解释(参数量是否随 `n_user` 膨胀),是论点而非噪声,建议在论文中明确给出。
