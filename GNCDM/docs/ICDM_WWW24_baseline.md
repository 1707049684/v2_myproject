# ICDM-WWW24 baseline 迁移说明

本仓库新增 `ICDM-WWW24 (adapted)`，用于补充现有 `ICD`（KDD'22）之外的图归纳认知诊断对照。

## 定位

- `ICD`：面向交互流的 Incremental Cognitive Diagnosis。
- `ICDM-WWW24`：面向未见学生的 Inductive Cognitive Diagnosis Model。
- 本适配器：保留 ICDM 的归纳图编码，并按本项目的旧题→新题两阶段协议顺序训练。

官方仓库没有明确 LICENSE，且依赖 `torch==1.13.1`、`dgl==1.1.2`、`numpy==1.23.5`、
`pandas==1.5.2`，与主环境冲突。因此这里没有复制官方源码，而是在当前 PyTorch 环境独立重实现：

1. Q 图上的 item/concept 多跳传播；
2. 正确与错误作答的双通道聚合；
3. 多图来源 attention 融合；
4. 基于 Q mask、题目区分度和 mastery/difficulty 差的 GLIF-style 预测；
5. 未见学生只使用训练所得 population prior 和自身 support edges，不读取未训练的 user-id embedding。

核心代码：

- `baselines/icdm_ww24.py`：模型与图表示；
- `experiments/_core/run_icdm_ww24.py`：协议、训练、评测和并表；
- `experiments/run_incremental_math1_{user,random}_split.py`：现有两个主入口。

## 无泄漏协议

训练阶段按用户将交互分成两个互斥 fold，逐 epoch 交替：

```text
odd epoch:  fold A 作为 response graph，fold B 作为 prediction target
even epoch: fold B 作为 response graph，fold A 作为 prediction target
```

因此 target response 不会在同一次 forward 中以“做对/做错”图边出现。validation/test 的 query 行也永远
不会进入 response graph。

### math1_user_split

- 参数学习用户：`train.csv` 用户；
- 新用户 support/query：与主管线相同，`frac=0.5, seed=7`；
- support 进入 response graph；
- query 按旧题/新题拆分后计算 AUC/RMSE/ACC/F1。

### math1_random_split

- test 用户与 train 用户共享；
- response graph 只包含 train 交互；
- test 交互不进入图，直接按旧题/新题预测。

两个划分都使用与主实验相同的严格拓扑二分：`new_concepts=[0,1,3,6]`，即 13 旧题/7 新题、
7 旧概念/4 新概念。

## 运行

只运行 ICDM 并把结果 upsert 到已有主表：

```bash
cd GNCDM/experiments
python run_incremental_math1_user_split.py --icdm-only
python run_incremental_math1_random_split.py --icdm-only
python run_incremental_junyi_random_split.py --icdm-only
python run_incremental_a0910_random_split.py --icdm-only
```

完整入口默认会在原有方法完成后追加 ICDM；如需保持旧行为：

```bash
python run_incremental_math1_user_split.py --skip-icdm
python run_incremental_math1_random_split.py --skip-icdm
```

可用 `--icdm-epochs N` 修改每阶段最大 epoch。默认 `25`，按 validation AUC 早停，patience 为 5。

## 当前结果

固定 `seed=42`、`dim=64`、`lr=2e-3`、CPU，每个划分运行一次：

| Split | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | RD* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math1_user_split | 0.7522 | 0.8202 | 0.4701 | 0.4147 | 0.6000 | 0.7508 | 0.4273 | 0.6062 | 0.2899 |
| math1_random_split | 0.7427 | 0.8185 | 0.4928 | 0.4051 | 0.5592 | 0.7543 | 0.2931 | 0.6093 | 0.4219 |
| junyi_random_split | 0.6733 | 0.6801 | 0.4420 | 0.4616 | 0.7116 | 0.6672 | 0.8230 | 0.7597 | 0.0720 |
| a0910_random_split | 0.6154 | 0.6810 | 0.4669 | 0.4571 | 0.6686 | 0.6761 | 0.7903 | 0.7746 | 0.0813 |

单行结果：

- `incremental_result/icdm_row_math1_user_split.csv`
- `incremental_result/icdm_row_math1_random_split.csv`
- `incremental_result/icdm_row_junyi_random_split.csv`
- `incremental_result/icdm_row_a0910_random_split.csv`

## 可比性红线

1. 名称必须保留 `(adapted)`；当前实现不是官方 DGL 代码的逐行复现。
2. ICDM 原论文研究新学生归纳，本适配器额外加入了 old→new 题目阶段，属于本项目协议适配。
3. 模型初始化时仍知道完整 item/concept 数量和 Q 矩阵，因此不是 Dynamic DNA 式的严格动态拓扑方法。
4. `RD*` 是相同旧题图上下文下、阶段 2 前后旧训练用户 mastery 的平均归一化 L2 漂移；其空间与
   G-NCDM 的概念 θ-RD、ICD 的 NCD trait RD 均不同，只能判断是否发生漂移，不能横向比较数值大小。
5. a0910 有 75 道题在原始 Q 矩阵中没有概念关联。它们在 Q 图传播中保持孤立 item，但预测时使用
   全概念 diagnostic fallback，使可学习 item representation 不会被全零 Q mask 消除。
