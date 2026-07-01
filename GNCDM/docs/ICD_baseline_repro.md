# ICD 基线复现（math1 random_split）

EduCDM 的 **ICD**（Incremental Cognitive Diagnosis, KDD'22）作为增量对照基线，产出 `all_methods_math1_random_split.{csv,md}` 里的 `ICD` 一行。ICD 依赖偏老（pandas/torch 版本与主仓库冲突），**只能在独立 venv 离线跑**，产出单行后并入总表，不进主管线（`cl_baselines_random_split.py`）。

## 结果（已并入总表）

| AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD(RD) |
|---|---|---|---|---|---|---|---|---|
| 0.7870 | 0.7258 | 0.4681 | 0.4721 | 0.6897 | 0.7129 | 0.6358 | 0.5197 | 0.9539 |

口径：与主管线**完全相同**的 `strict_bipartition(Q, NEW_CONCEPTS=[0,1,3,6])` → 13 旧题/7 新题、7 旧概念/4 新概念；test 用户与训练共享（预测口径）。TMD 在 ICD 的 NCD trait 空间，量级不可与 Ours/其他基线比，仅看 >0。

## 环境（独立 venv，本机仅 Python 3.13）

```bash
python -m venv _scratch/icd-venv
_scratch/icd-venv/Scripts/python -m pip install torch==2.9.1 EduCDM==2.* -i https://pypi.tuna.tsinghua.edu.cn/simple
_scratch/icd-venv/Scripts/python -m pip install "pandas==2.2.3"   # EduCDM 拉进来的 pandas3 需降级
```
- torch 用 **CPU** 版即可（math1 极小）；GitHub 不可达时 torch wheel 可 `curl -C -` 断点续传后本地 `pip install`。
- 装好的依赖含 longling/baize/fire，**无 dgl/mxnet**。

### 必打的源码补丁（pandas≥2.2 兼容）

ICD 老代码 `EduCDM/ICD/etl/etl.py` 用 `df.groupby(['user_id'])`（列表参数），pandas≥2.2 返回**元组键** `(uid,)`，导致 `transform` 全 miss、数据为空（指标只剩 doa）。改成标量：

```python
# user2items / item2users 里：
grouped = df.groupby("user_id")   # 原 df.groupby(["user_id"])
grouped = df.groupby("item_id")   # 原 df.groupby(["item_id"])
```

## 官方超参（关键）

来自 EduCDM `examples/ICD/ICD.py` 的 `main()` 默认：`cdm=ncd`、`alpha=0.2`、`tolerance=0.2`、`beta=0.9`、`epoch=1`、`warmup_ratio=0.1`、`weight_decay=0`、`inner_metrics=False`。
**注意**：alpha/tolerance 拍错（如 0.9/1e-3）会让拐点门控几乎不触发、模型近乎不训 → test auc≈0.5 退化。example 里的 `"math"` 配置是另一个大数据集（10269×17747×1488），**非本仓库 math1**，只借超参、维度仍用 math1 的 4209/20/11。

## 数据适配

- **log**：`data/math1_{train,test}_0.8_0.2.csv` 列 `user_id,item_id,score` 与 ICD 天然兼容；先 `remap_items` 按 strict_bipartition 重排（旧题 id 在前）。
- **item.csv**：由重排后的 Q 生成，列 `item_id,knowledge_code`，`knowledge_code` 写 **1-indexed** list 串（ICD `item2knowledge` 内部减 `k_offset=1`）。

## 跑法（两脚本在 `_scratch/`，按 memory 约定不进 repo）

1. 训练协议 = **单条 old→new 数据流**（阶段1 旧题 chunks + 阶段2 新题 chunks，一次 `ICD.train`；拆成两次 `train()` 会让对偶机制坍缩、F1=0）。
2. 末态模型用自写 eval（net 前向 + sklearn 算 AUC/RMSE/ACC/F1）分别评 old-13/new-7 题 test 子集。
3. **RD/TMD**：另起一个 old-only 参考模型当"阶段1之后"基线，RD = 旧用户 NCD trait 的平均 L2 漂移（对齐项目 `baseline_tmd` 的 b0 口径）。

```bash
cd _scratch && ./icd-venv/Scripts/python -u run_icd_math1_A.py
# -> icd_out_A/icd_row_math1_random_split.csv（即并入总表的那一行）
```

固定 `torch.manual_seed(0)`（ICD 内部）；本文件记录的数值为该配置下的产出。
