---
name: run-incremental
description: 运行 GNCDM 增量学习主实验（Dynamic DNA vs LoRA vs Full-Replay vs Naive），并汇报结果。当用户要求跑增量实验、复现增量学习结果、或对比增量策略时使用。
disable-model-invocation: true
---

运行 GNCDM 的增量学习实验。

参数 `$ARGUMENTS`：数据集名，`math1`（默认）或 `a0910`。

步骤：
1. `cd GNCDM/experiments`
2. 根据数据集运行对应脚本：
   - `math1` → `python run_incremental_math1.py`
   - `a0910` → `python run_incremental_a0910.py`
3. 两个脚本都自动跑 random_split 和 user_split 两个划分（random 走预测口径 forward_using_buf，user 走重构口径 forward）。实验默认用 GPU；无 GPU 时脚本会自动回落到 CPU（a0910 题量大，建议在 GPU 服务器上跑）。
4. 结果写入 `GNCDM/incremental_result/incremental_results_{split}.csv`（math1：`_random_split` / `_user_split`；a0910：`_a0910_random_split` / `_a0910_user_split`）。运行完读取对应 CSV，用表格汇报各策略在旧/新测试集上的 AUC、RMSE、ACC、F1 以及 TMD（旧知识漂移）。
5. 重点对比 **Ours (Dynamic DNA)** 与 Naive Fine-Tuning（下界）、Full Replay Oracle（上界）的差距。

不要修改模型的微方差初始化（`* 1e-3`）或固定随机种子 `set_seed(42)`，除非用户明确要求。
