# 固定参数的五 seed 显著性流程

使用 [run_fixed_significance_profiles.sh](../scripts/run_fixed_significance_profiles.sh) 在服务器上按指定数据集运行固定 profile。它固定训练 seed 为 `1 7 21 42 84`，不会在本轮实验中扫描 λ 或 memory size，也不会覆盖既有的 `all_methods_*.csv` 单 seed 表。

```bash
cd GNCDM
ICD_PYTHON=/opt/icd-venv/bin/python \
  bash scripts/run_fixed_significance_profiles.sh --dataset math1 --device cuda:0
```

入口脚本现在必须显式指定 `--dataset math1`、`--dataset a0910` 或 `--dataset junyi`，因此不会默认启动全部数据集。`--dataset a0910` 会依次运行 A0910 的 random 和 user profile；如只需其中一个，可额外指定 `--split random` 或 `--split user`。只有显式传入 `--all` 才会运行四个预设 profile。

中断后使用 `--resume`。只有通过 protocol 和完整方法集合校验的 `trials/seed_<seed>.csv` 会被复用。

## 固定 profile

| 数据集 / 划分 | EWC λ | DER++ memory | C-LoRA λ | C-LoRA-GNCDM λ | X-DER memory |
|---|---:|---:|---:|---:|---:|
| Math1 / random | 10000 | 5000 | 10000 | 10 | 5000 |
| A0910 / random | 10000 | 5000 | 10000 | 10 | 5000 |
| A0910 / user | 10000 | 5000 | 10 | 10 | 5000 |
| Junyi / random | 1000 | 5000 | 10 | 0.1 | 5000 |

除 `a0910/user` 的 X-DER `mem=5000`（按本次指定）外，这些数值来自对应的既有 `all_methods_*.csv` 结果；本运行器将其记录在 `protocol_manifest.json` 的 fixed profile 中。

## 运行模型集合

随机划分运行以下十个结果行：

1. `G-NCDM(Anchor)`（现有 `Base`）
2. `EWC`
3. `DER++`
4. `C-LoRA`
5. `X-DER`
6. `Full-Replay`
7. `C-LoRA-GNCDM`
8. `ICD`
9. `CLEAN-Full`
10. `CLEAN-LoRA`

`CLEAN-Full` 和 `CLEAN-LoRA` 分别是现有 `Ours (Dynamic DNA)` 和 `Ours (LoRA)` 的正式显示名，不会重复训练。`a0910/user` 的 X-DER 固定使用 `mem=5000`，并按该 trial 的 support/query seed 与训练 seed 运行。`Full-Replay` 是 oracle 上界，默认写入结果但不加入现实持续学习方法的优越性检验。

`G-NCDM(Anchor)` 对应旧任务 `Base`，没有新任务指标，因此会写入每 seed 结果表，但不能参与默认 `Balanced_AUC` 配对检验。

## ICD 独立环境

ICD 使用 EduCDM，和主实验环境隔离。入口通过 `--icd-python`（或环境变量 `ICD_PYTHON`）在该环境中按相同训练 seed 启动 ICD 子进程，并把每个 seed 的单行结果写到 `icd_raw/seed_<seed>.csv`。若该解释器不能 `import EduCDM`，运行器会在训练前明确失败，不会用既有单 seed ICD 行替代多 seed 结果。

## 统计解释

默认目标是 `CLEAN-Full`，比较对象为实际可比较的 `CLEAN-LoRA`、EWC、DER++、C-LoRA、C-LoRA-GNCDM、ICD，以及 random profile 中的 X-DER。统计采用 paired exact sign-flip test、bootstrap CI 与 Holm 校正。

五个 paired seed 的双侧精确检验最小 p 值是 `0.0625`，因此在 `alpha=0.05` 下不能给出显著性拒绝结论；结果会被标记为 `exploratory_underpowered`，但仍保留效应量、置信区间和精确 p 值。
