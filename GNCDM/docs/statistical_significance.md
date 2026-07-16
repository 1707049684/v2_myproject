# 增量学习的正式统计显著性流程

本流程针对 G-NCDM 增量实验的随机训练不确定性，使用同一数据划分、同一
训练 seed 下的配对方法结果作为统计单位。它不会覆盖既有的
`all_methods_*.csv` 单次点估计表，而是写入独立、可审计的长表。

## 预注册口径

- 主终点默认是 `Balanced_AUC = (AUC_old + AUC_new) / 2`，以等权方式衡量旧任务保持与新任务适应。
- 主比较是 `Ours (Dynamic DNA)` 对各个预先声明的非 oracle 基线：
  `Ours-Ablated`、`Ours (LoRA)`、`Naive FT (NFT)`、`EWC`、`DER++`、`C-LoRA`。
- `Full Replay Oracle` 只作为上界展示，默认不进入优越性检验；`Base` 没有新任务指标，不能进入 `Balanced_AUC` 检验。
- 主检验为双侧精确配对 sign-flip permutation test。10 个 seed 时枚举全部 `2^10=1024` 种配对符号翻转。
- 以配对差值的 20,000 次 bootstrap 报告 95% percentile CI；Holm 校正在每个 `(dataset, split, protocol)` 的全部预先声明比较中进行。
- `TMD/RD` 不跨 G-NCDM 概念空间与 EWC/DER++/C-LoRA 的 embedding 空间比较。运行器会拒绝这类无效的混合检验。

请在运行 test 前固定指标、基线集合和 seed 列表。若论文的主张是单独的
`AUC_new` 或某个 RMSE，可在命令中用 `--metrics AUC_new` 或
`--metrics RMSE_new` 替换默认值；不要依据 test 结果再挑选指标。

最少 seed 数由 alpha 和预先声明的 Holm 检验族大小共同决定：单个比较、alpha=0.05 时至少
需要 6 个；默认的 6 个比较在 Holm 校正后至少需要 8 个。默认的 10 个 seed 是推荐方案。
重复 seed 会被运行器拒绝。仅作代码冒烟时可用 `--skip-analysis`，但该结果不得报告为显著性结论。

## 运行多 seed 实验

从 `GNCDM/` 目录运行。推荐使用预先固定的 10 个 seed：

```powershell
python experiments/_core/run_statistical_trials.py `
  --dataset math1 --split random `
  --seeds 1 7 21 42 84 100 2024 3407 7777 10000
```

user split 的主分析固定 support/query 划分，仅改变训练 seed：

```powershell
python experiments/_core/run_statistical_trials.py `
  --dataset math1 --split user `
  --support-query-seed 7 --support-frac 0.5 `
  --seeds 1 7 21 42 84 100 2024 3407 7777 10000
```

支持 `math1`、`a0910`、`junyi` 和 `random`、`user` 两种划分。中断后加
`--resume` 会复用完整的 `seed_*.csv` 文件。默认请求 deterministic cuDNN；若
硬件或算子不兼容，可显式加 `--non-deterministic`，该选择会记录在 manifest。

`--resume` 还会核对数据集、划分、protocol hash 与 seed；只要 alpha、概念划分、
support/query 设置或 deterministic 设置不一致，旧缓存就会被视为无效并重新运行，
避免混合不同 protocol 的结果。

在每个 seed 内，EWC 和 C-LoRA 的 lambda 都只用 valid 集的
`Balanced_AUC` 选择，然后以同一 seed 重训一次，并只在 test 上评估一次。
这消除了旧宽表中“按 test 指标选 lambda”的泄漏。

## 产物

以 Math1 random split 为例，产物位于：

```text
incremental_result/significance_trials/math1_random/
  protocol_manifest.json
  trials/seed_<seed>.csv
  ours_raw/seed_<seed>.csv
  per_seed_results.csv
  formal_significance_summary.csv
  formal_significance_tests.csv
  formal_significance_report.md
```

`per_seed_results.csv` 是唯一可用于统计推断的原始长表。每行对应一个
`dataset × split × protocol × seed × method`；指标缺失一律写为数值 `NaN`，不使用
`"-"`。`selected_hparams` 与 `selection_source` 记录每个 seed 的基线选参。

`formal_significance_tests.csv` 包含 `raw_delta_mean`、统一为“正值表示
Dynamic DNA 更好”的 `oriented_delta_mean`、95% CI、精确 p 值、Holm p 值、胜/负/平次数和效应量 `cohens_dz`。

## 仅重新分析已有长表

若已经有符合上述 schema 的长表，可不重跑训练：

```powershell
python -m incremental.statistics `
  --input incremental_result/significance_trials/math1_random/per_seed_results.csv `
  --output-dir incremental_result/significance_trials/math1_random `
  --target "Ours (Dynamic DNA)" `
  --baselines "Ours-Ablated" "Ours (LoRA)" "Naive FT (NFT)" EWC "DER++" "C-LoRA" `
  --metrics Balanced_AUC
```

分析器会拒绝重复的 trial key、缺失的配对 seed、非数值指标；不会通过 inner join
静默丢弃试验。

## 额外基线

此运行器目前直接覆盖主实验的 6 个 G-NCDM 策略和 EWC/DER++/C-LoRA。ICDM-WWW24、X-DER、ICD 等独立脚本若需要加入检验，必须也用同一组训练 seed、同一 protocol 各运行一次，并以相同的长表 schema 追加结果；随后使用“仅重新分析”命令。不得把这些方法的单 seed 行与 10-seed 方法混合进配对检验。

若要针对测试学生的抽样不确定性进一步推断，应在另一个附录中保存逐题
`user_id, item_id, label, probability` 并执行 student-cluster bootstrap；它与这里的
seed-level 算法随机性检验是互补的，不能相互替代。
