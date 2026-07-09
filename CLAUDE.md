---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本仓库实现 **生成式认知诊断模型（Generative Cognitive Diagnosis）**：`GNCDM/`（生成式神经认知诊断 + 增量学习，主要工作）与 `GIRT/`（生成式 IRT 变体）。论文 arXiv:2507.09831。

## 环境

```bash
pip install -r GNCDM/requirements.txt   # torch==2.8.0, numpy, pandas, scikit-learn, tqdm, matplotlib
```
Python 3.10+。仓库里同时存在 cpython-310 和 cpython-313 的 pycache，本机为 3.13。

## 运行实验

**增量学习主实验**（核心贡献，对比 Dynamic DNA / LoRA / Full-Replay / Naive 等 6 种策略 + 3 CL 基线）。`experiments/` 下**只放 6 个 per-split 主入口**（3 数据集 × 2 划分），每个只跑一个划分、互不影响；在 `experiments/` 目录下运行（需 avalanche 给 EWC/DER）：
```bash
cd GNCDM/experiments
python run_incremental_math1_random_split.py   # Math1 严格拓扑二分(13旧/7新, ΔK={0,1,3,6}), alpha=0.20
python run_incremental_math1_user_split.py     # alpha=0.70
python run_incremental_a0910_random_split.py   # ASSIST a0910(17746题, GPU), alpha=0.1
python run_incremental_a0910_user_split.py     # alpha=0.6
python run_incremental_junyi_random_split.py   # junyi 稠密版(1000×712×39), alpha=0.1
python run_incremental_junyi_user_split.py     # alpha=0.6（未扫，占位）
```
**核心库在 `experiments/_core/`**（非入口，被上面 6 个入口 import）：`run_incremental_math1.py`（管线核心：`run_experiment`/`strict_bipartition`/`build_log_mat`/`IDCDataset` 等）、`run_incremental_a0910.py`（`auto_new_concepts` + a0910 维度）、`eval_all_methods_user_split.py`（user_split 的 `run_one` support/query 口径）；alpha 扫描脚本 `sweep_{base_alpha_random,junyi_random_alpha,a0910_random_alpha}.py` 也在 `_core/`。入口脚本把 `HERE/_core` 加入 `sys.path` 后裸 import 这些库；`_core` 文件用 `__file__` 往上数 3 层（库）/4 层（sweep）定位 `gncdm_dir`/`repo_root`，**勿删 `_core/` 或改其目录深度**。random 入口产 `all_methods_{ds}_random_split.{csv,md}`（Ours 经 `incremental_results_{ds}_random_split.csv` 中间表 + `cl_baselines_random_split.py` 合并基线），user 入口经 `eval_all_methods_user_split.run_one` 产 `all_methods_{ds}_user_split.{csv,md}`。按划分分派评测口径：random_split 走 `forward_using_buf` 无泄漏预测（论文 RQ2），user_split 走 support/query 冷启动 `forward` 重构（论文 RQ1，test/valid 用户互斥）。**严禁给 forward 喂 `torch.zeros` 作答**（生成式诊断需真实作答向量，否则 θ/ψ 退化为常数）。

**alpha（各划分各自单独取最优，互不相同）**：
- math1 `random_split=0.20`（0.05 步长扫 Base test ACC_old，0.20 见顶 0.7293，脚本 `experiments/_core/sweep_base_alpha_random.py`）
- math1 `user_split=0.70`（findings 第十八轮）
- a0910 `random_split=0.1`（`experiments/_core/sweep_a0910_random_alpha.py` 全扫 0.1~0.95、按 DNA mean(valid AUC) 选，0.1 见顶 0.7579；原对齐论文的 0.9 从未真扫、已被实扫超越，`run_incremental_a0910_random_split.py` 的 `ALPHA`）
- a0910 `user_split=0.6`（validation 全扫 0.1~0.95 按 valid_ACC 选定，0.6 见顶；优于默认 0.9，`eval_all_methods_user_split.py` 已硬编码）
- junyi `random_split=0.1`（稠密版 1000×712×39，`experiments/_core/sweep_junyi_random_alpha.py` 同口径选，0.1 见顶 0.8109，`run_incremental_junyi_random_split.py` 的 `ALPHA`）
- junyi `user_split=0.6`（暂未扫，沿用初值）
- xes(XES3G5M) `random_split=0.20`（`experiments/_core/sweep_xes_random_alpha.py` 实扫 0.1~0.7、按 DNA mean(valid AUC) 选，0.20 见顶 0.7605，0.1~0.4 为平台；新概念占比 82% 却未落 0.1，是经验规律的一处偏离，`run_incremental_xes_random_split.py` 的 `ALPHA`）

**经验规律**：random split 新概念占比越大、最优 alpha 越小（math1 36%→0.20；junyi 62%、a0910 67%→均 0.1，DNA mean(valid AUC) 标准在 0.1 见顶）。

改 math1 的 alpha 在 `run_incremental_math1.py` 的 `main()` 里 splits 元组；a0910/junyi 各 random 在对应 `run_incremental_*_random_split.py` 的 `ALPHA`、a0910 user 在 `eval_all_methods_user_split.py` 的 `configs`。

**口径易混点**：增量实验的 `Base` **只在旧题子集上训练+评测**（math1 是 13/20 题、7/11 概念），因此它的 ACC（math1 random≈0.72）**不能**直接对标论文「完整模型」数字（完整 20 题 G-NCDM 重构 user-split≈0.74~0.79，论文 0.749）。差距来自「只用旧子集」，非退化。`Ours(DNA/LoRA)` 旧任务恒等于 `Base`、TMD=0（架构隔离零遗忘）是预期结果，不是 bug。

**标准训练**（`core/run.py`）——`run.py` 用裸导入（`import model`、`from model_parser import parse_args`），且 `--save_path ./result/...`、`--training_config config/...` 都是相对路径。**必须从 `GNCDM/` 根目录用 `python core/run.py` 启动**：这样脚本目录 `core/` 自动进入 sys.path（满足裸导入），而 cwd 仍是 `GNCDM/`（满足相对路径）。参考 `GNCDM/scripts/*.sh` 取各数据集的超参。Math1 示例：
```bash
cd GNCDM
python core/run.py \
  --train_file data/math1_train_0.8_0.2.csv \
  --valid_file data/math1_valid_0.8_0.2.csv \
  --test_file  data/math1_test_0.8_0.2.csv \
  --Q_matrix   data/math1_Q_matrix.npy \
  --save_path  ./result/math1 \
  --n_user 4209 --n_item 20 --n_know 11 \
  --user_dim 32 --item_dim 32 --alpha 0.8 \
  --training_config config/training_config_math1.json
```

## 测试 / 代码风格

```bash
python -m pytest tests/ -q        # 最小冒烟测试（构造 + 前向 + IDCDataset）
ruff format .                     # 格式化（配置见 pyproject.toml，行宽 100）
ruff check .                      # 静态检查
```
`tests/conftest.py` 把 `GNCDM/` 加入 sys.path，所以测试用 `from core.model import GNCDM` 导入。`.py` 编辑后会被 PostToolUse hook 自动 `ruff format`。

## 设备约定

默认在 **GPU** 上训练：config（`config/training_config_*.json`）里写 `"device": "cuda:2"`，`scripts/*.sh` 用 `cuda:0`~`cuda:3`。无 GPU 时改成 `"cpu"`（部分实验脚本已硬编码 `torch.device('cpu')`，按需调整）。

## 数据与约定

- 数据集 CSV 列为 `user_id, item_id, score`（score 为 0/1）；Q 矩阵是 `(n_item, n_know)` 的 `.npy`。
- Math1：4209 users × 20 items × 11 concepts；文件名中 `0.8_0.2` 指 train/test 划分。
- 实验固定 `set_seed(42)`（含 `torch.cuda.manual_seed_all`）以保证可复现。
- 结果目录：`result/`（标准训练）、`incremental_result/`、`math1_result/`。
- **画图脚本**：统一放 `GNCDM/plot/`（见该目录 `README.md`），产物仍写入 `incremental_result/`；`experiments/` 只跑实验、不放 `plot_*.py`。

## 关键代码

- `core/model.py`：`GNCDM(nn.Module)` 编码器-解码器架构；增量方法 `expand_topology`（侧分支冻结旧参数）、`expand_topology_lora`（低秩适配）、`full_replay_oracle_expand_topology`（上界）。新分支用 **微方差初始化**（`* 1e-3`），不要随意改成标准初始化。
- `core/train.py`：训练循环、`IDCDataset`、评估（AUC/RMSE/ACC/F1）。
- `incremental/loss.py`、`incremental/metrics.py`：`TopologyAwareDecoupledLoss`、TMD（Trait Manifold Drift，衡量旧知识漂移）。

## Git

远程有两个：`origin`（v1_myproject，旧版）与 `v2`（v2_myproject）。**本项目默认推送到 `v2`**（`git push v2 <branch>`），不要推到 `origin`，除非用户明确要求。
