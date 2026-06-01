---
name: train-gncdm
description: 用 core/run.py 标准 CLI 训练 G-NCDM 模型，已预填 Math1 数据集参数。当用户要求训练 GNCDM、跑标准（非增量）训练、或在某数据集上重新训练时使用。
disable-model-invocation: true
---

用标准 CLI 训练 G-NCDM。

参数 `$ARGUMENTS`：数据集名，`math1`（默认）或 `a0910`；可附加 `--device cpu` 等覆盖项。

关键：`core/run.py` 用裸导入且相对路径，**必须从 `GNCDM/` 根目录用 `python core/run.py` 启动**（脚本目录 core/ 自动进入 sys.path，cwd 保持在 GNCDM/）。

Math1 命令模板：
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

对 `a0910` 等其他数据集：先看 `GNCDM/scripts/gncdm_<dataset>_*.sh` 取对应的 `n_user/n_item/n_know/alpha` 和 config 文件，再套用上面的模板。

设备：config 默认 `cuda:2`。本机无 GPU 时在 config json 里改 `"device": "cpu"`（不要直接改命令行，run.py 从 config 读 device）。

训练完成后到 `--save_path` 目录查看产物（`params_*.pt` 等），并汇报验证/测试指标。
