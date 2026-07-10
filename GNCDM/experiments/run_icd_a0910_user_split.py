# -*- coding: utf-8 -*-
"""ICD · a0910 · user_split（Approach A，旧题流→新题流，support/query 评测）。

薄入口：复用 run_icd_a0910_A.py，仅把 SPLIT_TAG 设为 user_split。
a0910 题量大(17746)，务必 GPU + EduCDM 环境（见 GNCDM/docs/ICD_baseline_repro.md）。

ACC/F1 口径（user_split）：在 support 上用 Youden J 选阈值，再在 query 上算 ACC/F1。
这是冷启动合法校准（support 评测时可见），不是在 query 上事后调参；AUC/RMSE 与阈值无关。
random_split 仍固定阈值 0.5。写入主表时请脚注说明。

运行：cd GNCDM/experiments && python run_icd_a0910_user_split.py [DATA_DIR] [CTX] [STREAM_PER_STAGE]
产物：experiments/icd_out_a0910/icd_row_a0910_user_split.csv
跑完后会把该 CSV 再打印到终端。
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DEFAULT_DATA = os.path.join(_REPO, "data", "a0910")
_OUT_CSV = os.path.join(_HERE, "icd_out_a0910", "icd_row_a0910_user_split.csv")

extra = list(sys.argv[1:])
_argv = [os.path.join(_HERE, "run_icd_a0910_A.py")]
if extra and os.path.isfile(os.path.join(extra[0], "Q_matrix.npy")):
    _argv.append(os.path.abspath(extra.pop(0)))
else:
    _argv.append(_DEFAULT_DATA)
_argv.extend(extra)
if _argv[-1] not in ("random_split", "user_split"):
    _argv.append("user_split")
sys.argv = _argv
runpy.run_path(_argv[0], run_name="__main__")

# 跑完后把结果再打一遍到终端（方便扫日志尾部）
if os.path.isfile(_OUT_CSV):
    import pandas as pd

    df = pd.read_csv(_OUT_CSV)
    print("\n" + "=" * 72)
    print(" ICD a0910 user_split — final result")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nCSV: {_OUT_CSV}")
    print("=" * 72)
else:
    print(f"[WARN] 未找到结果文件: {_OUT_CSV}")
