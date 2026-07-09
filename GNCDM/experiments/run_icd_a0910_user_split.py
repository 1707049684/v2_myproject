# -*- coding: utf-8 -*-
"""ICD · a0910 · user_split（Approach A，旧题流→新题流，test 上分 old/new 评测）。

薄入口：复用 run_icd_a0910_A.py，仅把 SPLIT_TAG 设为 user_split。
a0910 题量大(17746)，务必 GPU + EduCDM 环境（见 GNCDM/docs/ICD_baseline_repro.md）。

运行：cd GNCDM/experiments && python run_icd_a0910_user_split.py [DATA_DIR] [CTX] [STREAM_PER_STAGE]
产物：experiments/icd_out_a0910/icd_row_a0910_user_split.csv
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_argv = [os.path.join(_HERE, "run_icd_a0910_A.py"), *sys.argv[1:], "user_split"]
sys.argv = _argv
runpy.run_path(_argv[0], run_name="__main__")
