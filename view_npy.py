# -*- coding: utf-8 -*-
"""查看 .npy 文件内容的脚本"""
import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python view_npy.py <npy文件路径>")
    sys.exit(1)

file_path = sys.argv[1]
data = np.load(file_path)

print(f"文件: {file_path}")
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"\n内容预览（前5行）:")
print(data[:5] if len(data.shape) == 2 else data[:10])
