# -*- coding: utf-8 -*-
"""把 .xlsx 渲染成 Markdown 表格打到 stdout —— 让 xlsx 能直接在对话窗口里看。

用法：
    python tools/xlsx2md.py <file.xlsx>            # 打印全部 sheet
    python tools/xlsx2md.py <file.xlsx> <sheet>    # 只打某个 sheet
    python tools/xlsx2md.py <file.xlsx> --round 6  # 浮点保留位数(默认4)

依赖 pandas + openpyxl（环境已装）。浮点四舍五入、NaN 显示为 '-'。
"""

import sys
import pandas as pd

# Windows 上 Python stdout 默认 cp936/GBK，输出中文经 UTF-8 终端会乱码 → 强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _fmt(x, nd):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    # 含 '-'/NaN 的列会变 object，数字以字符串存 → 尝试按浮点 round
    if isinstance(x, str):
        try:
            return f"{float(x):.{nd}f}"
        except ValueError:
            return x
    return str(x)


def df_to_md(df, nd):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v, nd) for v in r.tolist()) + " |")
    return "\n".join(lines)


def main(argv):
    if not argv:
        print("用法: python tools/xlsx2md.py <file.xlsx> [sheet] [--round N]")
        return 1
    path = argv[0]
    nd = 4
    if "--round" in argv:
        nd = int(argv[argv.index("--round") + 1])
    sheet_arg = next((a for a in argv[1:] if not a.startswith("--") and a != str(nd)), None)

    xl = pd.ExcelFile(path)
    sheets = [sheet_arg] if sheet_arg else xl.sheet_names
    for s in sheets:
        df = xl.parse(s)
        print(f"\n### sheet: {s}  ({df.shape[0]}x{df.shape[1]})\n")
        print(df_to_md(df, nd))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
