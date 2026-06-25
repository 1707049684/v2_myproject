# -*- coding: utf-8 -*-
"""把 .xlsx 渲染成 Markdown 表格 —— 让 xlsx 能直接在对话窗口 / IDE 里看，不乱码。

用法：
    python tools/xlsx2md.py <file.xlsx>            # 打印全部 sheet 到 stdout
    python tools/xlsx2md.py <file.xlsx> <sheet>    # 只打某个 sheet
    python tools/xlsx2md.py <file.xlsx> --round 6  # 浮点保留位数(默认4)
    python tools/xlsx2md.py <file.xlsx> --out x.md # 写 UTF-8 的 .md 文件（IDE 里打开它最稳）
    python tools/xlsx2md.py <file.xlsx> --out      # 不带路径=写到 <同名>.md

依赖 pandas + openpyxl（环境已装）。浮点四舍五入、NaN 显示为 '-'。
★ 关于乱码：stdout 已强制 UTF-8（适配 UTF-8 终端）。若你 IDE 的运行控制台是 GBK/cp936，
  stdout 仍可能花——这时用 --out 写成 UTF-8 的 .md，在 IDE 里打开该文件即可，绕开控制台编码。
"""

import os
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
        print("用法: python tools/xlsx2md.py <file.xlsx> [sheet] [--round N] [--out [x.md]]")
        return 1
    path = argv[0]

    nd = 4
    if "--round" in argv:
        nd = int(argv[argv.index("--round") + 1])

    # --out [可选路径]：缺省写到 <同名>.md
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt and not nxt.startswith("--"):
            out_path = nxt
        else:
            out_path = os.path.splitext(path)[0] + ".md"

    consumed = {str(nd), out_path} if out_path else {str(nd)}
    sheet_arg = next(
        (a for a in argv[1:] if not a.startswith("--") and a not in consumed), None
    )

    xl = pd.ExcelFile(path)
    sheets = [sheet_arg] if sheet_arg else xl.sheet_names
    blocks = []
    for s in sheets:
        df = xl.parse(s)
        blocks.append(f"### sheet: {s}  ({df.shape[0]}x{df.shape[1]})\n\n" + df_to_md(df, nd))
    text = "\n\n".join(blocks)

    if out_path:
        # utf-8-sig 带 BOM，中文 Windows 的 IDE/记事本/Excel 更可靠识别为 UTF-8
        with open(out_path, "w", encoding="utf-8-sig") as f:
            f.write(text + "\n")
        print(f">>> 已写 UTF-8 markdown: {out_path}（在 IDE 里打开此文件即可，不受控制台编码影响）")
    else:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
