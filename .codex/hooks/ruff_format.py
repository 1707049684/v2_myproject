"""PostToolUse hook: run `ruff format` on an edited .py file.

Reads the hook JSON payload from stdin, extracts the edited file path, and
formats it with ruff when it is a Python file. Non-fatal by design.
"""
import json
import shutil
import subprocess
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_input = payload.get("tool_input") or {}
tool_response = payload.get("tool_response") or {}
path = tool_input.get("file_path") or tool_response.get("filePath") or ""

if path.endswith(".py") and shutil.which("ruff"):
    subprocess.run(["ruff", "format", path], check=False)
