#!/usr/bin/env python3
"""Cursor `postToolUse` hook.

Mirrors planning-with-files' PostToolUse hook: after any file-writing tool
call, remind the agent to update progress.md (and task_plan.md if a phase
just completed). Only fires when task_plan.md exists at the workspace root
(i.e. planning is actually active for this project).

No `matcher` is set in hooks.json for this hook because Cursor's documented
matcher values for postToolUse ("Shell, Read, Write, Grep, Delete, Task,
MCP:<tool>") are described as non-exhaustive, and the exact internal name for
edit-style tools is not guaranteed — filtering by keyword here is more
robust than guessing a matcher string.

Must never crash and must always emit valid JSON on stdout.
"""

import json
import os
import sys

WRITE_LIKE_KEYWORDS = ("write", "edit", "replace", "notebook", "delete")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({}))
        return

    tool_name = (payload.get("tool_name") or "").lower()
    if not any(k in tool_name for k in WRITE_LIKE_KEYWORDS):
        print(json.dumps({}))
        return

    roots = payload.get("workspace_roots") or [os.getcwd()]
    root = roots[0]
    plan_path = os.path.join(root, "task_plan.md")

    if not os.path.isfile(plan_path):
        print(json.dumps({}))
        return

    msg = (
        "[planning-with-files] Update progress.md with what you just did. "
        "If a phase is now complete, update task_plan.md status."
    )
    print(json.dumps({"additional_context": msg}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({}))
