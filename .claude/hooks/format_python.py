#!/usr/bin/env python3
"""PostToolUse hook for Edit|Write: ruff-format then ruff-lint the edited .py file.

Exit 2 returns lint errors so they get fixed at once. Files outside the project,
inside .venv, or not Python are skipped, as is everything if ruff is missing.
"""
import json
import os
import subprocess
import sys

root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
path = os.path.realpath(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
ruff = os.path.join(root, ".venv", "bin", "ruff")

if not path.endswith(".py") or not path.startswith(os.path.realpath(root) + os.sep):
    sys.exit(0)
if "/.venv/" in path or not os.path.exists(ruff) or not os.path.exists(path):
    sys.exit(0)

subprocess.run([ruff, "format", "-q", path])
r = subprocess.run([ruff, "check", path], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout + r.stderr, file=sys.stderr)
    sys.exit(2)
