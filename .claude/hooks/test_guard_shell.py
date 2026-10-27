#!/usr/bin/env python3
"""Sends example commands through guard_shell.py and checks each is blocked or allowed.

Run: python3 .claude/hooks/test_guard_shell.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

GUARD = Path(__file__).with_name("guard_shell.py")

# expect, branch, command
CASES = [
    ("block", "feat/data", "git push origin main"),
    ("block", "feat/data", "git push origin HEAD:main"),
    ("block", "feat/data", "git push origin feat/data:refs/heads/main"),
    ("block", "main", "git push"),
    ("block", "main", "git push -u origin"),
    ("block", "feat/data", "git push --force origin feat/data"),
    ("block", "feat/data", "git push -f origin feat/data"),
    ("block", "feat/data", "git push origin +feat/data"),
    ("block", "feat/data", "git status; git push origin main"),
    ("block", "main", "git commit -m 'feat: x'"),
    ("block", "feat/data", "git commit --amend"),
    ("block", "feat/data", "git commit --no-verify -m x"),
    ("block", "feat/data", "git add -A"),
    ("block", "feat/data", "git add ."),
    ("block", "feat/data", "git reset --hard HEAD~1"),
    ("block", "feat/data", "cat .env"),
    ("block", "feat/data", "cat kaggle.json"),
    ("allow", "feat/data", "git push -u origin feat/data"),
    ("allow", "feat/data", "git push origin fix-main-docs"),
    ("allow", "feat/data", "git push"),
    ("allow", "feat/data", "git add data.py model.py"),
    ("allow", "feat/data", "git commit -m 'docs: mention --no-verify and .env'"),
    ("allow", "feat/data", "cat .env.example"),
    ("allow", "feat/data", "gh pr create --base main --title 'feat: data'"),
    ("allow", "feat/data", "git switch main; git pull"),
]


def run(repo, command):
    event = json.dumps({"tool_input": {"command": command}})
    r = subprocess.run([sys.executable, str(GUARD)], input=event, text=True, cwd=repo, capture_output=True)
    return "block" if r.returncode == 2 else "allow"


def main():
    failed = 0
    with tempfile.TemporaryDirectory() as repo:  # throwaway repo, no tests/ dir, so commits skip pytest
        subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
        for expect, branch, command in CASES:
            subprocess.run(["git", "-C", repo, "symbolic-ref", "HEAD", f"refs/heads/{branch}"], check=True)
            got = run(repo, command)
            ok = got == expect
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {got:<5} on {branch:<10} {command}")
    print(f"{len(CASES) - failed} of {len(CASES)} passed")
    sys.exit(failed > 0)


if __name__ == "__main__":
    main()
