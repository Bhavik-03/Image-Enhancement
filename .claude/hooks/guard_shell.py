#!/usr/bin/env python3
"""PreToolUse hook for Bash: exit 2 blocks the command and sends the reason back.

Each chained command (&&, ||, ;, |, newline) is checked on its own. Commit-message
text is removed before matching, so "--no-verify" or ".env" inside a message is fine.
The commit-time test run is capped at 90 s, under the 120 s hook timeout in settings.json.
"""
import json
import os
import re
import subprocess
import sys

TEST_TIMEOUT = 90
SECRET = re.compile(r"(^|[\s/\"'=])(\.env(\.(?!example\b)[\w.-]+)?|kaggle\.json|secrets/)(?=$|[\s\"'])")


def block(msg):
    print(f"BLOCKED by guard_shell: {msg}", file=sys.stderr)
    sys.exit(2)


def branch():
    r = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    return r.stdout.strip()


def tests_pass():
    if not os.path.isdir("tests"):
        return True
    py = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python3"
    try:
        r = subprocess.run([py, "-m", "pytest", "-q", "-x"], capture_output=True, text=True, timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        block(f"tests did not finish in {TEST_TIMEOUT}s; run them yourself before committing")
    return r.returncode in (0, 5)  # 5 = no tests collected


def strip_messages(cmd):
    cmd = re.sub(r"<<-?\s*'?(\w+)'?.*?\n.*?\n\s*\1\b", " ", cmd, flags=re.S)  # heredocs
    return re.sub(r"(-m|--message)(\s+|=)(\"[^\"]*\"|'[^']*')", r"\1 MSG", cmd)


def check_segment(seg, on_branch):
    if SECRET.search(seg):
        block("command touches a secrets file (.env, kaggle.json, secrets/)")
    if not re.match(r"\s*git\b", seg):
        return
    if "--no-verify" in seg:
        block("--no-verify is not allowed")
    if re.match(r"\s*git\s+add\s+(-A|--all|\.)(\s|$)", seg):
        block("stage files by name, not 'git add -A' or 'git add .'")
    if re.search(r"git\s+reset\s+.*--hard", seg):
        block("git reset --hard is not allowed")
    if re.search(r"git\s+clean\s+.*-\w*f", seg):
        block("git clean -f is not allowed")
    if re.search(r"git\s+checkout\s+--\s+\.", seg):
        block("git checkout -- . is not allowed")
    if re.search(r"git\s+branch\s+.*-D\b", seg):
        block("git branch -D is not allowed")
    if re.match(r"\s*git\s+commit\b", seg):
        if "--amend" in seg:
            block("git commit --amend is not allowed; make a new commit")
        if on_branch == "main":
            block("do not commit on main; create a feature branch first")
        if not tests_pass():
            block("tests are failing; fix them before committing")
    if re.match(r"\s*git\s+push\b", seg):
        if re.search(r"(--force\b|--force-with-lease|\s-f\b|\s\+\S)", seg):
            block("force push is not allowed")
        if re.search(r"--(all|mirror)\b", seg):
            block("--all/--mirror push is not allowed")
        if re.search(r"(:|\s)(refs/heads/)?main(\s|$)", seg):
            block("pushing main is not allowed; merge through a PR")
        args = [a for a in seg.split()[2:] if not a.startswith("-")]
        if len(args) < 2 and on_branch == "main":
            block("bare 'git push' on main is not allowed")


def main():
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    current = branch()
    for seg in re.split(r"&&|\|\||;|\||\n", strip_messages(cmd)):
        check_segment(seg, current)


if __name__ == "__main__":
    main()
