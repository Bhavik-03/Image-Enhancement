---
name: ship-step
description: "Ship a finished step: final checks, show commits, push the feature branch and open the PR."
disable-model-invocation: true
argument-hint: "[feat/<topic>]"
allowed-tools: Bash(git status *) Bash(git log *) Bash(git diff *) Bash(git push -u origin feat/*) Bash(git push -u origin docs/*) Bash(gh pr create *) Bash(gh pr view *)
---
Ship $ARGUMENTS (default: the current branch), following the `git-workflow` skill. Stop at the first failed check and report why.

1. Check: on a `feat/*`, `fix/*`, `chore/*` or `docs/*` branch, not `main`; `git status --short` is empty.
2. Use the test-runner subagent. Continue only if tests and lint pass.
3. Run `git log --oneline main..HEAD` and `git diff --stat main...HEAD`. Show both and ask me to confirm the push. Do nothing more until I say yes.
4. Push the branch with `git push -u origin <branch>`.
5. Open the PR with `gh pr create --base main`: a short summary of what the step delivers and the command run as evidence, per `git-workflow`.
6. Report the PR link and the commit list. Do not merge: I merge after reading the PR, then `git switch main` and `git pull`.
