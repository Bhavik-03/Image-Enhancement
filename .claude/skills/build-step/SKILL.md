---
name: build-step
description: "Build one project step end to end: branch, code, checks, review, commit."
disable-model-invocation: true
argument-hint: "<step name, e.g. data | model-loss | train>"
---
Build step $ARGUMENTS from the plan (`~/.claude/plans/my-plan-is-to-zazzy-liskov.md`, section 5). Wait for each subagent's report before the next step. If a step can't be done, stop and report why.

1. Read the step in the plan and `CLAUDE.md`. If its files already exist, stop and ask.
2. Be on `main` with an empty `git status --short`, then `git switch -c feat/$ARGUMENTS`. Already on `feat/$ARGUMENTS`: continue.
3. Implement only the files the step names, with the least code that works. Follow `pytorch-conventions`. After the code, explain the concept it uses in a few lines (the user is new to this task).
4. Verify: use the test-runner subagent; fix and re-run, and stop after 3 failed rounds. For GPU or data work, also run the step's own check (e.g. one batch, `torch.cuda.is_available()`) and show the output. Real training or evaluation runs follow `training-runs`.
5. Use the reviewer subagent. Fix every BLOCKING item and repeat steps 4-5; stop after 2 BLOCKED rounds.
6. Commit following `git-workflow`: small atomic commits, files staged by name. Don't push.
7. Report (max 10 lines): files changed, check output, review verdict, the commit list, what to try next.

Never push or merge, and never touch files outside the step.
