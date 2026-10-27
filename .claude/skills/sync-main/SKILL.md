---
name: sync-main
description: "After a PR is merged: switch to main, pull, and delete the merged branch locally and on GitHub."
disable-model-invocation: true
argument-hint: "[branch]"
allowed-tools: Bash(git status *) Bash(git switch main) Bash(git pull origin main) Bash(git branch --show-current) Bash(git branch --merged *) Bash(git branch -d *) Bash(git branch) Bash(git fetch --prune) Bash(git ls-remote --heads origin *) Bash(git push origin --delete *) Bash(gh pr view *) Bash(gh pr list *)
---
Clean up after a merged PR, following the `git-workflow` skill. Branch: $ARGUMENTS (default: the current branch). Stop at the first failed check and report why.

1. `git status --short` must be empty, else stop. If the branch is `main` and no argument was given, list `git branch --merged main` (without `main`) and ask me which to delete.
2. Check the PR is merged: `gh pr view <branch> --json number,state,mergedAt`. If the state is not `MERGED`, stop. Never delete a branch whose PR is open or closed unmerged.
3. `git switch main`, then `git pull origin main`.
4. Delete the local branch with `git branch -d <branch>`. If it refuses, stop and tell me; never use `-D`.
5. Delete the remote branch with `git push origin --delete <branch>`, unless `git ls-remote --heads origin <branch>` shows GitHub already removed it. Then `git fetch --prune`.
6. Report in at most 5 lines: PR number, `main` at `git log --oneline -1`, branches deleted (local / remote), remaining local branches, and `gh pr list` (open PRs).

Never touch `main` or any branch other than the merged one, and never push anything else.
