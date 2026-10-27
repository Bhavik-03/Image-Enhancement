---
name: git-workflow
description: Branch, commit and PR conventions for this repo. Apply whenever creating branches, commits or pull requests.
user-invocable: false
---

- Never commit or push on `main`. Branch per step: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`.
- Conventional Commits, imperative subject under 72 chars: `feat(data): add paired dataset loader`. The body says why.
- One logical change per commit. Stage files by name, never `git add -A` or `git add .`.
- Run `test-runner` and `reviewer` before committing; fix before the commit, never `--no-verify` or `--amend`.
- Open PRs with `gh pr create`: a short summary and the command you ran as evidence. Merge through the PR only.
- Never commit data, checkpoints (`*.pth`, `*.pt`), `.env` or `kaggle.json`.
- End commits with the `Co-Authored-By` line from the session attribution reminder.
