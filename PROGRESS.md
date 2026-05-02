# Progress

Tracks the rolling cleanup / refactor PRs that prepare the repo for PathMNIST + MPS work.
Update the status / PR columns whenever a PR opens or merges.

## PR plan

PROGRESS PR # is local sequencing; GitHub PR # is the actual pull-request
number on Hold-0n-tight/safety-ai-v2 (the upstream `wognsths/safety-ai`
fork left codex PRs #1–#4 in the numbering).

| #   | Branch                       | Goal                                                                  | Status      | GH PR |
| --- | ---------------------------- | --------------------------------------------------------------------- | ----------- | ----- |
| 1   | `chore/cleanup-logs`         | Untrack root log files + tighten `.gitignore` (incl. pycache)         | merged      | #1    |
| 1.5 | `chore/untrack-data`         | Untrack `data/test/**.tif` and `data/split/*.json` (working copy kept)| merged      | #2    |
| 2   | `fix/test-path-and-history`  | Fix `loader.py` test-path mismatch + remove duplicate history save    | merged      | #3    |
| 3   | `feat/mps-support`           | Add MPS device branch (4 sites) + Ray `client_resources` from cfg     | in progress | -     |
| 4   | `feat/pathmnist-loader`      | Add `medmnist`, PathMNIST loader, `dataset_split.py` branch, config   | planned     | -     |
| 5   | `chore/smoke-test-config`    | Smoke-test FL config (rounds=2, tiny model, few clients)              | planned     | -     |

## Follow-ups (after PR 5)

- `git filter-repo` to shrink `.git` pack history (currently ~800 MB; the
  per-PR `git rm --cached` only removes blobs from HEAD, not from pack).
  Destructive history rewrite — schedule separately and coordinate with all
  collaborators before running.

## Notes / lessons learned

- **PR 1.5 footgun:** `git rm --cached` preserves the working tree at the
  time of removal, but a later `git pull` of that same merged commit applies
  the deletion to the working tree like any other diff. Local `data/` and
  `.DS_Store` were lost on the post-merge pull and recovered via
  `git restore --source=<pre-untrack-commit> --worktree`. For the next
  untrack-style PR (PR #4 may produce one for the new dataset cache dir),
  back up the affected paths *outside the repo* before pulling the merged
  develop, or move them to an external location and symlink.
