# Progress

Tracks the rolling cleanup / refactor PRs that prepare the repo for PathMNIST + MPS work.
Update the status / PR columns whenever a PR opens or merges.

## PR plan

| #   | Branch                       | Goal                                                                  | Status      | PR  |
| --- | ---------------------------- | --------------------------------------------------------------------- | ----------- | --- |
| 1   | `chore/cleanup-logs`         | Untrack root log files + tighten `.gitignore` (incl. pycache)         | merged      | #1  |
| 1.5 | `chore/untrack-data`         | Untrack `data/test/**.tif` and `data/split/*.json` (working copy kept)| in progress | -   |
| 2   | `fix/test-path-and-history`  | Fix `loader.py` test-path mismatch + remove duplicate history save    | planned     | -   |
| 3   | `feat/mps-support`           | Add MPS device branch (4 sites) + Ray `client_resources` from cfg     | planned     | -   |
| 4   | `feat/pathmnist-loader`      | Add `medmnist`, PathMNIST loader, `dataset_split.py` branch, config   | planned     | -   |
| 5   | `chore/smoke-test-config`    | Smoke-test FL config (rounds=2, tiny model, few clients)              | planned     | -   |

## Follow-ups (after PR 5)

- `git filter-repo` to shrink `.git` pack history (currently ~800 MB; the
  per-PR `git rm --cached` only removes blobs from HEAD, not from pack).
  Destructive history rewrite — schedule separately and coordinate with all
  collaborators before running.
