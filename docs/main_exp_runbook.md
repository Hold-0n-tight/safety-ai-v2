# Main experiment runbook

Operational guide for running the 3 strategies × 4 alphas matrix
(`config/fl/main/`) on Colab T4 free tier. The infra needed for this
phase already merged in PR #6–#10 (per-round checkpoints, Drive
autosave, `--resume`, the 12 configs themselves). This runbook is
about how to actually drive it across Colab session limits without
losing work.

This is a *living* doc: update it as we learn what works.

## Operating constraints

- **Single-run cost:** ~25 min on T4 free with `efficientnet_b0` at
  `size=224`, 50 rounds, 5 clients, `local_epochs=5`, `batch_size=64`.
- **Matrix total:** ~5 h for 12 runs.
- **Free Colab T4 reality:**
  - ~3 h ceiling on a single active session before idle / runtime
    disconnect (sometimes shorter, occasionally longer).
  - Daily cumulative compute cap (~12 h, varies). Easily fits the
    matrix in one calendar day, but only if you do not waste a
    session by sitting idle.
  - GPU is not guaranteed — sometimes you only get CPU. PR #4's
    device-aware Ray resources prevents the run from hanging on a
    CPU-only assignment, but a CPU run will be much slower than the
    25 min estimate. If you don't get a T4, *abort and reconnect*
    rather than letting a CPU run eat session time.

## Session strategy: one strategy per session

Recommended split: 4 alphas of one strategy per Colab session
(~100 min total). Three sessions clear the matrix:

```
session 1: fedavg   (a05 → a1 → a5 → a10)
session 2: fedbn    (a05 → a1 → a5 → a10)
session 3: fedprox  (a05 → a1 → a5 → a10)
```

**Why not all 12 in one notebook:**
- 5 h exceeds the free session ceiling — guaranteed mid-loop disconnect.
- Resume mid-loop is annoying (have to figure out which run was active
  and skip the completed ones).
- One strategy per session keeps each session well under the ceiling.

**Why not 12 separate sessions:**
- Per-session overhead (mount Drive, clone repo, install deps,
  download MedMNIST cache) is ~3 min × 12 = 36 min wasted.
- Hybrid (4 runs / session) is the sweet spot.

## Drive folder layout

Set `DRIVE_DIR` once for the whole matrix; let timestamped per-run
basenames disambiguate:

```
/content/drive/MyDrive/safety-ai-runs/
└── main_exp/
    ├── fl_fedavg_20260505-1430/   ← timestamped per run
    │   ├── checkpoints/           (round_001.pt … round_050.pt + latest.pt)
    │   ├── history.csv
    │   ├── history.png
    │   └── run.log                ← see "Always pass --log-file" below
    ├── fl_fedavg_20260505-1500/
    ├── ...
    └── fl_fedprox_20260505-2200/
```

Strategy is in the basename. Alpha is *not* — pull it from `run.log`
(the `스플릿: data/split/pathmnist_dirichlet_alphaX.json` line) or
from the cfg file the run used. The post-hoc analysis snippet below
extracts it for you.

## Session boilerplate (first cell, every session)

```python
# 1) Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2) Drive sync target — fixed for the whole matrix
import os
os.environ['DRIVE_DIR'] = '/content/drive/MyDrive/safety-ai-runs/main_exp'

# 3) Code + deps
%cd /content
!git clone https://github.com/Hold-0n-tight/safety-ai-v2.git || true
%cd safety-ai-v2
!git pull --ff-only
!pip install -q -r requirements.txt

# 4) Splits (idempotent — dataset_split.py is fast if data is cached;
#    safe to re-run each session, splits are tiny)
!for ALPHA in pathmnist_dirichlet_alpha_05 \
              pathmnist_dirichlet_alpha1 \
              pathmnist_dirichlet_alpha5 \
              pathmnist_dirichlet_alpha10; do \
    python scripts/dataset_split.py --split config/split/${ALPHA}.yaml; \
done

# 5) Sanity-check we actually got a GPU
import torch
assert torch.cuda.is_available(), "No CUDA — abort and reconnect for a T4"
print(torch.cuda.get_device_name(0))
```

## Always pass `--log-file`

`run_federated.py` accepts `--log-file <path>`. Use it. The log line
that records `cfg.dataset.split_path` is currently the only post-hoc
way to recover which alpha a finished run used.

```python
import subprocess

STRATEGY = 'fedavg'   # change per session
for A in ['05', '1', '5', '10']:
    cfg = f'config/fl/main/{STRATEGY}_a{A}.yaml'
    log = f'/tmp/{STRATEGY}_a{A}.log'
    !python run_federated.py --config {cfg} --log-file {log}
    # The run dir is named results/fl_<strategy>_<ts>/. Copy the log
    # in so the end-of-run drive mirror picks it up automatically.
    latest = !ls -td results/fl_{STRATEGY}_*/ | head -1
    !cp {log} {latest[0]}/run.log
```

The end-of-run `mirror_tree` call in `run_federated.py` (PR #8) will
include `run.log` in the Drive copy because it lives inside the run dir
by the time training finishes.

## When a run breaks: resume

`--resume <run_dir>` accepts both local paths and Drive paths and
auto-mirrors the Drive tree back to local first (PR #9).

```python
# Common case: session died mid-fedavg_a5; new session, same Drive.
!python run_federated.py \
    --config config/fl/main/fedavg_a5.yaml \
    --resume /content/drive/MyDrive/safety-ai-runs/main_exp/fl_fedavg_20260505-1530
```

Resume is round-aligned: it picks up at the round *after* the last
saved checkpoint. If `latest.meta.json` says round 37, the new
simulation does rounds 38–50 and saves them as `round_038.pt`…
`round_050.pt` in the same dir. The merged `history.csv` covers 1–50
in one continuous file.

If the previous run actually completed (`round_offset == cfg.train.rounds`)
the resume call exits cleanly with a "이미 완료된 run" log — safe to
re-run by mistake.

## Skipping already-finished cells

If you reconnect mid-session and don't remember which alphas finished,
check Drive:

```python
from pathlib import Path
import json

ROOT = Path('/content/drive/MyDrive/safety-ai-runs/main_exp')
for run_dir in sorted(ROOT.glob('fl_*')):
    meta = run_dir / 'checkpoints' / 'latest.meta.json'
    if not meta.exists():
        continue
    m = json.loads(meta.read_text())
    print(f"{run_dir.name}: strategy={m['strategy']}, last_round={m['round']}")
```

A `last_round == 50` means the cell is done; skip it. Anything less is
either still running or was killed mid-run — resume it with the matching
cfg.

## Post-matrix analysis

Once all 12 cells reach round 50, this snippet builds a single
DataFrame keyed by `(strategy, alpha, round)`:

```python
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path('/content/drive/MyDrive/safety-ai-runs/main_exp')

def alpha_from_log(log_path: Path):
    """Pull alpha out of run.log via the 스플릿 line."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'pathmnist_dirichlet_alpha(_05|10|5|1)\b', text)
    if not m:
        return None
    raw = m.group(1)
    return {'_05': 0.5, '1': 1.0, '5': 5.0, '10': 10.0}[raw]

rows = []
for run_dir in sorted(ROOT.glob('fl_*')):
    hist = run_dir / 'history.csv'
    meta = run_dir / 'checkpoints' / 'latest.meta.json'
    if not (hist.exists() and meta.exists()):
        continue
    m = json.loads(meta.read_text())
    df = pd.read_csv(hist)
    df['strategy'] = m['strategy']
    df['alpha'] = alpha_from_log(run_dir / 'run.log')
    df['run_dir'] = run_dir.name
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
print(all_df.groupby(['strategy', 'alpha'])['accuracy'].max().unstack())
```

Plotting (3 lines per alpha, one per strategy):

```python
import matplotlib.pyplot as plt

for alpha in sorted(all_df['alpha'].dropna().unique()):
    fig, ax = plt.subplots(figsize=(6, 4))
    for strat in ['fedavg', 'fedbn', 'fedprox']:
        sub = all_df[(all_df.alpha == alpha) & (all_df.strategy == strat)]
        if not sub.empty:
            ax.plot(sub['round'], sub['accuracy'], label=strat)
    ax.set_title(f'PathMNIST · alpha={alpha}')
    ax.set_xlabel('round'); ax.set_ylabel('centralized accuracy')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
```

If you find yourself running this often, promote it to a small
analysis script — that's the right scope for a separate PR (e.g.
`scripts/aggregate_main_exp.py`), not this runbook.

## What goes in PRs vs. lives here

- **Bug fixes found while running the matrix** → small fix PRs against
  `develop`. After merge, decide case-by-case whether to re-run
  affected cells or accept the existing result.
- **New analysis / plotting code** → a small `scripts/` PR.
- **Operational tweaks (different DRIVE_DIR layout, batch size, etc.)**
  → update this runbook directly on `develop`. No PR.
- **Anything that changes training behavior** (loss, optimizer, eval) →
  PR. Don't smuggle behavior changes in as runbook edits.
