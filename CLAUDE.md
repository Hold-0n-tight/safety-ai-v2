# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Federated Learning research code for 9-class medical image classification. PyTorch + Flower (`flwr.simulation.start_simulation`, Ray backend). Three strategies are implemented: FedAvg, FedBN, FedProx. A `moon.yaml` config exists but **MOON is not implemented** in `train/strategies.py` — `get_strategy()` only branches on `fedavg|fedprox|fedbn` and raises `ValueError` for anything else.

The current `data/test/` already contains the 9 NCT-CRC-HE-100K classes (`ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`), which is the upstream dataset for MedMNIST PathMNIST.

## Common commands

The project has no test framework, no linter config, and no Makefile. Workflow is split-then-train.

```bash
# 1) Generate a client split (writes data/split/<name>.json + <name>_dist.png).
#    Must be done before any FL run. The split YAML's name (stem) becomes the JSON filename.
python scripts/dataset_split.py --split config/split/dirichlet_alpha5.yaml

# 2) Federated run. cfg.dataset.split_path must point at the JSON produced in step 1.
python run_federated.py --config config/fl/fedavg.yaml
python run_federated.py -c config/fl/fedbn.yaml -v --log-file run.log

# 3) Centralized baseline (single-process training).
python run_centralized.py --config config/centralized/custom9.yaml

# 4) Convenience: run fedavg then fedprox back-to-back.
bash run.sh
```

There is no `pip install -e .` — modules are imported via the `train.` package from the repo root, so always invoke Python from the project root.

## Architecture

### Configuration layering (OmegaConf YAML, **not** Hydra)

Configs are flat YAMLs loaded with `OmegaConf.load`. There is no composition / defaults-list. Three independent config trees:

- `config/split/*.yaml` — input to `scripts/dataset_split.py`. Defines `dataset.split.{type, alpha, num_clients, seed}`. Only `type: dirichlet` is supported.
- `config/fl/*.yaml` — input to `run_federated.py`. Required top-level sections (validated in `run_federated.py:108`): `model`, `train`, `fl`, `dataset`. `dataset.split_path` must already exist on disk.
- `config/centralized/*.yaml` — input to `run_centralized.py`. Different shape (`train.epochs` instead of `train.rounds`, plus `save.path`).

The split YAML and FL YAML are **decoupled**: an FL run does not regenerate splits, it just reads `cfg.dataset.split_path`. Mismatched `num_clients` between the split file and `cfg.fl.min_*_clients` will surface as missing-key errors deep in `client_fn`.

### FL execution flow (`train/federated.py`)

`run_federated_training(cfg)` →
1. Loads `data/split/<name>.json`, reads the `splits` dict (`{"client_0": [idx, ...], ...}`).
2. Defines `client_fn(context)` that pulls `partition-id` from `context.node_config`, builds per-client `DataLoader`s via `get_dataloaders_from_split`, instantiates a fresh model via `init_net`, and wraps in `FederatedClient`.
3. Builds a strategy via `get_strategy(cfg)` (FedAvg / FedProx / FedBN).
4. Calls `fl.simulation.start_simulation(...)` with `num_clients=len(client_splits)` and **hardcoded `client_resources={"num_gpus": 1, "num_cpus": 1}`** (`train/federated.py:230`). This will not work on CPU-only or MPS machines without modification — Ray will spin trying to allocate a GPU.
5. `run_federated.py` creates the timestamped run dir (`results/fl_<strategy>_<ts>/`) **before** training, injects it as `cfg.train.run_dir`, and writes `history.csv` there once training finishes. The strategy uses the same dir to drop per-round checkpoints into `<run_dir>/checkpoints/`.

### Strategy specifics (`train/strategies.py`)

All three strategies subclass `flwr.server.strategy.FedAvg` and inject `evaluate_fn=_create_centralized_evaluate_fn(cfg)`, which evaluates the aggregated global model on `data/test/` every round. The eval function infers test path as `Path(cfg.dataset.root).parent.parent / "test"` — so `cfg.dataset.root` **must** be two levels deep (e.g. `data/train/raw`), or centralized eval breaks silently with a warning.

- **FedAvg** — thin wrapper, just adds round logging.
- **FedProx** — overrides `configure_fit` to inject `mu` into the per-round client config dict. The client picks it up in `fit()` and adds the proximal term to its loss.
- **FedBN** — server-side aggregation is identical to FedAvg (delegates via `super().aggregate_fit`). The FedBN distinction is purely client-side: in `FederatedClient.set_parameters` (`train/federated.py`), when `strategy == "fedbn"`, BN running stats received from the server are *ignored* and the client keeps its own. The earlier implementation emitted zeros at BN-stat slots in `aggregate_fit`, which was correct for clients (they ignored the zeros) but catastrophic for the centralized `evaluate_fn` — `model.eval()` BatchNorm with `running_var=0` and ~50 BN layers in EfficientNet-B0 overflows fp32 → NaN loss starting at round 1. Source-of-truth predicate for BN keys is now `_is_bn` in `federated.py` only.

### Data loading (`train/loader.py`)

`get_dataloaders_from_split` builds the **full** ImageFolder per client and wraps it in `Subset(indices)` — every client process loads the same on-disk index, then narrows by indices.

Test set lookup logic differs between modules:
- `loader.py:124` — `data_root.parent / "test"` (so `data/train/raw` → `data/train/test`, which doesn't exist; falls through to `val`, then to first 10% of the client's own train split).
- `strategies.py:67` — `data_root.parent.parent / "test"` (so `data/train/raw` → `data/test`, which is correct).

The per-client `test_loader` therefore typically falls back to a tiny held-out slice of the client's own training data, while centralized server-side eval uses the real `data/test/`. Don't trust the client-reported `evaluate` accuracy — the metric of record comes from `evaluate_fn`.

`_infer_img_size` returns 380 only for `efficientnet_b4`, otherwise 224.

### Models (`train/models.py`)

`init_net(name, output_dim, pretrained=False, device=None)` factory dispatches to torchvision builders. Supported names: `resnet50, resnet34, efficientnet_b4, efficientnet_b0`. Add a model by extending `_MODEL_FACTORY`.

### Device handling — known limitation

Device selection is hardcoded as `cuda if available else cpu` in **four** places:
- `train/federated.py:53` (`DEFAULT_DEVICE`)
- `train/federated.py:73-80` (per-client, uses `ray.get_gpu_ids()`)
- `train/strategies.py:62` (centralized eval)
- `train/centralized.py:15`

There is no MPS branch yet. Adding Apple Silicon support requires editing all four sites *and* changing the Ray `client_resources` in `federated.py:230`, since requesting `num_gpus: 1` on a non-CUDA machine will hang scheduling.

### Results

`results/fl_<strategy>[_<ts>]/history.{csv,png}`. CSV schema is `round,loss,accuracy`. `loss` and `accuracy` come from `history.losses_centralized` and `history.metrics_centralized["accuracy"]` — i.e. server-side centralized evaluation, not client-aggregated metrics.

## Repo hygiene notes

- `.gitignore` excludes `*.log` and `data/`, but several multi-MB run logs are checked in as `*.txt` at the repo root (`full_log.txt`, `dirichlet*.txt`, `container_logs.txt`, `fl_training_log.txt`, `federated_avg_dir5.txt`). The `main` branch is ~788 MB largely because of these.
- `data/test/` is committed (real images), but `data/train/` is excluded by `.dockerignore` and gitignore — training data must be brought in out-of-band.
- `train/__init__.py` is empty (1-line stub).
- `Dockerfile` pins `torch==2.3.1+cu121` from the PyTorch CUDA index — not portable to Mac/MPS.
