"""Patch ``config/fl/main/*.yaml`` for Colab-Free-tier execution.

Sets three fields, idempotently, on every (or one selected) main-experiment
FL config:

* ``dataset.size`` — the on-disk PathMNIST resolution. Pair with PR-A
  (``fix/loader-honor-cfg-img-size``) so the model also receives images at
  this resolution rather than upsampling to 224.
* ``fl.client_resources.{num_cpus, num_gpus}`` — Ray's per-client
  reservation. The default ``num_gpus=1`` serialises clients on a single
  T4. With Colab Free's 2 vCPU + 1 T4, the practical maximum is
  ``num_cpus=1, num_gpus=0.2`` → Ray fits ``min(floor(2/1), floor(1/0.2))
  = min(2, 5) = 2`` actors, so 5 clients run as 3 waves of 2/2/1
  rather than serially. Bumping ``num_cpus`` to 2 silently drops the
  pool back to a single actor on Colab Free — confirmed against
  observed behaviour.
* ``train.rounds`` — number of FL rounds. Default 25 (down from the
  earlier 50) reflects the fedavg_a1 convergence observation: peak
  centralized accuracy at round 15 (92.77 %), then plateau through
  round 22 (92.24 %). Cells that turn out to need more can extend
  with ``run_federated.py --resume <run_dir>`` (PR #9) without
  re-running rounds 1..25.

Re-runnable. The user kept the script under ``scripts/`` so future bulk
adjustments (e.g. dropping size further or changing batching) are a single
command rather than hand-edits across 12 files.

Usage
-----
Patch one config (validate first):
    python scripts/patch_main_configs.py --only fedavg_a1.yaml

Patch all 12 in one go (after the single-cell run looks good):
    python scripts/patch_main_configs.py

Custom values:
    python scripts/patch_main_configs.py --size 128 --num-gpus 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_DIR = Path("config/fl/main")
DEFAULT_SIZE = 64
DEFAULT_NUM_CPUS = 1
DEFAULT_NUM_GPUS = 0.2
DEFAULT_ROUNDS = 25


def patch_one(path: Path, size: int, num_cpus: int, num_gpus: float, rounds: int) -> None:
    cfg = OmegaConf.load(path)
    cfg.dataset.size = int(size)
    cfg.train.rounds = int(rounds)
    if "client_resources" not in cfg.fl:
        cfg.fl.client_resources = {}
    cfg.fl.client_resources.num_cpus = int(num_cpus)
    cfg.fl.client_resources.num_gpus = float(num_gpus)
    OmegaConf.save(cfg, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help="Directory of FL configs to patch")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help="dataset.size value (default: 64)")
    p.add_argument("--num-cpus", type=int, default=DEFAULT_NUM_CPUS,
                   help="fl.client_resources.num_cpus (default: 1, sized for Colab Free 2 vCPU)")
    p.add_argument("--num-gpus", type=float, default=DEFAULT_NUM_GPUS,
                   help="fl.client_resources.num_gpus (default: 0.2)")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="train.rounds (default: 25, based on fedavg_a1 plateau ~round 15)")
    p.add_argument("--only", type=str, default=None,
                   help="Patch only this filename inside --dir (e.g. fedavg_a1.yaml)")
    args = p.parse_args()

    files = sorted(args.dir.glob("*.yaml"))
    if args.only:
        files = [f for f in files if f.name == args.only]
        if not files:
            raise SystemExit(f"--only={args.only} not found under {args.dir}")

    for f in files:
        patch_one(f, args.size, args.num_cpus, args.num_gpus, args.rounds)
        print(f"[+] patched {f}")
    print(f"\nDone. {len(files)} file(s) patched "
          f"(size={args.size}, num_cpus={args.num_cpus}, "
          f"num_gpus={args.num_gpus}, rounds={args.rounds}).")


if __name__ == "__main__":
    main()
