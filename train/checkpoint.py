"""Per-round checkpointing for federated runs.

Saves the aggregated global model after each FL round so that:
  * long Colab/Drive runs can survive disconnects (PR #8 ships them off-host),
  * a future ``--resume`` flag (PR #9) can pick up from the latest weights.

The on-disk layout is intentionally simple:

    <run_dir>/checkpoints/
        round_001.pt           # torch.save(state_dict)
        round_001.meta.json    # round, strategy, timestamp, num_tensors
        ...
        latest.pt              # copy of the most-recent round_*.pt
        latest.meta.json       # copy of the most-recent meta.json

``latest.pt`` is a copy (not a symlink) because Windows symlinks need either
admin or developer-mode, and the file is small enough that copying is cheap.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import flwr as fl
import torch

from train.drive_sync import mirror_file


def save_round_checkpoint(
    parameters: fl.common.Parameters,
    round_num: int,
    out_dir: Path,
    state_dict_keys: Iterable[str],
    strategy: str,
    drive_dir: Optional[Path] = None,
) -> Path:
    """Persist ``parameters`` as a ``state_dict`` plus sidecar metadata.

    When ``drive_dir`` is set, the four written files (round_NNN.pt /
    round_NNN.meta.json / latest.pt / latest.meta.json) are also mirrored
    there. Drive mirror failures are logged but never raised.

    Returns the path to the round's ``.pt`` file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ndarrays = fl.common.parameters_to_ndarrays(parameters)
    keys = list(state_dict_keys)
    if len(keys) != len(ndarrays):
        raise ValueError(
            f"state_dict_keys length {len(keys)} does not match "
            f"aggregated tensor count {len(ndarrays)}"
        )
    state_dict = {k: torch.from_numpy(arr.copy()) for k, arr in zip(keys, ndarrays)}

    pt_path = out_dir / f"round_{round_num:03d}.pt"
    meta_path = out_dir / f"round_{round_num:03d}.meta.json"
    torch.save(state_dict, pt_path)
    meta_path.write_text(
        json.dumps(
            {
                "round": round_num,
                "strategy": strategy,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "num_tensors": len(state_dict),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_pt = out_dir / "latest.pt"
    latest_meta = out_dir / "latest.meta.json"
    shutil.copyfile(pt_path, latest_pt)
    shutil.copyfile(meta_path, latest_meta)

    if drive_dir is not None:
        for f in (pt_path, meta_path, latest_pt, latest_meta):
            mirror_file(f, Path(drive_dir))

    return pt_path
