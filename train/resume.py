"""Resume an interrupted FL run from a checkpoint dir.

Reads ``<resume_path>/checkpoints/latest.pt`` plus its sidecar meta, validates
the saved strategy matches the requested one, and prepares a Flower
``Parameters`` object so the next ``start_simulation`` call seeds the global
model from the previous round's aggregated weights.

If ``resume_path`` lives outside the project's ``results/`` tree (e.g. a Drive
path), the whole tree is mirrored back into ``results/<basename>/`` first so
that subsequent per-round checkpoints land alongside the old ones — keeping
the local layout, the Drive layout, and ``history.csv`` continuous.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
from omegaconf import DictConfig

log = logging.getLogger("train.resume")


class ResumeError(RuntimeError):
    """Raised when the resume request can't be satisfied (missing files,
    strategy mismatch, etc.). Caller should surface the message and abort."""


def _ndarrays_from_state_dict(state_dict) -> list:
    """Convert a torch state_dict into the ndarray list Flower expects.

    Order is preserved (state_dict is an OrderedDict), which matches how
    ``FederatedClient.get_parameters`` serialises weights.
    """
    return [t.detach().cpu().numpy() for t in state_dict.values()]


def load_resume_state(
    resume_path: Path, cfg: DictConfig
) -> Tuple[fl.common.Parameters, int]:
    """Validate, mirror-to-local-if-needed, and load the resume checkpoint.

    Side effects (on success): mutates ``cfg.train`` to set ``run_dir`` to the
    local mirror and ``round_offset`` to the last completed absolute round.

    Returns ``(initial_parameters, round_offset)`` for the caller to forward
    into ``run_federated_training``.
    """
    resume_path = Path(resume_path).expanduser().resolve()
    ckpt_dir = resume_path / "checkpoints"
    pt_path = ckpt_dir / "latest.pt"
    meta_path = ckpt_dir / "latest.meta.json"

    if not pt_path.exists() or not meta_path.exists():
        raise ResumeError(
            f"Resume path missing latest.pt or latest.meta.json under {ckpt_dir}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    saved_strategy = str(meta.get("strategy", "")).lower()
    cfg_strategy = str(cfg.train.strategy).lower()
    if saved_strategy != cfg_strategy:
        raise ResumeError(
            f"Strategy mismatch on resume: checkpoint was saved with "
            f"'{saved_strategy}' but cfg.train.strategy is '{cfg_strategy}'. "
            f"Resume requires the same strategy."
        )

    round_offset = int(meta["round"])

    # If the resume path is outside results/, mirror it back into results/
    # under the same basename so the local layout stays the canonical home
    # for new round_*.pt files.
    local_run_dir = (Path("results") / resume_path.name).resolve()
    if local_run_dir != resume_path:
        log.info("Mirroring resume tree %s -> %s", resume_path, local_run_dir)
        shutil.copytree(resume_path, local_run_dir, dirs_exist_ok=True)
    # Re-point pt_path at the local copy (now guaranteed present).
    pt_path = local_run_dir / "checkpoints" / "latest.pt"

    state_dict = torch.load(pt_path, weights_only=True, map_location="cpu")
    ndarrays = _ndarrays_from_state_dict(state_dict)
    initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)

    cfg.train.run_dir = str(local_run_dir)
    cfg.train.round_offset = round_offset
    log.info(
        "Resume loaded: strategy=%s, last completed round=%d, run_dir=%s",
        saved_strategy,
        round_offset,
        local_run_dir,
    )
    return initial_parameters, round_offset
