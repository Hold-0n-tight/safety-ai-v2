"""Mirror per-round checkpoints and final results to Google Drive.

Designed for Colab: when ``cfg.train.drive_dir`` (or ``$DRIVE_DIR``) points at
a mounted Drive path, every checkpoint is mirrored as it is written and the
full run dir is mirrored at end-of-run. All operations are best-effort — if
Drive is unavailable, full, or read-only, the run logs a warning and keeps
training. Drive failures must never crash the FL simulation.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

log = logging.getLogger("train.drive_sync")


def resolve_drive_dir(cfg: DictConfig) -> Optional[Path]:
    """Return the Drive root for this project, or ``None`` if not configured.

    Resolution order: ``cfg.train.drive_dir`` overrides the ``DRIVE_DIR`` env
    var. The path must already exist (the user is responsible for mounting
    Drive); we do not create it for them.
    """
    raw = cfg.train.get("drive_dir", None) if "train" in cfg else None
    if not raw:
        raw = os.environ.get("DRIVE_DIR")
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.exists():
        log.warning("DRIVE_DIR=%s does not exist; drive sync disabled", path)
        return None
    if not path.is_dir():
        log.warning("DRIVE_DIR=%s is not a directory; drive sync disabled", path)
        return None
    return path


def per_round_enabled(cfg: DictConfig) -> bool:
    """Whether to mirror checkpoints at the end of every round."""
    return bool(cfg.train.get("drive_sync_per_round", True))


def mirror_file(src: Path, drive_dst_dir: Path) -> None:
    """Copy ``src`` into ``drive_dst_dir`` (overwrite). Errors are logged."""
    try:
        drive_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, drive_dst_dir / src.name)
    except Exception as exc:  # noqa: BLE001 — drive failures must not crash FL
        log.warning("drive mirror failed for %s -> %s: %s", src, drive_dst_dir, exc)


def mirror_tree(src_dir: Path, drive_dst_dir: Path) -> None:
    """Mirror an entire run dir to Drive, overwriting existing files."""
    try:
        drive_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, drive_dst_dir, dirs_exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("drive tree mirror failed for %s -> %s: %s", src_dir, drive_dst_dir, exc)
