#!/usr/bin/env python3
"""
Federated Learning 실행 스크립트

사용법:
    python run_federated.py --config config/fl/fedavg.yaml
    python run_federated.py --config config/fl/fedbn.yaml  
    python run_federated.py --config config/fl/fedprox.yaml

필수 전제조건:
    1. 데이터셋 스플릿이 미리 생성되어 있어야 함:
       python scripts/dataset_split.py --split config/split/dirichlet_alpha5.yaml
    
    2. 훈련 데이터가 data/train/raw에 준비되어 있어야 함
    3. 테스트 데이터가 data/test에 준비되어 있어야 함
"""

import argparse
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt
import csv

from omegaconf import OmegaConf
from train.drive_sync import mirror_tree, resolve_drive_dir
from train.federated import run_federated_training
from train.resume import ResumeError, load_resume_state
from datetime import datetime


def save_history(history, out_dir: Path, round_offset: int = 0) -> None:
    """Save FL history to CSV and PNG plot.

    On resume (``round_offset > 0``), Flower's history reports rounds 1..N for
    the resumed simulation; we offset them to absolute round numbers and merge
    with any pre-existing ``history.csv`` from the original run so the file
    stays a single contiguous record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds: list = []
    acc: list = []
    loss: list = []

    if getattr(history, "metrics_centralized", None):
        acc = [v for _, v in history.metrics_centralized.get("accuracy", [])]
    if getattr(history, "losses_centralized", None):
        loss = [v for _, v in history.losses_centralized]
        rounds = [r + round_offset for r, _ in history.losses_centralized]

    # Merge with existing rows (if any) up to the first new round, so resume
    # history.csv preserves the original 1..round_offset entries.
    csv_path = out_dir / "history.csv"
    existing_rows: list = []
    if round_offset > 0 and csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                try:
                    r = int(row[0])
                except ValueError:
                    continue
                if r <= round_offset:
                    existing_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])
        for row in existing_rows:
            writer.writerow(row)
        for i, r in enumerate(rounds):
            a = acc[i] if i < len(acc) else ""
            l = loss[i] if i < len(loss) else ""
            writer.writerow([r, l, a])

    # Plot the merged series so the PNG also reflects the full timeline.
    all_rounds = [int(row[0]) for row in existing_rows] + rounds
    all_loss = [float(row[1]) for row in existing_rows if row[1] != ""] + loss
    all_acc = [float(row[2]) for row in existing_rows if row[2] != ""] + acc
    if all_rounds:
        plt.figure()
        if all_loss:
            plt.plot(all_rounds[: len(all_loss)], all_loss, label="loss")
        if all_acc:
            plt.plot(all_rounds[: len(all_acc)], all_acc, label="accuracy")
        plt.xlabel("Round")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "history.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Federated Learning 훈련 실행")
    parser.add_argument(
        "--config", 
        "-c",
        type=Path,
        required=True,
        help="FL 설정 YAML 파일 경로 (예: config/fl/fedavg.yaml)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="로그를 저장할 파일 경로"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="기존 run dir에서 학습 재개 (예: results/fl_fedavg_20260505-1430 또는 Drive 경로)"
    )

    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger("run_federated")

    # 설정 파일 로드
    if not args.config.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    
    cfg = OmegaConf.load(args.config)
    
    # 필수 설정 검증
    required_keys = ["model", "train", "fl", "dataset"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"설정 파일에서 필수 섹션 '{key}'를 찾을 수 없습니다")
    
    # 데이터셋 스플릿 파일 존재 확인
    split_path = Path(cfg.dataset.split_path)
    if not split_path.exists():
        log.warning(f"데이터셋 스플릿 파일이 없습니다: {split_path}")
        return
    log.info(f"   스플릿: {split_path}")
    
    # 데이터 디렉터리 존재 확인  
    data_root = Path(cfg.dataset.root)
    if not data_root.exists():
        log.warning(f"훈련 데이터 디렉터리가 없습니다: {data_root}")
        log.warning("data/train/raw 디렉터리에 훈련 데이터를 준비하세요")
        return
    
    log.info("🚀 Federated Learning 시작")
    log.info(f"   전략: {cfg.train.strategy.upper()}")
    log.info(f"   모델: {cfg.model.name}")
    log.info(f"   라운드: {cfg.train.rounds}")
    log.info(f"   클라이언트: {cfg.fl.min_available_clients}")
    log.info(f"   스플릿: {split_path}")
    
    # FL 훈련 실행
    try:
        initial_parameters = None
        round_offset = 0
        if args.resume is not None:
            # Resume path: load latest.pt, validate strategy, mirror tree if
            # the resume dir is outside results/, and seed cfg.train.run_dir
            # / round_offset for the rest of this function.
            try:
                initial_parameters, round_offset = load_resume_state(args.resume, cfg)
            except ResumeError as e:
                log.error(f"❌ Resume 실패: {e}")
                return
            if round_offset >= int(cfg.train.rounds):
                log.info(
                    f"이미 완료된 run입니다 (round_offset={round_offset} >= "
                    f"cfg.train.rounds={cfg.train.rounds}). 학습 진입 안 함."
                )
                return
            log_dir = Path(cfg.train.run_dir)
            log.info(f"   재개 시작 라운드: {round_offset + 1} / {cfg.train.rounds}")
        else:
            # Fresh run: create the run dir up-front so the strategy can
            # write per-round checkpoints into <run_dir>/checkpoints/ during
            # training (PR #7).
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = Path("results") / f"fl_{cfg.train.strategy}_{ts}"
            log_dir.mkdir(parents=True, exist_ok=True)
            cfg.train.run_dir = str(log_dir)
        log.info(f"   결과 디렉토리: {log_dir}")

        history = run_federated_training(cfg, initial_parameters=initial_parameters)
        save_history(history, log_dir, round_offset=round_offset)

        # End-of-run Drive mirror: copies the entire run dir (history + any
        # straggler files not caught by per-round mirror). No-op if Drive is
        # not configured or not reachable.
        drive_root = resolve_drive_dir(cfg)
        if drive_root is not None:
            drive_run_dir = drive_root / log_dir.name
            mirror_tree(log_dir, drive_run_dir)
            log.info(f"   Drive 백업: {drive_run_dir}")

        log.info("✅ Federated Learning 완료!")
    except Exception as e:
        log.error(f"❌ 훈련 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 
