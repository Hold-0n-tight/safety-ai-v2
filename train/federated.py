"""PyTorch × Flower 연합학습: GPU + tqdm 지원 버전.

- 각 가상 클라이언트마다 `ray.get_gpu_ids()`로 GPU 지정
- Ray worker stdout ↔ tqdm 충돌 해결: worker 내 진행바 자동 비활성화
- FedBN, FedProx, 기본 FedAvg 모두 지원
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import flwr as fl
import ray  # GPU ID 확인용
import torch
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm as _tqdm  # Jupyter/CLI 모두 대응

# ──────────────────── tqdm → Ray‑safe wrapper ────────────────────
_IS_RAY_WORKER = "RAY_WORKER_ID" in os.environ  # Ray worker 여부 판별

def tqdm(*args, **kwargs):  # noqa: N802  (snake_case 유지)
    """Return a tqdm iterator that **disables itself** inside Ray workers.

    Ray가 `\r` 커서 이동 문자를 `\n`(개행)으로 바꿔 드라이버에 전달하는 바람에
    진행바가 *매 업데이트마다 한 줄씩* 쌓이는 문제를 방지한다.
    """
    kwargs["disable"] = _IS_RAY_WORKER or kwargs.get("disable", False)
    return _tqdm(*args, **kwargs)

# tqdm.write 그대로 쓰도록 별칭 유지
tqdm.write = _tqdm.write  # type: ignore[attr-defined]

# ────────────────────── 로깅 ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")

# ────────────────────── 사용자 모듈 ──────────────────────
# 경로: project_root/train/{loader,models,strategies}.py
from train.device import pick_device
from train.loader import get_dataloaders_from_split
from train.models import init_net
from train.strategies import get_strategy

# ────────────────────── 전역 장치 (백업) ──────────────────────
DEFAULT_DEVICE = pick_device()

# ────────────────────── Helper ──────────────────────

def _is_bn(name: str) -> bool:
    """BatchNorm 계층 파라미터 여부 (FedBN에서 사용)."""
    return any(k in name for k in (".running_mean", ".running_var", ".num_batches_tracked"))

# ────────────────────── Flower 클라이언트 ──────────────────────

class FederatedClient(NumPyClient):
    """Flower NumPyClient with GPU awareness & tqdm logging."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        cfg: DictConfig,
    ) -> None:
        gpu_ids: Sequence[int] = ray.get_gpu_ids()  # ex) [0]
        if gpu_ids:
            torch.cuda.set_device(int(gpu_ids[0]))
            self.device = torch.device("cuda")
            log.info(f"Client assigned GPU {gpu_ids[0]}")
        else:
            self.device = pick_device()  # mps on Apple Silicon, else cpu
            log.info(f"Client using {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self._local_bn_params: Dict[str, torch.Tensor] = {}

    # ---------------- Flower 필수 메서드 ----------------
    def get_parameters(self, config=None):  # noqa: D401  (Flower 시그니처)
        # state_dict()는 OrderedDict 보장 → key 순서 유지
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params: List, config=None):  # noqa: D401
        state = self.model.state_dict()
        is_fedbn = self.cfg.train.strategy.lower() == "fedbn"
        for name, tensor in zip(state.keys(), params):
            if is_fedbn and _is_bn(name):  # FedBN → BN 로컬 유지
                continue
            state[name] = torch.tensor(tensor, device=self.device)
        self.model.load_state_dict(state, strict=True)

        # FedBN: 로컬 BN 파라미터 캐싱 (향후 복원용)
        if is_fedbn:
            for n, p in state.items():
                if _is_bn(n):
                    self._local_bn_params[n] = p.clone()

    # ---------------- Fit ----------------
    def fit(self, params, config=None):  # noqa: D401
        self.set_parameters(params)
        self.model.train()

        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        criterion = torch.nn.CrossEntropyLoss()
        mu = float(config.get("mu", 0.0))  # FedProx μ (0이면 off)
        global_params = [p.detach().clone() for p in self.model.parameters()]

        epochs = self.cfg.train.local_epochs
        cid = config.get("partition_id", 0) if config else 0  # 클라이언트 ID

        # 시작 로그 출력
        log.info(f"🟢 Client {cid} starting {epochs} local epochs")

        total_loss = 0.0
        for ep in range(epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"C{cid}-E{ep + 1}/{epochs}",
                leave=False,
                dynamic_ncols=True,
                mininterval=15.0,  # 업데이트 빈도 더 줄임
                disable=_IS_RAY_WORKER,  # Ray worker에서 진행바 비활성화
            )
            running_loss = 0.0
            for step, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()

                # forward
                logits = self.model(x)
                loss = criterion(logits, y)

                # FedProx 규제항
                if mu > 0:
                    prox = sum(
                        (w - w0).pow(2).sum()
                        for w, w0 in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + 0.5 * mu * prox

                # backward + step
                loss.backward()
                optim.step()

                running_loss += loss.item()
                total_loss += loss.item()
                # 진행바 업데이트 (20 step마다)
                if step % 20 == 0:
                    pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}", refresh=False)

        # 완료 로그 출력
        avg_loss = total_loss / (epochs * len(self.train_loader))
        log.info(f"🔵 Client {cid} completed training - Average Loss: {avg_loss:.4f}")

        # 학습 완료 → 파라미터 반환
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    # ---------------- Evaluate ----------------
    def evaluate(self, params, config=None):  # noqa: D401
        self.set_parameters(params)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Eval", leave=False, disable=_IS_RAY_WORKER):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        return loss_sum / total, total, {"accuracy": correct / total}

# ────────────────────── 시뮬레이션 런처 ─────────────────────

def run_federated_training(cfg: DictConfig):
    """Hydra DictConfig → Flower Simulation 실행."""

    # 1) 클라이언트별 인덱스 불러오기
    split_path = Path(cfg.dataset.split_path)
    with split_path.open("r", encoding="utf-8") as f:
        client_splits = json.load(f)["splits"]

    data_root = Path(cfg.dataset.root)

    # 2) Flower 클라이언트 팩토리
    def client_fn(context: Context):
        # Flower 최신 버전: Context 객체 사용
        part_id = int(context.node_config["partition-id"])
        key = f"client_{part_id}"
        if key not in client_splits:
            raise KeyError(f"Split indices for {key} not found in {split_path}")

        log.info(f"Creating client {part_id} with {len(client_splits[key])} samples")

        # 데이터로더 생성 (cfg.dataset.size: pathmnist의 다운로드 해상도)
        train_loader, test_loader = get_dataloaders_from_split(
            client_id=part_id,
            split_indices=client_splits[key],
            data_root=data_root,
            batch_size=cfg.train.batch_size,
            dataset_name=cfg.dataset.name,
            size=int(cfg.dataset.get("size", 28)),
        )

        # 모델 초기화
        model = init_net(cfg.model.name, cfg.model.output_dim)
        log.info(f"Client {part_id} model initialized on {DEFAULT_DEVICE}")
        numpy_client = FederatedClient(model, train_loader, test_loader, cfg)
        return numpy_client.to_client()  # NumPyClient를 Client로 변환

    # 3) 전략 객체 생성 (FedAvg / FedProx / FedBN)
    strategy = get_strategy(cfg)

    # 4) Ray client_resources: cfg.fl.client_resources overrides; otherwise
    #    auto-detect — request a GPU only when CUDA is actually available, so
    #    Mac/CPU runs do not block on GPU scheduling.
    if "client_resources" in cfg.fl:
        client_resources = OmegaConf.to_container(cfg.fl.client_resources, resolve=True)
    else:
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 1 if torch.cuda.is_available() else 0,
        }
    log.info(f"Flower simulation starting … (client_resources={client_resources})")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_splits),
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.train.rounds),
        strategy=strategy,
    )
    log.info("Simulation finished")

    ray.shutdown()
    return history
