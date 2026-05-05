"""
strategies.py

Custom FL strategies for Flower.

* FedAvgStrategy  – Wrapper form. Use base FedAvg
* FedProxStrategy – Send client to FedAvg + μ(proximal)
* FedBNStrategy   – Exclude BatchNorm parameter in average (FedBN)
* get_strategy()  – Returns appropriate strategy base on .yaml configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from train.checkpoint import save_round_checkpoint
from train.device import pick_device
from train.models import init_net
from train.loader import _infer_img_size, get_test_dataset


# ──────────────────────────────────────────────────────────────
# Helper: BN 여부 판별
# ──────────────────────────────────────────────────────────────
def _is_bn_param(name: str) -> bool:
    return (
        ".running_mean" in name
        or ".running_var" in name
        or ".num_batches_tracked" in name
    )


# ──────────────────────────────────────────────────────────────
# Helper: per-round checkpointing (PR #7)
# ──────────────────────────────────────────────────────────────
def _init_checkpointing(strategy_obj, cfg: DictConfig, state_dict_keys) -> None:
    """Attach checkpoint config + model state_dict keys onto a strategy.

    ``cfg.train.run_dir`` is injected by ``run_federated.py`` before training
    starts; if it is missing (older callers) checkpointing silently no-ops.
    """
    train_cfg = cfg.train
    strategy_obj._save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
    strategy_obj._checkpoint_every = int(train_cfg.get("checkpoint_every", 1))
    strategy_obj._strategy_name = str(train_cfg.strategy).lower()
    strategy_obj._state_dict_keys = list(state_dict_keys)

    run_dir = train_cfg.get("run_dir", None)
    strategy_obj._checkpoint_dir = (
        Path(run_dir) / "checkpoints" if run_dir else None
    )


def _maybe_save_checkpoint(
    strategy_obj, server_round: int, parameters: Optional[fl.common.Parameters]
) -> None:
    if parameters is None:
        return
    if not getattr(strategy_obj, "_save_checkpoints", False):
        return
    if strategy_obj._checkpoint_dir is None:
        return
    if server_round % strategy_obj._checkpoint_every != 0:
        return
    save_round_checkpoint(
        parameters=parameters,
        round_num=server_round,
        out_dir=strategy_obj._checkpoint_dir,
        state_dict_keys=strategy_obj._state_dict_keys,
        strategy=strategy_obj._strategy_name,
    )


# ──────────────────────────────────────────────────────────────
# 중앙화된 평가 함수
# ──────────────────────────────────────────────────────────────
def _create_centralized_evaluate_fn(cfg: DictConfig):
    """data/test 데이터로 중앙화된 평가를 수행하는 함수를 생성합니다."""
    
    def evaluate_fn(server_round: int, parameters, config: Dict[str, fl.common.Scalar]):
        # 글로벌 모델 생성
        model = init_net(cfg.model.name, cfg.model.output_dim)
        
        # 파라미터 로드 (parameters가 이미 numpy array 리스트인 경우와 Parameters 객체인 경우 모두 처리)
        if hasattr(parameters, 'tensors'):
            # Parameters 객체인 경우
            param_arrays = fl.common.parameters_to_ndarrays(parameters)
        else:
            # 이미 numpy array 리스트인 경우
            param_arrays = parameters
            
        params_dict = zip(model.state_dict().keys(), param_arrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # 디바이스 설정
        device = pick_device()
        model.to(device)
        model.eval()
        
        # 테스트 데이터셋 (custom9 → data/test, pathmnist → built-in test split)
        img_size = _infer_img_size(cfg.dataset.name)
        test_dataset = get_test_dataset(
            cfg.dataset.name,
            cfg.dataset.root,
            img_size,
            size=int(cfg.dataset.get("size", 28)),
        )
        if test_dataset is None:
            print(f"Warning: No test dataset available for {cfg.dataset.name}")
            return None

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        
        # 평가 수행
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                
                total_loss += loss.item() * x.size(0)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == y).sum().item()
                total_samples += y.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        print(f"Round {server_round} - Centralized Test | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate_fn


# ──────────────────────────────────────────────────────────────
# 1. FedAvg (래퍼)
# ──────────────────────────────────────────────────────────────
class FedAvgStrategy(fl.server.strategy.FedAvg):
    """얇은 래퍼—Flower 기본 FedAvg와 동일하지만 cfg 인자를 통일."""

    def __init__(self, cfg: DictConfig):
        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
            evaluate_fn=_create_centralized_evaluate_fn(cfg),  # 중앙화된 평가 추가
        )
        ref_model = init_net(cfg.model.name, cfg.model.output_dim)
        _init_checkpointing(self, cfg, ref_model.state_dict().keys())

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        """라운드별 참여 클라이언트 로깅 추가"""
        config = super().configure_fit(server_round, parameters, client_manager)
        client_ids = [int(proxy.cid) for proxy, _ in config]
        print(f"\n🔄 Round {server_round} - 참여 클라이언트: {sorted(client_ids)} (총 {len(client_ids)}개)")
        return config

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        _maybe_save_checkpoint(self, server_round, aggregated_parameters)
        return aggregated_parameters, metrics


# ──────────────────────────────────────────────────────────────
# 2. FedProx
# ──────────────────────────────────────────────────────────────
class FedProxStrategy(fl.server.strategy.FedAvg):
    """FedAvg + proximal term(μ)을 클라이언트 config로 전달."""

    def __init__(self, cfg: DictConfig):
        self.mu: float = float(cfg.train.mu)
        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
            evaluate_fn=_create_centralized_evaluate_fn(cfg),  # 중앙화된 평가 추가
        )
        ref_model = init_net(cfg.model.name, cfg.model.output_dim)
        _init_checkpointing(self, cfg, ref_model.state_dict().keys())

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        _maybe_save_checkpoint(self, server_round, aggregated_parameters)
        return aggregated_parameters, metrics

    def configure_fit(  # noqa: D401
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # 기본 FedAvg 설정을 가져온 뒤 config에 μ 추가
        fit_config = super().configure_fit(server_round, parameters, client_manager)
        
        # 참여 클라이언트 로깅
        client_ids = [int(proxy.cid) for proxy, _ in fit_config]
        print(f"\n🔄 Round {server_round} (FedProx μ={self.mu}) - 참여 클라이언트: {sorted(client_ids)} (총 {len(client_ids)}개)")
        
        patched: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]] = []
        for client_proxy, fit_ins in fit_config:
            new_conf = dict(fit_ins.config)
            new_conf["mu"] = self.mu
            patched.append((client_proxy, fl.common.FitIns(fit_ins.parameters, new_conf)))
        return patched


# ──────────────────────────────────────────────────────────────
# 3. FedBN  (BN 파라미터 제외 평균)
# ──────────────────────────────────────────────────────────────
class FedBNStrategy(fl.server.strategy.FedAvg):
    """BatchNorm 파라미터를 평균에서 제외하는 FedBN 구현."""

    def __init__(self, cfg: DictConfig):
        # 모델 한 번 생성 → state_dict 키 순서 확보
        model = init_net(cfg.model.name, cfg.model.output_dim)
        self._parameter_names: List[str] = list(model.state_dict().keys())

        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
            evaluate_fn=_create_centralized_evaluate_fn(cfg),  # 중앙화된 평가 추가
        )
        _init_checkpointing(self, cfg, self._parameter_names)

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        """라운드별 참여 클라이언트 로깅 추가"""
        config = super().configure_fit(server_round, parameters, client_manager)
        client_ids = [int(proxy.cid) for proxy, _ in config]
        print(f"\n🔄 Round {server_round} (FedBN) - 참여 클라이언트: {sorted(client_ids)} (총 {len(client_ids)}개)")
        return config

    def aggregate_fit(  # noqa: D401
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Tuple[fl.common.Parameters | None, Dict[str, fl.common.Scalar]]:
        """FedBN aggregation: 비-BN 파라미터만 평균화, BN 파라미터는 제외"""
        if not results:
            return None, {}

        # 클라이언트별 파라미터 수집
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # 비-BN 파라미터만 평균화
        aggregated_ndarrays = []
        for i, param_name in enumerate(self._parameter_names):
            if _is_bn_param(param_name):
                # BN 파라미터: 서버에서 전혀 건드리지 않음 (클라이언트가 로컬 값 유지)
                # 더미 값으로 0으로 채운 배열 사용 (실제로는 클라이언트에서 무시됨)
                dummy_shape = weights_results[0][0][i].shape
                aggregated_ndarrays.append(np.zeros(dummy_shape, dtype=weights_results[0][0][i].dtype))
            else:
                # 비-BN 파라미터: 가중 평균
                total_examples = sum(num_examples for _, num_examples in weights_results)
                weighted_avg = np.zeros_like(weights_results[0][0][i])
                
                for weights, num_examples in weights_results:
                    weighted_avg += weights[i] * (num_examples / total_examples)
                
                aggregated_ndarrays.append(weighted_avg)

        # 메트릭 계산
        metrics_aggregated = {}
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            metrics_aggregated["total_examples"] = total_examples

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_ndarrays)
        # FedBN: BN slots in the saved checkpoint are zero (server-side dummies).
        # Each client keeps its own BN locally per FedBN's design — the resume
        # path (PR #9) re-initialises BN per client, which matches that contract.
        _maybe_save_checkpoint(self, rnd, aggregated_parameters)
        return aggregated_parameters, metrics_aggregated


# ──────────────────────────────────────────────────────────────
# 4. Strategy Factory
# ──────────────────────────────────────────────────────────────
def get_strategy(cfg: DictConfig) -> fl.server.strategy.Strategy:
    """cfg.train.strategy 문자열에 맞는 Strategy 인스턴스를 반환."""
    strat = cfg.train.strategy.lower()
    if strat == "fedavg":
        return FedAvgStrategy(cfg)
    if strat == "fedprox":
        return FedProxStrategy(cfg)
    if strat == "fedbn":
        return FedBNStrategy(cfg)
    raise ValueError(f"Unknown strategy '{cfg.train.strategy}'")
