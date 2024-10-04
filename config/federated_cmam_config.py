from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from modalities import Modality

from config.cmam_config import CMAMModelConfig
from config.config import (
    BaseConfig,
    Config,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    resolve_criterion,
    resolve_optimizer,
)
from config.federated_config import FederatedDataConfig, FederatedLoggingConfig


def federated_cmam_server_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedCMAMServerConfig(**value)


def federated_cmam_client_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedCMAMClientConfig(**value)


yaml.SafeLoader.add_constructor(
    "!FederatedCMAMServerConfig", federated_cmam_server_constructor
)
yaml.SafeLoader.add_constructor(
    "!FederatedCMAMClientConfig", federated_cmam_client_constructor
)


@dataclass(kw_only=True)
class FederatedCMAMClientConfig(BaseConfig):
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    criterion: str
    criterion_kwargs: Dict[str, Any]
    local_epochs: int
    local_batch_size: int
    model_config: ModelConfig
    cmam_config: CMAMModelConfig
    available_modalities: set[Modality | str]
    scheduler: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    target_metric: str = "loss"
    logging: LoggingConfig
    target_modality: Modality | str
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedCMAMServerConfig(BaseConfig):
    rounds: int
    num_clients: int
    aggregation_strategy: str
    global_model_config: ModelConfig
    cmam_model_configs: (
        list[CMAMModelConfig] | CMAMModelConfig
    )  ## The global model has all the modalities and therefore needs to train all the C-MAMs (in the incongruent setting anyways)
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    criterion: str
    criterion_kwargs: Dict[str, Any]
    epochs: int
    target_missing_type: str
    logging: LoggingConfig
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedCMAMConfig(Config):
    server_config: FederatedCMAMServerConfig
    client_config: FederatedCMAMClientConfig
    data_config: FederatedDataConfig
    logging: FederatedLoggingConfig  ## primary logging

    @classmethod
    def load(cls, path: str | Path | PathLike, run_id: int) -> "FederatedCMAMConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        experiment_config = data["experiment"]
        experiment_config.run_id = run_id

        data_config = data["data"]
        data_config = FederatedDataConfig.from_dict(data_config)

        if data_config.distribution_type == "non_iid":
            experiment_config.name += (
                f"_non_iid_{str(data_config.alpha).replace(".", "_")}"
            )

        primary_logging_config = data["logging"]
        primary_logging_config = FederatedLoggingConfig.from_dict(
            primary_logging_config,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        federated_config = data["federated"]
        server_config = federated_config["server_config"]

        assert all(
            [
                k in server_config.logging
                for k in ["log_path", "model_output_path", "metrics_path"]
            ]
        ), "Missing logging paths in server config"
        server_config.logging = LoggingConfig.from_dict(
            server_config.logging,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        client_config = federated_config["client_config"]

        client_config.logging = LoggingConfig.from_dict(
            client_config.logging,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        metrics_config = data["metrics"]

        metrics_config = MetricsConfig.from_dict(metrics_config)

        return cls(
            experiment=experiment_config,
            data_config=data_config,
            server_config=server_config,
            client_config=client_config,
            logging=primary_logging_config,
            metrics=metrics_config,
        )

    def _get_optimizer(self, model, is_global: bool = True):
        optimizer_cls = resolve_optimizer(
            self.server_config.optimizer if is_global else self.client_config.optimizer
        )
        optimizer = optimizer_cls(
            model.parameters(),
            **(
                self.client_config.optimizer_kwargs
                if not is_global
                else self.server_config.optimizer_kwargs
            ),
        )
        return optimizer

    def _get_criterion(self, is_global: bool = True):
        criterion_cls = resolve_criterion(
            self.server_config.criterion if is_global else self.client_config.criterion
        )
        criterion = criterion_cls(
            **(
                self.client_config.criterion_kwargs
                if not is_global
                else self.server_config.criterion_kwargs
            )
        )
        return criterion

    def get_client_config(self) -> FederatedCMAMClientConfig:
        return self.client_config

    def get_criterion(self, criterion_name: str, criterion_kwargs: Dict[str, Any] = {}):
        criterion_cls = resolve_criterion(criterion_name=criterion_name)
        criterion = criterion_cls(**criterion_kwargs)
        return criterion
