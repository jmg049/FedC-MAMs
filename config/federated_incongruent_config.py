from dataclasses import dataclass

from config.cmam_config import CMAMModelConfig
from config.config import Config, MetricsConfig
from dataclasses import field
from os import PathLike
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from config.config import (
    BaseConfig,
    LoggingConfig,
    ModelConfig,
    resolve_criterion,
    resolve_optimizer,
)
from config.federated_config import FederatedDataConfig, FederatedLoggingConfig
from torch.optim import Optimizer
from torch.nn import Module
from modalities import Modality


def federated_incongruent_server_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedIncongruentServerConfig(**value)


def federated_incongruent_client_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedIncongruentClientConfig(**value)


yaml.SafeLoader.add_constructor(
    "!FederatedIncongruentServerConfig", federated_incongruent_server_constructor
)
yaml.SafeLoader.add_constructor(
    "!FederatedIncongruentClientConfig", federated_incongruent_client_constructor
)


@dataclass(kw_only=True)
class FederatedIncongruentServerConfig(BaseConfig):
    num_clients: int
    rounds: int
    mm_epochs: int
    cmam_epochs: int | list[int] = None
    aggregation_strategy: str = "fedavg"
    selection_strategy: str = "all"
    model_config: ModelConfig
    cmam_configs: dict[str, CMAMModelConfig] = None
    cls_logging: LoggingConfig

    cmam_logging: dict[str, LoggingConfig] = None

    cls_criterion: str
    cls_criterion_kwargs: dict[str, Any]

    cmam_criterion: str = None
    cmam_criterion_kwargs: dict[str, Any] = None

    cls_optimizer: str
    cls_optimizer_kwargs: dict[str, Any]

    cmam_optimizers: dict[str, str] = None
    cmam_optimizer_kwargs: dict[str, dict[str, Any]] = None

    cls_scheduler: Optional[str] = None
    cls_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    cmam_schedulers: dict[str, Optional[str]] = None
    cmam_scheduler_kwargs: dict[str, dict[str, Any]] = None

    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "loss"
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedIncongruentClientConfig(BaseConfig):
    local_epochs: int
    local_batch_size: int
    logging: LoggingConfig

    model_config: ModelConfig

    cls_optimizer: str
    cls_optimizer_kwargs: dict[str, Any]

    cls_criterion: str
    cls_criterion_kwargs: dict[str, Any]

    ## No cmam optimizer or criterion here, we won't have the ground truth embeddings to compare against
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "loss"
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedIncongruentConfig(Config):
    server_config: FederatedIncongruentServerConfig
    client_config: FederatedIncongruentClientConfig
    data_config: FederatedDataConfig
    logging: LoggingConfig

    @classmethod
    def load(
        cls, path: str | Path | PathLike, run_id: int
    ) -> "FederatedIncongruentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        experiment_config = data["experiment"]
        experiment_config.run_id = run_id

        data_config = data["data"]
        data_config = FederatedDataConfig.from_dict(data_config)

        federated_config = data["federated"]
        server_config = federated_config["server_config"]
        client_config = federated_config["client_config"]

        # Handle optional CMAM configurations
        if server_config.cmam_configs is not None:
            ## change the cmam logging keys to Modality
            cmam_loggings = {}
            for key, cmam_logging in server_config["cmam_logging"].items():
                k = Modality.from_str(key)
                cmam_loggings[k] = cmam_logging

            for key, cmam_logging in cmam_loggings.items():
                server_config["cmam_logging"][key] = LoggingConfig.from_dict(
                    cmam_logging,
                    experiment_name=experiment_config.name,
                    run_id=experiment_config.run_id,
                )

            keys_to_remove = [
                k
                for k in server_config["cmam_logging"].keys()
                if k not in cmam_loggings.keys()
            ]
            for key in keys_to_remove:
                server_config["cmam_logging"].pop(key)
        server_config.cls_logging = LoggingConfig.from_dict(
            server_config.cls_logging,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )
        client_config.logging = LoggingConfig.from_dict(
            client_config.logging,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        logging = data["logging"]
        primary_logging_config = FederatedLoggingConfig.from_dict(
            logging,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        metrics_config = data["metrics"]

        metrics_config = MetricsConfig.from_dict(metrics_config)

        return cls(
            experiment=experiment_config,
            server_config=server_config,
            client_config=client_config,
            data_config=data_config,
            logging=primary_logging_config,
            metrics=metrics_config,
        )

    def _get_optimizer(self, models, is_global: bool = True) -> Optimizer:
        if is_global:
            optimizer_name = self.server_config.cls_optimizer
            optimizer_kwargs = self.server_config.cls_optimizer_kwargs
        else:
            optimizer_name = self.client_config.cls_optimizer
            optimizer_kwargs = self.client_config.cls_optimizer_kwargs

        optimizer_cls = resolve_optimizer(optimizer_name)
        if isinstance(models, list):
            # If we have multiple models, return a single optimizer for all of them.
            all_parameters = []
            for model in models:
                all_parameters.extend(model.parameters())
            return optimizer_cls(all_parameters, **optimizer_kwargs)
        else:
            # Otherwise, return a single optimizer for the single model.
            return optimizer_cls(models.parameters(), **optimizer_kwargs)

    def _get_criterion(self, is_global: bool = True, cmam=False) -> Module:

        if cmam:
            if is_global:
                criterion_name = self.server_config.cmam_criterion
                criterion_kwargs = self.server_config.cmam_criterion_kwargs
            else:
                criterion_name = self.client_config.cmam_criterion
                criterion_kwargs = self.client_config.cmam_criterion_kwargs

        elif is_global:
            criterion_name = self.server_config.cls_criterion
            criterion_kwargs = self.server_config.cls_criterion_kwargs
        else:
            criterion_name = self.client_config.cls_criterion
            criterion_kwargs = self.client_config.cls_criterion_kwargs

        criterion_cls = resolve_criterion(criterion_name)
        return criterion_cls(**criterion_kwargs)

    def get_client_config(self) -> FederatedIncongruentClientConfig:
        return self.client_config

    def get_criterion(
        self, criterion_name: str, criterion_kwargs: Dict[str, Any] = {}
    ) -> Module:
        criterion_cls = resolve_criterion(criterion_name)
        return criterion_cls(**criterion_kwargs)
