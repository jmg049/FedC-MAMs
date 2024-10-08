from dataclasses import dataclass

from config.cmam_config import CMAMModelConfig
from config.config import Config
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from config.config import (
    BaseConfig,
    Config,
    DataConfig,
    DatasetConfig,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    resolve_criterion,
    resolve_optimizer,
)
from data.label_functions import avmnist_get_label_fn, cmu_get_label_fn
from torch.optim import Optimizer
from torch.nn import Module


def federated_server_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedIncongruentServerConfig(**value)


def federated_client_constructor(loader, node):
    value = loader.construct_mapping(node)
    return FederatedIncogruentClientConfig(**value)


yaml.SafeLoader.add_constructor(
    "!FederatedIncongruentServerConfig", federated_server_constructor
)
yaml.SafeLoader.add_constructor(
    "!FederatedIncogruentClientConfig", federated_client_constructor
)


@dataclass(kw_only=True)
class FederatedIncongruentServerConfig(BaseConfig):
    num_clients: int
    rounds: int
    mm_epochs: int
    cmam_epochs: int | list[int]

    model_config: ModelConfig
    cmam_configs: dict[str, CMAMModelConfig]  ## gotta train em' all
    cls_logging: list[
        LoggingConfig
    ]  ## gotta log em' all (should be the same length as cmam_configs)

    cmam_logging: dict[str, LoggingConfig]

    cls_criterion: str
    cls_criterion_kwargs: dict[str, Any]

    cmam_criterion: str
    cmam_criterion_kwargs: dict[str, Any]

    cls_optimizer: str
    cls_optimizer_kwargs: dict[str, Any]

    cmam_optimizers: dict[str, str]
    cmam_optimizer_kwargs: dict[str, dict[str, Any]]

    cls_scheduler: Optional[str] = None
    cls_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    cmam_schedulers: dict[str, Optional[str]] = None
    cmam_scheduler_kwargs: dict[str, dict[str, Any]] = None

    early_stopping: bool
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float = 0.001

    def __post_init__(self):
        assert len(self.cmam_configs) == len(
            self.logging
        ), "Number of CMAM configs and logging configs must be the same"
        assert len(self.cmam_configs) == len(
            self.cmam_optimizers
        ), "Number of CMAM configs and optimizers must be the same"
        assert len(self.cmam_configs) == len(
            self.cmam_schedulers
        ), "Number of CMAM configs and schedulers must be the same"
        assert len(self.cmam_configs) == len(
            self.cmam_optimizer_kwargs
        ), "Number of CMAM configs and optimizer kwargs must be the same"

        if isinstance(self.cmam_epochs, int):
            self.cmam_epochs = [self.cmam_epochs] * len(self.cmam_configs)

        assert len(self.cmam_epochs) == len(
            self.cmam_configs
        ), "Number of CMAM epochs and configs must be the same"

        assert len(self.cmam_logging) == len(
            self.cmam_configs
        ), "Number of CMAM configs and logging configs must be the same"


@dataclass(kw_only=True)
class FederatedIncogruentClientConfig(BaseConfig):
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
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedIncongruentDataConfig(DataConfig):
    pass


@dataclass(kw_only=True)
class FederatedIncongreuntConfig(Config):
    server_config: FederatedIncongruentServerConfig
    client_config: FederatedIncogruentClientConfig
    data_config: FederatedIncongruentDataConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str | Path | PathLike) -> "FederatedIncongreuntConfig":
        pass

    def _get_optimizer(self) -> Optimizer:
        pass

    def _get_criterion(self) -> Module:
        pass
