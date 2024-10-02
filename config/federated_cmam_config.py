from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from config.cmam_config import CMAMModelConfig
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
from utils import SafeDict
from modalities import Modality


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
class FederatedDataConfig(DataConfig):
    distribution_type: str = "iid"
    sampling_strategy: str = "uniform"
    alpha: float = 1.0
    min_samples_per_client: int = 10
    global_fraction: float = 0.5  # 50% for global by default
    client_proportions: list[float] = field(default_factory=list)
    get_label_fn: Optional[str] | callable = None

    def __post_init__(self):
        match self.get_label_fn:
            case "mosei" | "mosi":
                self.get_label_fn = cmu_get_label_fn
            case "avmnist":
                self.get_label_fn = avmnist_get_label_fn
            case _:
                self.get_label_fn = lambda x: x[1]

    def __iter__(self):
        return super.__iter__()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedDataConfig":
        cls_data = {}
        datasets = {}
        for dataset_name, dataset_config in data.items():
            if dataset_name not in ["train", "validation", "test"]:
                continue
            datasets[dataset_name] = DatasetConfig.from_dict(dataset_config)
        cls_data["datasets"] = datasets
        cls_data["distribution_type"] = data.get("distribution_type", "iid")
        cls_data["sampling_strategy"] = data.get("sampling_strategy", "stratified")
        cls_data["alpha"] = data.get("alpha", 1.0)
        cls_data["min_samples_per_client"] = data.get("min_samples_per_client", 10)
        cls_data["global_fraction"] = data.get("global_fraction", 0.5)
        cls_data["client_proportions"] = data.get("client_proportions", None)
        cls_data["get_label_fn"] = data.get("get_label_fn", None)
        return cls(
            datasets=cls_data["datasets"],
            distribution_type=cls_data["distribution_type"],
            sampling_strategy=cls_data["sampling_strategy"],
            alpha=cls_data["alpha"],
            min_samples_per_client=cls_data["min_samples_per_client"],
            global_fraction=cls_data["global_fraction"],
            client_proportions=cls_data["client_proportions"],
            get_label_fn=cls_data["get_label_fn"],
        )


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
    logging_config: LoggingConfig
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class FederatedCMAMConfig(Config):
    server_config: FederatedCMAMServerConfig
    client_config: FederatedCMAMClientConfig
    data_config: FederatedDataConfig

    @classmethod
    def load(cls, path: str | Path | PathLike, run_id: int) -> "FederatedCMAMConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        experiment_config = data["experiment"]
        experiment_config.run_id = run_id

        data_config = data["data"]
        data_config = FederatedDataConfig.from_dict(data_config)

        federated_config = data["federated"]
        server_config = federated_config["server_config"]
        client_config = federated_config["client_config"]

        server_config.logging_config[
            "model_output_path"
        ] = server_config.logging_config["model_output_path"].format_map(
            SafeDict(
                save_metric=server_config.logging_config["save_metric"],
                experiment_name=experiment_config.name,
                run_id=experiment_config.run_id,
            )
        )

        server_config.logging_config = LoggingConfig.from_dict(
            server_config.logging_config,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        client_config.logging["model_output_path"] = client_config.logging[
            "model_output_path"
        ].format_map(
            SafeDict(
                save_metric=client_config.logging["save_metric"],
                experiment_name=experiment_config.name,
                run_id=experiment_config.run_id,
            )
        )

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
            logging=None,
            metrics=metrics_config,
        )

    def __post_init__(self):
        assert self.server_config.num_clients > 0, "Number of clients must be positive"
        assert self.server_config.rounds > 0, "Number of rounds must be positive"

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
