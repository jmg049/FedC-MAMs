from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from config import (
    Config,
    BaseConfig,
    MetricsConfig,
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    LoggingConfig,
    resolve_criterion,
)


@dataclass(kw_only=True)
class TrainingConfig(BaseConfig):
    """Configuration for training."""

    epochs: int
    batch_size: int
    optimizer: str = "adam"
    optim_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_args: Dict[str, Any] = field(default_factory=dict)
    criterion: str = "cross_entropy"
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)
    validation_interval: int = 1
    num_modalities: int = 3
    missing_rates: Optional[List[float]] = None
    do_validation_visualization: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    def __post_init__(self):
        assert self.num_modalities >= 1, "Number of modalities must be at least 1"
        if self.missing_rates is not None:
            assert (
                len(self.missing_rates) == self.num_modalities
            ), f"Number of missing rates must match number of modalities. {len(self.missing_rates)} != {self.num_modalities}"
            assert all(
                0.0 <= rate <= 1.0 for rate in self.missing_rates
            ), "Missing rates must be between 0 and 1"
        else:
            self.missing_rates = [0.0] * self.num_modalities


@dataclass(kw_only=True)
class StandardConfig(Config):
    training: TrainingConfig
    metrics: MetricsConfig

    @classmethod
    def load(cls, path: Union[str, Path, PathLike], run_id: int) -> "StandardConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        experiment_config = data["experiment"]
        experiment_config["run_id"] = run_id
        experiment_config = ExperimentConfig.from_dict(experiment_config)

        data_config = data["data"]
        data_config = DataConfig.from_dict(data_config)

        model_config = data["model"]
        model_name = model_config["name"]
        model_pretrained_path = model_config.get("pretrained_path", None)
        model_kwargs = {
            k: v
            for k, v in model_config.items()
            if k not in ["name", "pretrained_path"]
        }
        model_config = ModelConfig(
            name=model_name, pretrained_path=model_pretrained_path, kwargs=model_kwargs
        )

        logging_config = data["logging"]
        logging_config = LoggingConfig.from_dict(
            logging_config,
            experiment_name=experiment_config.name,
            run_id=experiment_config.run_id,
        )

        training_config = data["training"]
        training_config = TrainingConfig.from_dict(training_config)

        metrics_config = data["metrics"]
        metrics_config = MetricsConfig.from_dict(metrics_config)

        return cls(
            experiment=experiment_config,
            data=data_config,
            model=model_config,
            logging=logging_config,
            training=training_config,
            metrics=metrics_config,
        )

    def get_criterion(self, criterion_name: str, criterion_kwargs: Dict[str, Any] = {}):
        criterion_cls = resolve_criterion(criterion_name=criterion_name)
        criterion = criterion_cls(**criterion_kwargs)
        return criterion
