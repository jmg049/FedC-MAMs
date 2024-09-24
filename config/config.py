from dataclasses import dataclass, field
import os
from pathlib import Path
import traceback
from typing import Iterator, Optional, Dict, Any, Tuple
import numpy as np
import torch
import yaml
import time
import logging
from cmam_loss import CMAMLoss
from models.cmams import BasicCMAM
from models.utt_fusion_model import UttFusionModel
from modalities import Modality

LOGGER = logging.getLogger("rich")


def modality_constructor(loader, node):
    value = loader.construct_scalar(node)
    return Modality.from_str(value)


yaml.SafeLoader.add_constructor("!Modality", modality_constructor)


@dataclass(kw_only=True)
class BaseConfig:
    """Base configuration class with utility methods."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a config instance from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for the experiment."""

    name: str
    seed: Optional[int] = None
    device: str = "cuda"
    debug: bool = False
    run_id: int = -1
    do_test: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(time.time())

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        np.random.seed(self.seed)

        LOGGER.info(f"Seed: {self.seed}")
        match torch.cuda.is_available():
            case True:
                self.device = torch.device(self.device)
            case False:
                self.device = torch.device("cpu")

        LOGGER.info(f"Device: {self.device}")


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for a single dataset."""

    dataset: str
    data_fp: str
    target_modality: str
    split: str
    shuffle: bool = False
    is_cmam_dataset: Tuple[bool, str] = (False, "")
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Iterator function for DatasetConfig.
        Yields tuples of (attribute_name, attribute_value).
        """
        for f in self.__dataclass_fields__:
            yield f, getattr(self, f)


@dataclass
class DataConfig(BaseConfig):
    """Configuration for all datasets."""

    datasets: Dict[str, DatasetConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        datasets = {}
        for dataset_name, dataset_config in data.items():
            datasets[dataset_name] = DatasetConfig.from_dict(dataset_config)
        return cls(datasets=datasets)


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for a model."""

    name: str
    pretrained_path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def from_dict(data: dict):
        return ModelConfig(
            name=data["name"],
            pretrained_path=data.get("pretrained_path", None),
            kwargs=data.get("kwargs", None),
        )


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for logging."""

    save_metric: str
    save_condition: Optional[str] = "AVL"
    log_path: str = "logs/log.txt"
    model_output_path: str = "models/model.pt"
    metrics_path: str = "metrics/metrics.josn"
    run_id: Optional[str] = None
    experiment_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, run_id: str, experiment_name: str):
        """Create LoggingConfig from a dictionary with formatted paths."""
        log_path = Path(
            data["log_path"].format(experiment_name=experiment_name, run_id=run_id)
        )
        model_output_path = Path(
            data["model_output_path"].format(
                experiment_name=experiment_name,
                run_id=run_id,
                save_metric=data["save_metric"],
            )
        )
        print(f"MOP: {model_output_path}")
        print(f"Run ID: {run_id}")
        print(f"metriccs {data['metrics_paths']}")
        metrics_path = Path(
            data["metrics_paths"].format(experiment_name=experiment_name, run_id=run_id)
        )
        print(f"metrics path: {metrics_path}")

        return cls(
            save_metric=data["save_metric"],
            log_path=str(log_path),
            model_output_path=str(model_output_path),
            metrics_path=str(metrics_path),
            run_id=run_id,
            experiment_name=experiment_name,
        )


def ensure_directories(config: LoggingConfig) -> None:
    """
    Ensure that all directories specified in the LoggingConfig exist.

    Args:
        config (LoggingConfig): The logging configuration.
    """
    directories = [
        os.path.dirname(config.log_path),
        os.path.dirname(config.model_output_path),
        os.path.dirname(config.metrics_path),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


@dataclass
class MetricsConfig(BaseConfig):
    """Configuration for metrics."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    main_metric: str = "accuracy"

    def items(self):
        return self.metrics.items()


@dataclass(kw_only=True)
class Config(BaseConfig):
    """Main configuration class."""

    experiment: ExperimentConfig
    data: DataConfig
    model: Optional[ModelConfig] = None
    logging: Optional[LoggingConfig] = None
    metrics: Optional[MetricsConfig] = None
    _config_path: Optional[str] = None

    def save(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def ensure_directories(self):
        """Ensure that all directories specified in the LoggingConfig exist."""
        print(self.logging)
        if self.logging:
            directories = [
                os.path.dirname(self.logging.log_path),
                os.path.dirname(self.logging.model_output_path),
                os.path.dirname(self.logging.metrics_path),
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                LOGGER.info(f"Created directory: {directory}")

    def get_optimizer(self, model):
        """Get the optimizer based on the configuration."""
        try:
            optimizer_class = resolve_optimizer(self.training.optimizer)
            optimizer = optimizer_class(
                model.parameters(), **self.training.optim_kwargs
            )

            return optimizer
        except Exception as e:
            error_msg = f"Error creating optimizer: {str(e)}\n{traceback.format_exc()}"
            raise Exception(error_msg)

    def get_scheduler(self, optimizer):
        """Get the scheduler based on the configuration."""
        try:
            scheduler_class = resolve_scheduler(self.training.scheduler)

            ## if the it is a lambda scheduler, we need to pass do some more work
            if self.training.scheduler == "lambda":
                scheduler_args = self.training.scheduler_args
                lambda_lr = scheduler_args.pop("lr_lambda")
                lambda_func = eval(lambda_lr, scheduler_args)
                return scheduler_class(optimizer, lr_lambda=lambda_func)
            else:
                scheduler = scheduler_class(optimizer, **self.training.scheduler_args)
            return scheduler
        except Exception as e:
            error_msg = f"Error creating scheduler: {str(e)}\n{traceback.format_exc()}"
            raise Exception(error_msg)


def resolve_optimizer(optimizer_name: str):
    """Resolve optimizer name to PyTorch optimizer class."""
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    return optimizers[optimizer_name]


def resolve_criterion(criterion_name: str):
    """Resolve criterion name to PyTorch loss function class."""
    criteria = {
        "cross_entropy": torch.nn.CrossEntropyLoss,
        "l1": torch.nn.L1Loss,
        "cmam": CMAMLoss,
    }
    if criterion_name.lower() not in criteria:
        raise ValueError(f"Invalid criterion: {criterion_name}")
    return criteria[criterion_name.lower()]


def resolve_scheduler(scheduler_name: str) -> torch.optim.lr_scheduler._LRScheduler:
    """Resolve scheduler name to PyTorch scheduler class."""
    schedulers = {
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "lambda": torch.optim.lr_scheduler.LambdaLR,
    }
    if scheduler_name not in schedulers:
        raise ValueError(f"Invalid scheduler: {scheduler_name}")
    return schedulers[scheduler_name]


def resolve_model_name(model_name: str):
    match model_name.lower():
        case "uttfusionmodel":
            return UttFusionModel
        case "basiccmam":
            return BasicCMAM
        case _:
            raise ValueError(f"Invalid model: {model_name}")
