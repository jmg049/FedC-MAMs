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
from models import ConvBlockArgs
from models.avmnist import AVMNIST, MNISTAudio, MNISTImage
from models.cmams import BasicCMAM, DualCMAM
from models.utt_fusion_model import UttFusionModel
from modalities import Modality

from utils import SafeDict

LOGGER = logging.getLogger("rich")


def modality_constructor(loader, node):
    value = loader.construct_scalar(node)
    return Modality.from_str(value)


def mnist_audio_constructor(loader, node):
    value = loader.construct_mapping(node)
    return MNISTAudio(**value)


def mnist_image_constructor(loader, node):
    value = loader.construct_mapping(node)
    return MNISTImage(**value)


def conv_block_constructor(loader, node):
    value = loader.construct_mapping(node)
    return ConvBlockArgs(**value)


def model_config_constructor(loader, node):
    value = loader.construct_mapping(node)
    return ModelConfig.from_dict(value)


def experiment_config_constructor(loader, node):
    value = loader.construct_mapping(node)
    return ExperimentConfig(**value)


yaml.SafeLoader.add_constructor("!Modality", modality_constructor)
yaml.SafeLoader.add_constructor("!ConvBlock", conv_block_constructor)
yaml.SafeLoader.add_constructor("!MNISTAudio", mnist_audio_constructor)
yaml.SafeLoader.add_constructor("!MNISTImage", mnist_image_constructor)
yaml.SafeLoader.add_constructor("!ModelConfig", model_config_constructor)
yaml.SafeLoader.add_constructor("!ExperimentConfig", experiment_config_constructor)


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

    def __getitem__(self, key):
        return getattr(self, key)


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

    def items(self):
        yield from self.__iter__()

    def get(self, key, default=None):
        return getattr(self, key, default)


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
        name = data["name"]
        pretrained_path = data.get("pretrained_path", None)
        kwargs = {
            k: v for k, v in data.items() if k != "name" and k != "pretrained_path"
        }
        return ModelConfig(
            name=name,
            pretrained_path=pretrained_path,
            kwargs=kwargs,
        )


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for logging."""

    log_path: str
    run_id: str
    experiment_name: str
    save_metric: Optional[str] = None
    model_output_path: Optional[str] = None
    metrics_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, run_id: str, experiment_name: str):
        """Create LoggingConfig from a dictionary with formatted paths."""
        log_path = data["log_path"].format_map(
            SafeDict(experiment_name=experiment_name, run_id=run_id)
        )

        ## convert all whitespace to underscores

        experiment_name = experiment_name.replace(" ", "_")
        experiment_name = experiment_name.replace("-", "_")

        log_path = log_path.replace(" ", "_")
        log_path = log_path.replace("-", "_")

        f_data = {
            "log_path": log_path,
            "run_id": run_id,
            "experiment_name": experiment_name,
        }

        save_metric = data.get("save_metric", None)
        if save_metric:
            save_metric = f"_{save_metric}"
        else:
            save_metric = ""

        f_data["save_metric"] = save_metric

        model_output_path = data.get("model_output_path", None)
        if model_output_path:
            model_output_path = model_output_path.format_map(
                SafeDict(
                    experiment_name=experiment_name,
                    run_id=run_id,
                    save_metric=save_metric,
                )
            )
            model_output_path = model_output_path.replace(" ", "_")
            model_output_path = model_output_path.replace("-", "_")
            f_data["model_output_path"] = model_output_path
        metrics_path = data.get("metrics_path", None)
        if metrics_path:
            metrics_path = data["metrics_path"].format_map(
                SafeDict(experiment_name=experiment_name, run_id=run_id)
            )
            metrics_path = metrics_path.replace(" ", "_")
            metrics_path = metrics_path.replace("-", "_")
            f_data["metrics_path"] = metrics_path

        iid_metrics_path = data.get("iid_metrics_path", None)
        if iid_metrics_path:
            iid_metrics_path = iid_metrics_path.format_map(
                SafeDict(experiment_name=experiment_name, run_id=run_id)
            )
            iid_metrics_path = iid_metrics_path.replace(" ", "_")
            iid_metrics_path = iid_metrics_path.replace("-", "_")
            f_data["iid_metrics_path"] = iid_metrics_path

        return cls(**f_data)


@dataclass
class MetricsConfig(BaseConfig):
    """Configuration for metrics."""

    metrics: Dict[str, Any] = field(default_factory=dict)

    def items(self):
        return self.metrics.items()


@dataclass(kw_only=True)
class Config(BaseConfig):
    """Main configuration class."""

    experiment: ExperimentConfig
    data: DataConfig = None
    model: Optional[ModelConfig] = None
    logging: Optional[LoggingConfig] = None
    metrics: Optional[MetricsConfig] = None
    _config_path: Optional[str] = None

    def save(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

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
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    return optimizers[optimizer_name.lower()]


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
        case "avmnist":
            return AVMNIST
        case "basiccmam":
            return BasicCMAM
        case "dualcmam":
            return DualCMAM
        case _:
            raise ValueError(f"Invalid model: {model_name}")
