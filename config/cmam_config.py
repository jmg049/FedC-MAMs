from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from modalities import Modality

from config import (
    BaseConfig,
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    LoggingConfig,
    MetricsConfig,
    Config,
    resolve_criterion,
)


@dataclass(kw_only=True)
class CMAMModelConfig(BaseConfig):
    """Configuration for the CMAM model."""

    name: str
    input_encoder_info: Dict[Modality, Dict[str, Any]]
    target_modality: Modality
    assoc_net_input_size: int
    assoc_net_hidden_size: int
    assoc_net_output_size: int
    assoc_dropout: float = 0.0
    assoc_use_bn: bool = False
    fusion_fn: str = "concat"
    grad_clip: float = 0.0

    def __post_init__(self):
        if self.fusion_fn not in ["concat", "sum", "mean"]:
            raise ValueError(
                f"Invalid fusion function: {self.fusion_fn}. Must be one of ['concat', 'sum', 'mean']"
            )

    def __dict__(self):
        return {
            "target_modality": self.target_modality,
            "assoc_net_input_size": self.assoc_net_input_size,
            "assoc_net_hidden_size": self.assoc_net_hidden_size,
            "assoc_net_output_size": self.assoc_net_output_size,
            "assoc_dropout": self.assoc_dropout,
            "assoc_use_bn": self.assoc_use_bn,
            "fusion_fn": self.fusion_fn,
            "grad_clip": self.grad_clip,
        }


@dataclass(kw_only=True)
class DualCMAMModelConfig(BaseConfig):
    """Configuration for the CMAM model."""

    name: str
    input_encoder_info: Dict[Modality, Dict[str, Any]]
    decoder_hidden_size: int
    shared_encoder_output_size: int
    target_modality_one_embd_size: int
    target_modality_two_embd_size: int
    input_modality: Modality
    target_modality_one: Modality
    target_modality_two: Modality
    attention_heads: int = 2
    dropout: float = 0.0
    grad_clip: float = 0.0

    def __dict__(self):
        return {
            "input_modality": self.input_modality,
            "target_modality_one": self.target_modality_one,
            "target_modality_two": self.target_modality_two,
            "decoder_hidden_size": self.decoder_hidden_size,
            "shared_encoder_output_size": self.shared_encoder_output_size,
            "target_modality_one_embd_size": self.target_modality_one_embd_size,
            "target_modality_two_embd_size": self.target_modality_two_embd_size,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "grad_clip": self.grad_clip,
        }


@dataclass(kw_only=True)
class CMAMTrainingConfig(BaseConfig):
    """Configuration for training a CMAM model."""

    epochs: int
    batch_size: int
    optimizer: str = "adam"
    optim_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_args: Dict[str, Any] = field(default_factory=dict)
    criterion: str = "CMAM"
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)
    rec_weight: float = 1.0
    cls_weight: float = 1.0
    num_modalities: int = -1
    missing_rates: Optional[List[float]] = None
    validation_interval: int = 1
    do_tsne: bool = False
    target_missing_type: str = "AVL"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass(kw_only=True)
class CMAMConfig(Config):
    training: CMAMTrainingConfig
    cmam: CMAMModelConfig | DualCMAMModelConfig
    prediction_metrics: MetricsConfig
    rec_metrics: MetricsConfig

    @classmethod
    def load(cls, path: Union[str, Path, PathLike], run_id: int) -> "CMAMConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        experiment_config = data["experiment"]
        experiment_config["run_id"] = run_id
        experiment_config = ExperimentConfig.from_dict(experiment_config)

        data_config = data["data"]
        data_config = DataConfig.from_dict(data_config)

        ## Base multimodal model, not CMAM
        model_config = data["model"]
        model_config = ModelConfig.from_dict(model_config)

        logging_config = data["logging"]
        logging_config = LoggingConfig.from_dict(
            logging_config, experiment_config.run_id, experiment_config.name
        )

        cmam_training_config = data["training"]
        cmam_training_config = CMAMTrainingConfig.from_dict(cmam_training_config)

        cmam_model_config = data["cmam"]
        if cmam_model_config["name"].lower() == "basiccmam":
            cmam_model_config = CMAMModelConfig.from_dict(cmam_model_config)
        else:
            cmam_model_config = DualCMAMModelConfig.from_dict(cmam_model_config)

        prediction_metrics = data["metrics"]["prediction_metrics"]
        rec_metrics = data["metrics"].get("rec_metrics", {})

        return cls(
            experiment=experiment_config,
            data=data_config,
            model=model_config,
            logging=logging_config,
            training=cmam_training_config,
            cmam=cmam_model_config,
            prediction_metrics=prediction_metrics,
            rec_metrics=rec_metrics,
        )

    def get_criterion(self, criterion_name: str, criterion_kwargs: Dict[str, Any] = {}):
        try:
            criterion_class = resolve_criterion(criterion_name=criterion_name)
            criterion = criterion_class(**criterion_kwargs)
            return criterion
        except Exception as e:
            raise ValueError(f"Could not resolve rec criterion: {e}")
