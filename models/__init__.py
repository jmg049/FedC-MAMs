from typing import Dict, Protocol, Any, TypeVar, Type
import torch
from torch.nn import init
from torch.optim import Optimizer
from torch.nn import Conv2d, Linear, BatchNorm2d, Module
from models.networks import LSTMEncoder, TextCNN
from utils.metric_recorder import MetricRecorder
from modalities import Modality

T = TypeVar("T")


def check_protocol_compliance(instance: T, protocol: Type[T]) -> None:
    """
    Check if the instance complies with the given protocol.
    """
    pass


class MultimodalModelProtocol(Protocol):
    def set_metric_recorder(self, metric_recorder: MetricRecorder) -> None: ...

    def get_encoder(self, modality: Modality | str) -> Module: ...

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
    ) -> Dict[str, Any]: ...

    def evaluate(
        self, batch: Dict[str, Any], criterion: Module, device: torch.device
    ) -> Dict[str, Any]: ...


class CMAMProtocol(Protocol):
    def set_predictions_metric_recorder(
        self, metric_recorder: MetricRecorder
    ) -> None: ...

    def set_rec_metric_recorder(self, metric_recorder: MetricRecorder) -> None: ...

    def reset_metric_recorders(self) -> None: ...

    def train_step(
        self,
        batch: Dict[str, Any],
        labels: torch.Tensor,
        criterion: Module,
        device: torch.device,
        optimizer: Optimizer,
        trained_model: Module,
    ) -> Dict[str, Any]: ...

    def evaluate(
        self,
        batch: Dict[str, Any],
        labels: torch.Tensor,
        criterion: Module,
        device: torch.device,
        trained_model: Module,
    ) -> Dict[str, Any]: ...


def kaiming_init(module):
    if isinstance(module, (Conv2d, Linear)):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


def resolve_encoder(_type: str):
    match _type.lower():
        case "lstmencoder":
            return LSTMEncoder
        case "textcnn":
            return TextCNN
        case _:
            raise ValueError(f"Unknown encoder type: {_type}")
