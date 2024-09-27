from typing import Dict, Protocol, Any, TypeVar, Type
import torch
from torch.nn import init
from torch.optim import Optimizer
from torch.nn import Conv2d, Linear, BatchNorm2d, Module
from federated import FederatedResult
from models.networks import LSTMEncoder, TextCNN
from utils.metric_recorder import MetricRecorder
from modalities import Modality
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch import Tensor
from torch.nn import ReLU

T = TypeVar("T")


def check_protocol_compliance(instance: T, protocol: Type[T]) -> None:
    """
    Check if the instance complies with the given protocol.
    """
    pass


class FederatedMultimodalClientProtocol(Protocol):
    def set_metric_recorder(self, metric_recorder: MetricRecorder) -> None: ...

    def train_round(
        self,
    ) -> None: ...

    def _train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def _evaluate_step(
        self,
        batch: Dict[str, Any],
    ) -> FederatedResult: ...

    def evaluate_round(
        self,
    ) -> FederatedResult: ...


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

    def flatten_parameters(self) -> None: ...


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


@dataclass
class ConvBlockArgs:
    conv_one_in: int
    conv_one_out: int
    conv_one_kernel_size: int | tuple[int, int] = (3, 3)
    conv_one_stride: int | tuple[int, int] = (1, 1)
    conv_one_padding: int | tuple[int, int] = (1, 1)


class ConvBlock(Module):

    def __init__(
        self,
        conv_block_one_args: ConvBlockArgs,
        conv_block_two_args: ConvBlockArgs,
        batch_norm: bool = True,
    ) -> "ConvBlock":

        super(ConvBlock, self).__init__()
        self.conv_one = Conv2d(
            conv_block_one_args.conv_one_in,
            conv_block_one_args.conv_one_out,
            kernel_size=conv_block_one_args.conv_one_kernel_size,
            stride=conv_block_one_args.conv_one_stride,
            padding=conv_block_one_args.conv_one_padding,
        )
        self.conv_two = Conv2d(
            conv_block_two_args.conv_one_in,
            conv_block_two_args.conv_one_out,
            kernel_size=conv_block_two_args.conv_one_kernel_size,
            stride=conv_block_two_args.conv_one_stride,
            padding=conv_block_two_args.conv_one_padding,
        )
        self.relu = ReLU()
        self.do_batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_one = BatchNorm2d(conv_block_one_args.conv_one_out)
            self.batch_norm_two = BatchNorm2d(conv_block_two_args.conv_one_out)
        else:
            self.batch_norm_one, self.batch_norm_two = None, None

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.conv_one(tensor)
        if self.batch_norm_one is not None:
            tensor = self.batch_norm_one(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv_two(tensor)
        if self.batch_norm_two is not None:
            tensor = self.batch_norm_two(tensor)
        tensor = self.relu(tensor)
        return tensor

    def running_stats(self, val: bool) -> None:
        self.batch_norm_one.track_running_stats = val
        self.batch_norm_two.track_running_stats = val
