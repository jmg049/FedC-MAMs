from functools import partial
from typing import Any, Dict
import numpy as np
from torch import Tensor
from torch.nn.functional import softmax
import torch
from torch.nn import (
    Module,
    Sequential,
    ReLU,
    Linear,
    MaxPool2d,
    Dropout,
    Flatten,
    Identity,
)
from torch.optim import Optimizer
from modalities import Modality

from models import ConvBlock, ConvBlockArgs
from utils.metric_recorder import MetricRecorder


class MNISTAudio(Module):
    def __init__(
        self,
        conv_block_one_args: ConvBlockArgs,
        conv_block_two_args: ConvBlockArgs,
        hidden_dim: int,
        *,
        conv_batch_norm: bool = True,
        max_pool_kernel_size: int | tuple[int, int] = (2, 2),
    ):
        super(MNISTAudio, self).__init__()
        conv_block = ConvBlock(
            conv_block_one_args=conv_block_one_args,
            conv_block_two_args=conv_block_two_args,
            batch_norm=conv_batch_norm,
        )

        ## calculate the output dimensions of the conv block
        conv_block_out_dim = 24064  ## adjust this value based on the output of the conv block unfortunately, it is not trivial to calculate this value before hand

        fc_in = Linear(conv_block_out_dim, hidden_dim)
        max_pool = MaxPool2d(kernel_size=max_pool_kernel_size)
        self.hidden_dim = hidden_dim
        self.net = Sequential(
            conv_block,
            max_pool,
            Flatten(),
            fc_in,
        )

    def get_embedding_size(self):
        return self.hidden_dim

    def forward(self, audio: Tensor) -> Tensor:
        audio = audio.unsqueeze(1)
        return self.net(audio)

    def __str__(self):
        return str(self.net)


class MNISTImage(Module):
    def __init__(
        self,
        conv_block_one_one_args: ConvBlockArgs,
        conv_block_one_two_args: ConvBlockArgs,
        conv_block_two_one_args: ConvBlockArgs,
        conv_block_two_two_args: ConvBlockArgs,
        hidden_dim: int,
        *,
        conv_batch_norm: bool = True,
        max_pool_kernel_size: int | tuple[int, int] = (2, 2),
    ) -> "MNISTImage":
        super(MNISTImage, self).__init__()
        conv_block_one = ConvBlock(
            conv_block_one_args=conv_block_one_one_args,
            conv_block_two_args=conv_block_one_two_args,
            batch_norm=conv_batch_norm,
        )

        conv_block_two = ConvBlock(
            conv_block_one_args=conv_block_two_one_args,
            conv_block_two_args=conv_block_two_two_args,
            batch_norm=conv_batch_norm,
        )
        max_pool = MaxPool2d(kernel_size=max_pool_kernel_size)
        conv_block_out_dim = 3136  ## adjust this value based on the output of the conv block unfortunately, it is not trivial to calculate this value before hand
        self.hidden_dim = hidden_dim
        self.net = Sequential(
            conv_block_one,
            max_pool,
            conv_block_two,
            max_pool,
            Flatten(),
            Linear(conv_block_out_dim, hidden_dim),
        )

    def get_embedding_size(self):
        return self.hidden_dim

    def forward(self, image: Tensor) -> Tensor:
        # image = image.unsqueeze(1)
        return self.net(image)

    def __str__(self):
        return str(self.net)


class AVMNIST(Module):
    NUM_CLASSES = 10

    def __init__(
        self,
        audio_encoder: MNISTAudio,
        image_encoder: MNISTImage,
        hidden_dim: int,
        metric_recorder: MetricRecorder,
        dropout: float = 0.0,
        fusion_fn: str = "concat",
    ):
        super(AVMNIST, self).__init__()

        self.embd_size_A = audio_encoder.net[-1].out_features
        self.embd_size_I = image_encoder.net[-1].out_features
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        fc_fusion = Linear(self.embd_size_A + self.embd_size_I, hidden_dim)
        fc_intermediate = Linear(hidden_dim, hidden_dim // 2)
        fc_out = Linear(hidden_dim // 2, AVMNIST.NUM_CLASSES)

        self.net = Sequential(
            fc_fusion,
            ReLU(),
            Dropout(dropout) if dropout > 0 else Identity(),
            fc_intermediate,
            ReLU(),
            fc_out,
        )

        match fusion_fn.lower():
            case "concat":
                self.fusion_fn = partial(torch.cat, dim=1)
            case _:
                raise ValueError(f"Unknown fusion function: {fusion_fn}")

        self.metric_recorder = metric_recorder

    def get_encoder(self, modality: str | Modality) -> Module:
        if isinstance(modality, str):
            modality = Modality.from_str(modality)

            match modality:
                case Modality.AUDIO:
                    return self.audio_encoder
                case Modality.IMAGE:
                    return self.image_encoder
                case _:
                    raise ValueError(f"Unknown modality: {modality}")

    def flatten_parameters(self):
        pass

    def forward(
        self,
        A: Tensor = None,
        I: Tensor = None,
        is_embd_A: bool = False,
        is_embd_I: bool = False,
        **kwargs,
    ) -> Tensor:
        assert not all((A is None, I is None)), "At least one of A, I must be provided"
        assert not all([is_embd_A, is_embd_I]), "Cannot have all embeddings as True"

        if A is not None:
            batch_size = A.size(0)
        else:
            batch_size = I.size(0)

        if A is None:
            A = torch.zeros(batch_size, self.embd_size_A)
        if I is None:
            I = torch.zeros(batch_size, self.embd_size_I)

        audio = self.audio_encoder(A) if not is_embd_A else A
        image = self.image_encoder(I) if not is_embd_I else I
        z = self.fusion_fn((audio, image))
        return self.net(z)

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
    ) -> Dict:
        A, I, labels, missing_index, _miss_type = (
            batch["A"],
            batch["I"],
            batch["label"],
            batch["missing_index"],
            batch["miss_type"],
        )

        A, I, labels, missing_index = (
            A.to(device),
            I.to(device),
            labels.to(device),
            missing_index.to(device),
        )

        for i in range(A.size(0)):
            A[i] = A[i] * missing_index[i, 0].unsqueeze(0).unsqueeze(0)

        for i in range(I.size(0)):
            I[i] = I[i] * missing_index[i, 1].unsqueeze(0).unsqueeze(0)

        A = A.float()
        I = I.float()

        self.train()
        optimizer.zero_grad()

        logits = self.forward(A=A, I=I)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        predictions = softmax(logits, dim=1).argmax(dim=1)

        labels = labels.detach().cpu().numpy()
        metrics = {}
        metrics = self.metric_recorder.calculate_metrics(
            predictions.detach().cpu().numpy(), labels
        )
        return {"loss": loss.item(), **metrics}

    def evaluate(self, batch, criterion: Module, device: torch.device) -> Dict:
        self.eval()

        with torch.no_grad():
            A, I, labels, missing_index, miss_type = (
                batch["A"],
                batch["I"],
                batch["label"],
                batch["missing_index"],
                batch["miss_type"],
            )

            A, I, labels, missing_index = (
                A.to(device),
                I.to(device),
                labels.to(device),
                missing_index.to(device),
            )

            for i in range(A.size(0)):
                A[i] = A[i] * missing_index[i, 0].unsqueeze(0).unsqueeze(0)

            for i in range(I.size(0)):
                I[i] = I[i] * missing_index[i, 1].unsqueeze(0).unsqueeze(0)

            A = A.float()
            I = I.float()

            logits = self.forward(A=A, I=I)

            loss = criterion(logits, labels)
            labels = labels.detach().cpu().numpy()
            predictions = softmax(logits, dim=1).argmax(dim=1)

            metrics = self.metric_recorder.calculate_metrics(
                predictions=predictions, targets=labels
            )

            miss_type = np.array(miss_type)
            for m_type in set(miss_type):
                mask = miss_type == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                mask_metrics = self.metric_recorder.calculate_metrics(
                    predictions=mask_preds, targets=mask_labels
                )
                for k, v in mask_metrics.items():
                    metrics[f"{k}_{m_type.replace('z', '').upper()}"] = v

        self.train()
        return {
            "loss": loss.item(),
            **metrics,
        }

    def __str__(self):
        return (
            str(self.audio_encoder)
            + "\n"
            + str(self.image_encoder)
            + "\n"
            + str(self.net)
        )
