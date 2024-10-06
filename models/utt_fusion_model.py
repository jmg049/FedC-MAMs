from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from torch.nn import Module
from modalities import Modality

from utils.metric_recorder import MetricRecorder


class UttFusionModel(Module):
    def __init__(
        self,
        classification_layers: List[int],
        input_size_a: int,
        input_size_v: int,
        input_size_t: int,
        embd_size_a: int,
        embd_size_v: int,
        embd_size_t: int,
        embd_method_a: str,
        embd_method_v: str,
        output_dim: int,
        metric_recorder: MetricRecorder,
        *,
        dropout: float = 0.0,
        use_bn: bool = False,
        clip: float = 0.5,
    ):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(UttFusionModel, self).__init__()

        # acoustic model
        self.netA = LSTMEncoder(
            input_size=input_size_a, hidden_size=embd_size_a, embd_method=embd_method_a
        )

        # visual model
        self.netV = LSTMEncoder(
            input_size=input_size_v, hidden_size=embd_size_v, embd_method=embd_method_v
        )

        # text model
        self.netL = TextCNN(input_size=input_size_t, embd_size=embd_size_t)

        cls_input_size = embd_size_a + embd_size_v + embd_size_t
        self.netC = FcClassifier(
            cls_input_size,
            classification_layers,
            output_dim=output_dim,
            dropout=dropout,
            use_bn=use_bn,
        )

        self.clip = clip
        self.metric_recorder = None
        self.input_size_a = input_size_a
        self.input_size_v = input_size_v
        self.input_size_t = input_size_t

        self.metric_recorder = metric_recorder

    def get_encoder(self, modality: Modality | str):
        if isinstance(modality, str):
            modality = Modality.from_str(modality)
        match modality:
            case Modality.AUDIO:
                return self.netA
            case Modality.VIDEO:
                return self.netV
            case Modality.TEXT:
                return self.netL
            case _:
                raise ValueError(f"Unknown modality: {modality}")

    ## new forward function takes in values rather than from self. Missing data is expected to be applied PRIOR to calling forward
    ## C-MAM generated features should be passed in in-place of the original feature(s)
    def forward(
        self,
        A: torch.Tensor = None,
        V: torch.Tensor = None,
        T: torch.Tensor = None,
        is_embd_A: bool = False,
        is_embd_V: bool = False,
        is_embd_T: bool = False,
        device: torch.device = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        assert not all(
            (A is None, V is None, T is None)
        ), "At least one of A, V, L must be provided"
        assert not all(
            [is_embd_A, is_embd_V, is_embd_T]
        ), "Cannot have all embeddings as True"

        if A is not None:
            batch_size = A.size(0)
        elif V is not None:
            batch_size = V.size(0)
        else:
            batch_size = T.size(0)
        if A is None:
            A = torch.zeros(batch_size, 1, self.input_size_a).to(device)
        if V is None:
            V = torch.zeros(
                batch_size,
                1,
                self.input_size_v,
            ).to(device)
        if T is None:
            T = torch.zeros(batch_size, 50, self.input_size_t).to(device)

        a_embd = self.netA(A) if not is_embd_A else A
        v_embd = self.netV(V) if not is_embd_V else V
        t_embd = self.netL(T) if not is_embd_T else T
        fused = torch.cat([a_embd, v_embd, t_embd], dim=-1)
        logits = self.netC(fused)
        return logits

    def evaluate(
        self, batch, criterion, device, return_test_info: bool = False
    ) -> Dict[str, Any]:
        self.eval()

        if return_test_info:
            all_predictions = []
            all_labels = []
            all_miss_types = []

        with torch.no_grad():
            (
                A,
                V,
                T,
                labels,
                missing_index,
                miss_type,
            ) = (
                batch[Modality.AUDIO],
                batch[Modality.VIDEO],
                batch[Modality.TEXT],
                batch["label"],
                batch["missing_index"],
                batch["miss_type"],
            )

            A, V, T, labels, missing_index = (
                A.to(device),
                V.to(device),
                T.to(device),
                labels.to(device),
                missing_index.to(device),
            )

            A = A * missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            V = V * missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            T = T * missing_index[:, 2].unsqueeze(1).unsqueeze(2)

            A = A.float()
            V = V.float()
            T = T.float()

            logits = self.forward(A, V, T)
            predictions = logits.argmax(dim=1)

            if return_test_info:
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels)
                all_miss_types.append(miss_type)

            (
                binary_preds,
                binary_truth,
                non_zeros_mask,
            ) = UttFusionModel.msa_binarize(
                predictions.cpu().numpy(), labels.cpu().numpy()
            )

            miss_types = np.array(miss_type)
            loss = criterion(logits, labels)

            miss_types = np.array(miss_type)
            metrics = {}

            for miss_type in set(miss_types):
                mask = miss_types == miss_type

                # Apply the mask to get values for the current miss type
                binary_preds_masked = binary_preds[mask]
                binary_truth_masked = binary_truth[mask]
                non_zeros_mask_masked = non_zeros_mask[mask]

                # Calculate metrics for all elements (including zeros)
                binary_metrics = self.metric_recorder.calculate_metrics(
                    predictions=binary_preds_masked, targets=binary_truth_masked
                )

                # Calculate metrics for non-zero elements only using the non_zeros_mask
                non_zeros_binary_preds_masked = binary_preds_masked[
                    non_zeros_mask_masked
                ]
                non_zeros_binary_truth_masked = binary_truth_masked[
                    non_zeros_mask_masked
                ]

                non_zero_metrics = self.metric_recorder.calculate_metrics(
                    predictions=non_zeros_binary_preds_masked,
                    targets=non_zeros_binary_truth_masked,
                )

                # Store metrics in the metrics dictionary
                for k, v in binary_metrics.items():
                    metrics[f"HasZero_{k}_{miss_type.replace('z', '').upper()}"] = v

                for k, v in non_zero_metrics.items():
                    metrics[f"NonZero_{k}_{miss_type.replace('z', '').upper()}"] = v

        if return_test_info:
            return {
                "loss": loss.item(),
                **metrics,
                "predictions": all_predictions,
                "labels": all_labels,
                "miss_types": all_miss_types,
            }
        return {"loss": loss.item(), **metrics}

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> dict:
        """
        Perform a single training step.

        Args:
            batch (tuple): Tuple containing (audio, video, text, labels, mask, lengths).
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            criterion (torch.nn.Module): The loss function.
            device (torch.device): The device to run the computations on.

        Returns:
            dict: A dictionary containing the loss and other metrics.

        """

        A, V, T, labels, missing_index, _miss_type = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch["label"],
            batch["missing_index"],
            batch["miss_type"],
        )

        A, V, T, labels, missing_index = (
            A.to(device),
            V.to(device),
            T.to(device),
            labels.to(device),
            missing_index.to(device),
        )

        A = A * missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        V = V * missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        T = T * missing_index[:, 2].unsqueeze(1).unsqueeze(2)

        A = A.float()
        V = V.float()
        T = T.float()

        self.train()
        logits = self.forward(A, V, T, device=device)
        predictions = logits.argmax(dim=1)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netV.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netL.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netC.parameters(), self.clip)

        optimizer.step()

        labels = labels.detach().cpu().numpy()
        (
            binary_preds,
            binary_truth,
            non_zeros_mask,
        ) = UttFusionModel.msa_binarize(predictions.cpu().numpy(), labels)

        # Calculate metrics for all elements (including zeros)
        binary_metrics = self.metric_recorder.calculate_metrics(
            predictions=binary_preds, targets=binary_truth
        )

        # Calculate metrics for non-zero elements only using the non_zeros_mask
        non_zeros_binary_preds = binary_preds[non_zeros_mask]
        non_zeros_binary_truth = binary_truth[non_zeros_mask]

        non_zero_metrics = self.metric_recorder.calculate_metrics(
            predictions=non_zeros_binary_preds,
            targets=non_zeros_binary_truth,
        )

        # Store the metrics in a dictionary
        metrics = {}

        for k, v in binary_metrics.items():
            metrics[f"HasZero_{k}"] = v

        for k, v in non_zero_metrics.items():
            metrics[f"NonZero_{k}"] = v

        return {"loss": loss.item(), **metrics}

    def flatten_parameters(self):
        self.netA.rnn.flatten_parameters()
        self.netV.rnn.flatten_parameters()

    @staticmethod
    def msa_binarize(preds, labels):
        test_preds = preds - 1
        test_truth = labels - 1
        non_zeros_mask = test_truth != 0

        binary_truth = test_truth >= 0
        binary_preds = test_preds >= 0

        return (
            binary_preds,
            binary_truth,
            non_zeros_mask,
        )
