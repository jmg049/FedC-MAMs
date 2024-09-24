from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from torch.nn import Module


class UttFusionModel(Module):
    def __init__(
        self,
        classification_layers: List[int],
        input_size_a: int,
        input_size_v: int,
        input_size_l: int,
        embd_size_a: int,
        embd_size_v: int,
        embd_size_l: int,
        embd_method_a: str,
        embd_method_v: str,
        output_dim: int,
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
        self.netL = TextCNN(input_size=input_size_l, embd_size=embd_size_l)

        cls_input_size = embd_size_a + embd_size_v + embd_size_l
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
        self.input_size_l = input_size_l

    def set_metric_recorder(self, metric_recorder):
        self.metric_recorder = metric_recorder

    def get_encoder(self, modality: str):
        match modality:
            case "A_feat":
                return self.netA
            case "V_feat":
                return self.netV
            case "L_feat":
                return self.netL
            case _:
                raise ValueError(f"Unknown modality: {modality}")

    ## new forward function takes in values rather than from self. Missing data is expected to be applied PRIOR to calling forward
    ## C-MAM generated features should be passed in in-place of the original feature(s)
    def forward(
        self,
        A: torch.Tensor = None,
        V: torch.Tensor = None,
        L: torch.Tensor = None,
        is_embd_A: bool = False,
        is_embd_V: bool = False,
        is_embd_L: bool = False,
        device: torch.device = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        assert not all(
            (A is None, V is None, L is None)
        ), "At least one of A, V, L must be provided"
        assert not all(
            [is_embd_A, is_embd_V, is_embd_L]
        ), "Cannot have all embeddings as True"

        if A is not None:
            batch_size = A.size(0)
        elif V is not None:
            batch_size = V.size(0)
        else:
            batch_size = L.size(0)
        if A is None:
            A = torch.zeros(batch_size, 1, self.input_size_a).to(device)
        if V is None:
            V = torch.zeros(
                batch_size,
                1,
                self.input_size_v,
            ).to(device)
        if L is None:
            L = torch.zeros(batch_size, 50, self.input_size_l).to(device)

        a_embd = self.netA(A) if not is_embd_A else A
        v_embd = self.netV(V) if not is_embd_V else V
        t_embd = self.netL(L) if not is_embd_L else L
        fused = torch.cat([a_embd, v_embd, t_embd], dim=-1)
        logits = self.netC(fused)
        return logits

    def evaluate(self, batch, criterion, device) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            (
                A,
                V,
                T,
                labels,
                missing_index,
                miss_type,
            ) = (
                batch["A_feat"],
                batch["V_feat"],
                batch["L_feat"],
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
            loss = criterion(logits, labels)

            labels = labels.detach().cpu().numpy()

            metrics = {}
            if self.metric_recorder is not None:
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
            batch["A_feat"],
            batch["V_feat"],
            batch["L_feat"],
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
        metrics = {}
        if self.metric_recorder is not None:
            metrics = self.metric_recorder.calculate_metrics(
                predictions=predictions, targets=labels
            )

        return {"loss": loss.item(), **metrics}
