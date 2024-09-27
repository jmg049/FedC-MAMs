from typing import Any, Dict, Union
import numpy as np
import torch
from torch.nn import (
    Parameter,
    Module,
    Linear,
    ReLU,
    Sequential,
    Dropout,
    BatchNorm1d,
    Identity,
    ModuleDict,
    MultiheadAttention,
    ModuleList,
)

from cmam_loss import CMAMLoss
from models import resolve_encoder
from modalities import Modality


class BasicCMAM(Module):
    def __init__(
        self,
        input_encoder_info: Dict[Modality, Dict[str, Any]],
        target_modality: Modality,
        assoc_net_input_size: int,
        assoc_net_hidden_size: int,
        assoc_net_output_size: int,
        *,
        assoc_dropout: float = 0.0,
        assoc_use_bn: bool = False,
        fusion_fn: str = "concat",
        grad_clip: float = 0.0,
    ):
        super(BasicCMAM, self).__init__()
        self.encoders = ModuleDict()
        for modality, encoder_params in input_encoder_info.items():
            if isinstance(encoder_params, Module):
                self.encoders[str(modality)] = encoder_params
                continue
            encoder_cls = resolve_encoder(encoder_params["name"])
            encoder_params.pop("name")
            self.encoders[str(modality)] = encoder_cls(**encoder_params)
        assert (
            target_modality not in self.encoders
        ), "The target should not be in the input modalities"
        self.target_modality = target_modality
        self.assoc_net = Sequential(
            Linear(assoc_net_input_size, assoc_net_hidden_size),
            BatchNorm1d(assoc_net_hidden_size) if assoc_use_bn else Identity(),
            ReLU(),
            Dropout(assoc_dropout) if assoc_dropout > 0 else Identity(),
            Linear(assoc_net_hidden_size, assoc_net_output_size),
        )

        self.x_dim = assoc_net_input_size
        self.z_dim = assoc_net_output_size

        match fusion_fn.lower():
            case "concat":
                self.fusion_fn = torch.cat
            case "sum":
                self.fusion_fn = torch.sum
            case "mean":
                self.fusion_fn = torch.mean
            case _:
                raise ValueError(f"Unknown fusion function: {fusion_fn}")

        self.predictions_metric_recorder = None
        self.rec_metric_recorder = None
        self.grad_clip = grad_clip

    def to(self, device):
        super().to(device)
        for encoder in self.encoders.values():
            encoder.to(device)
        self.assoc_net.to(device)
        return self

    def set_predictions_metric_recorder(self, metric_recorder):
        self.predictions_metric_recorder = metric_recorder

    def set_rec_metric_recorder(self, metric_recorder):
        self.rec_metric_recorder = metric_recorder

    def reset_metric_recorders(self):
        if self.predictions_metric_recorder is not None:
            self.predictions_metric_recorder.reset()

        if self.rec_metric_recorder is not None:
            self.rec_metric_recorder.reset()

    def forward(
        self,
        modalities: Union[Dict[Modality, torch.Tensor], torch.Tensor],
        return_z: bool = False,
    ) -> torch.Tensor:
        if isinstance(modalities, dict):
            embeddings = [
                self.encoders[modality](data) for modality, data in modalities.items()
            ]
        else:
            encoder_keys = list(self.encoders.keys())
            assert len(encoder_keys) == 1, "Single modality expected"
            embeddings = self.encoders[encoder_keys[0]](modalities)

        z = self.fusion_fn(embeddings, dim=1)
        return self.assoc_net(z) if not return_z else (self.assoc_net(z), z)

    def train_step(
        self,
        batch: Dict[Modality, torch.Tensor],
        labels: torch.Tensor,
        cmam_criterion: CMAMLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        trained_model: Module,
    ):
        self.train()
        target_modality = batch[self.target_modality]
        input_modalities = {
            modality: batch[modality].float().to(device) for modality in self.encoders
        }

        mi_input_modalities = [v.clone() for k, v in input_modalities.items()]

        labels = labels.to(device)

        # Get the target embedding without computing gradients
        with torch.no_grad():
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))

        # Ensure trained_model's parameters do not require gradients
        for param in trained_model.parameters():
            param.requires_grad = False

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through CMAM
        rec_embd = self.forward(input_modalities)

        # Compute reconstruction loss

        # Prepare input for the pretrained model
        encoder_data = {
            str(k)[0]: batch[k].to(device=device) for k in self.encoders.keys()
        }
        m_kwargs = {
            **encoder_data,
            f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
            f"is_embd_{str(self.target_modality)[0]}": True,
        }
        # Compute logits without torch.no_grad()
        logits = trained_model(**m_kwargs, device=device)
        predictions = logits.argmax(dim=1)

        pred_metrics = self.predictions_metric_recorder.calculate_metrics(
            predictions, labels
        )

        # Total loss and backward pass
        loss_dict = cmam_criterion(
            predictions=rec_embd,
            targets=target_embd,
            originals=mi_input_modalities,
            reconstructed=rec_embd,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

        # Compute reconstruction metrics
        if self.rec_metric_recorder is not None:
            rec_metrics = self.rec_metric_recorder.calculate_metrics(
                rec_embd, target_embd
            )

        return {
            "loss": total_loss.item(),
            **other_losses,
            **rec_metrics,
            **pred_metrics,
        }

    def evaluate(
        self,
        batch,
        labels,
        cmam_criterion: CMAMLoss,
        device,
        trained_model,
        return_eval_data=False,
    ):
        self.eval()
        trained_model.eval()
        with torch.no_grad():
            target_modality = batch[self.target_modality]
            input_modalities = {
                modality: batch[modality].float().to(device)
                for modality in self.encoders
            }
            mi_input_modalities = [v.clone() for k, v in input_modalities.items()]
            miss_type = batch[
                "miss_type"
            ]  ## should be a list of the same string/miss_type

            labels = labels.to(device)

            ## get the target
            target_modality = target_modality.to(device)

            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality)
            rec_embd = self.forward(input_modalities)

            rec_metrics = self.rec_metric_recorder.calculate_metrics(
                rec_embd, target_embd
            )

            encoder_data = {
                str(k)[0]: batch[k].to(device=device) for k in self.encoders.keys()
            }
            m_kwargs = {
                **encoder_data,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                f"is_embd_{str(self.target_modality)[0]}": True,
            }
            logits = trained_model(**m_kwargs, device=device)
            predictions = logits.argmax(dim=1)

            ## compute all the losses
            loss_dict = cmam_criterion(
                predictions=rec_embd,
                targets=target_embd,
                originals=mi_input_modalities,  ## None for now since they refer to the original and reconstructed data for Cyclic Consistency Loss
                reconstructed=rec_embd,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            total_loss = loss_dict["total_loss"]
            other_losses = {
                k: v.item() for k, v in loss_dict.items() if k != "total_loss"
            }

            ## cls metrics
            pred_metrics = self.predictions_metric_recorder.calculate_metrics(
                predictions, labels
            )
            miss_type = np.array(miss_type)
            for m_type in set(miss_type):
                mask = miss_type == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                mask_metrics = self.predictions_metric_recorder.calculate_metrics(
                    predictions=mask_preds,
                    targets=mask_labels,
                    skip_metrics=["ConfusionMatrix"],
                )
                for k, v in mask_metrics.items():
                    pred_metrics[f"{k}_{m_type.replace('z', '').upper()}"] = v
            self.train()

            if return_eval_data:
                return {
                    "loss": total_loss.item(),
                    **other_losses,
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd": rec_embd,
                    "target_embd": target_embd,
                    **rec_metrics,
                    **pred_metrics,
                }

            return {
                "loss": total_loss.item(),
                **other_losses,
                **rec_metrics,
                **pred_metrics,
            }


class DualCMAM(Module):
    """
    Given a single modality this C-MAM will reconstruct the embeddings of two other modalities.
    """

    def __init__(
        self,
        input_encoder: Module,
        shared_encoder_output_size: int,
        decoder_hidden_size: int,
        target_modality_one_embd_size: int,
        target_modality_two_embd_size: int,
        attention_heads: int = 2,
        dropout: float = 0.1,
        grad_clip: float = 0.0,
    ):
        super(DualCMAM, self).__init__()

        self.input_encoder = input_encoder
        self.attention = MultiheadAttention(
            embed_dim=shared_encoder_output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.decoders = ModuleList(
            [
                Sequential(
                    Linear(shared_encoder_output_size, decoder_hidden_size),
                    ReLU(),
                    Dropout(dropout),
                    Linear(decoder_hidden_size, target_modality_one_embd_size),
                ),
                Sequential(
                    Linear(shared_encoder_output_size, decoder_hidden_size),
                    ReLU(),
                    Dropout(dropout),
                    Linear(decoder_hidden_size, target_modality_two_embd_size),
                ),
            ]
        )

        self.grad_clip = grad_clip
        self.modality_weights = Parameter(torch.ones(2))

    def forwad(self, input_modality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_embd = self.input_encoder(
            input_modality
        )  ## expects the encoder to produce a single, flat tensor with shape (batch_size, embd_size)

        attention_encoding, _ = self.attention(input_embd, input_embd, input_embd)

        reconstructed_embd_one = self.decoders[0](attention_encoding)
        reconstructed_embd_two = self.decoders[1](attention_encoding)

        return reconstructed_embd_one, reconstructed_embd_two
