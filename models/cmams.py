from typing import Any, Dict, Tuple, Union
import numpy as np
import torch
from torch.nn import (
    Module,
    Linear,
    ReLU,
    Sequential,
    Dropout,
    BatchNorm1d,
    Identity,
    ModuleDict,
)

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
        rec_criterion: Module,
        pred_criterion: Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        trained_model: Module,
        *,
        rec_weight: float = 1.0,
        cls_weight: float = 1.0,
    ):
        self.train()
        target_modality = batch[self.target_modality]
        input_modalities = {
            modality: batch[modality].float().to(device) for modality in self.encoders
        }

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

        # Forward pass through your model
        rec_embd = self.forward(input_modalities)

        # Compute reconstruction loss
        rec_loss = rec_criterion(rec_embd, target_embd)
        if isinstance(rec_loss, dict):
            total_rec_loss = rec_loss["total_loss"]
        elif isinstance(rec_loss, torch.Tensor):
            total_rec_loss = rec_loss
        else:
            raise ValueError("Unexpected type for rec_loss")
        total_rec_loss = total_rec_loss * rec_weight

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

        # Compute classification loss
        cls_loss = pred_criterion(logits, labels) * cls_weight
        pred_metrics = self.predictions_metric_recorder.calculate_metrics(
            predictions, labels
        )

        # Total loss and backward pass
        loss = total_rec_loss + cls_loss
        loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        # Compute reconstruction metrics
        if self.rec_metric_recorder is not None:
            rec_metrics = self.rec_metric_recorder.calculate_metrics(
                rec_embd, target_embd
            )

        return {
            "loss": loss.item(),
            "rec_loss": total_rec_loss.item(),
            "cls_loss": cls_loss.item(),
            **rec_metrics,
            **pred_metrics,
        }

    def evaluate(
        self,
        batch,
        labels,
        rec_criterion,
        pred_criterion,
        device,
        trained_model,
        *,
        rec_weight=1.0,
        cls_weight=1.0,
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
            miss_type = batch["miss_type"]
            labels = labels.to(device)

            ## get the target
            target_modality = target_modality.to(device)

            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality)
            rec_embd = self.forward(input_modalities)
            rec_loss = rec_criterion(rec_embd, target_embd)

            cls_loss = pred_criterion(rec_embd, labels) * cls_weight

            if isinstance(rec_loss, dict):
                total_rec_loss = rec_loss["total_loss"]
            elif isinstance(rec_loss, torch.Tensor):
                total_rec_loss = rec_loss
            else:
                raise ValueError("Unexpected type for rec_loss")

            total_rec_loss = total_rec_loss * rec_weight

            loss = total_rec_loss + cls_loss
            ## rec metrics
            if self.rec_metric_recorder is not None:
                rec_metrics = self.rec_metric_recorder.calculate_metrics(
                    rec_embd, target_embd
                )

            ## cls metrics
            if self.predictions_metric_recorder is not None:
                encoder_data = {
                    str(k)[0]: batch[k].to(device=device) for k in self.encoders.keys()
                }
                m_kwargs = {
                    **encoder_data,
                    f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                    f"is_embd_{str(self.target_modality)[0]}": True,
                }
                with torch.no_grad():
                    logits = trained_model(**m_kwargs, device=device)
                    predictions = logits.argmax(dim=1)
                    cls_loss = pred_criterion(logits, labels)

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
                    "loss": loss.item(),
                    "rec_loss": total_rec_loss.item(),
                    "cls_loss": cls_loss.item(),
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd": rec_embd,
                    "target_embd": target_embd,
                    **rec_metrics,
                    **pred_metrics,
                }

            return {
                "loss": loss.item(),
                "rec_loss": total_rec_loss.item(),
                "cls_loss": cls_loss.item(),
                **rec_metrics,
                **pred_metrics,
            }
