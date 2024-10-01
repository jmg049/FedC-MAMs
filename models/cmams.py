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

from utils.metric_recorder import MetricRecorder


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
        metric_recorder: MetricRecorder,
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
        self.metric_recorder = metric_recorder
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

    def reset_metric_recorders(self):
        self.metric_recorder.reset()

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

        pred_metrics = self.metric_recorder.calculate_metrics(predictions, labels)

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

        return {
            "loss": total_loss.item(),
            **other_losses,
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
            pred_metrics = self.metric_recorder.calculate_metrics(predictions, labels)
            miss_type = np.array(miss_type)
            for m_type in set(miss_type):
                mask = miss_type == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                mask_metrics = self.metric_recorder.calculate_metrics(
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
                    **pred_metrics,
                }

            return {
                "loss": total_loss.item(),
                **other_losses,
                **pred_metrics,
            }


class DualCMAM(Module):
    """
    Given a single modality this C-MAM will reconstruct the embeddings of two other modalities.
    """

    def __init__(
        self,
        input_encoder_info: Dict[Modality, Dict[str, Any]],
        shared_encoder_output_size: int,
        decoder_hidden_size: int,
        target_modality_one_embd_size: int,
        target_modality_two_embd_size: int,
        input_modality: Modality,
        target_modality_one: Modality,
        target_modality_two: Modality,
        metric_recorder: MetricRecorder,
        attention_heads: int = 2,
        dropout: float = 0.1,
        grad_clip: float = 0.0,
    ):
        super(DualCMAM, self).__init__()
        self.target_modality_one = target_modality_one
        self.target_modality_two = target_modality_two
        self.input_modality = input_modality
        encoders = []
        for modality, encoder_params in input_encoder_info.items():
            if isinstance(encoder_params, Module):
                encoders.append(encoder_params)
                continue
            encoder_cls = resolve_encoder(encoder_params["name"])
            encoder_params.pop("name")
            encoders.append(encoder_cls(**encoder_params))
        self.input_encoder = encoders[0]
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
        self.modality_weights = Parameter(torch.tensor([1.0, 1.0]))
        self.metric_recorder = metric_recorder

    def reset_metric_recorders(self):
        self.metric_recorder.reset()

    def to(self, device):
        super().to(device)
        self.input_encoder.to(device)
        self.attention.to(device)
        self.decoders.to(device)
        return self

    def forward(
        self, input_modality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embd = self.input_encoder(
            input_modality
        )  ## expects the encoder to produce a single, flat tensor with shape (batch_size, embd_size)

        attention_encoding, _ = self.attention(input_embd, input_embd, input_embd)

        reconstructed_embd_one = self.decoders[0](attention_encoding)
        reconstructed_embd_two = self.decoders[1](attention_encoding)

        return reconstructed_embd_one, reconstructed_embd_two

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
        target_one = batch[self.target_modality_one].float().to(device)
        target_two = batch[self.target_modality_two].float().to(device)
        input_modalities = batch[self.input_modality].float().to(device)
        mi_input_modalities = input_modalities.clone()

        labels = labels.to(device)

        # Get the target embedding without computing gradients
        with torch.no_grad():
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality_one)
            target_embd_one = trained_encoder(target_one)
            trained_encoder = trained_model.get_encoder(self.target_modality_two)
            target_embd_two = trained_encoder(target_two)

        # Ensure trained_model's parameters do not require gradients
        for param in trained_model.parameters():
            param.requires_grad = False

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through CMAM
        rec_embd_one, rec_embd_two = self.forward(input_modalities)

        # prepare input for the pretrained model
        encoder_data = {
            str(self.input_modality)[0]: batch[self.input_modality].to(device=device)
        }

        m_kwargs = {
            **encoder_data,
            f"{str(self.target_modality_one)[0]}": rec_embd_one.to(device=device),
            f"{str(self.target_modality_two)[0]}": rec_embd_two.to(device=device),
            f"is_embd_{str(self.target_modality_one)[0]}": True,
            f"is_embd_{str(self.target_modality_two)[0]}": True,
        }

        # Compute logits without torch.no_grad()
        logits = trained_model(**m_kwargs, device=device)
        predictions = logits.argmax(dim=1)

        pred_metrics = self.metric_recorder.calculate_metrics(predictions, labels)

        rec_one_loss_dict = cmam_criterion(
            predictions=rec_embd_one,
            targets=target_embd_one,
            originals=mi_input_modalities,
            reconstructed=rec_embd_one,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )

        rec_two_loss_dict = cmam_criterion(
            predictions=rec_embd_two,
            targets=target_embd_two,
            originals=mi_input_modalities,
            reconstructed=rec_embd_two,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )

        total_loss = (
            rec_one_loss_dict["total_loss"] * self.modality_weights[0]
            + rec_two_loss_dict["total_loss"] * self.modality_weights[1]
        )

        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        rec_one_other_losses = {
            k: v.item() for k, v in rec_one_loss_dict.items() if k != "total_loss"
        }
        rec_two_other_losses = {
            k: v.item() for k, v in rec_two_loss_dict.items() if k != "total_loss"
        }

        other_losses = {f"rec_{k}_one": v for k, v in rec_one_other_losses.items()}
        other_losses.update(
            {f"rec_{k}_two": v for k, v in rec_two_other_losses.items()}
        )

        return {
            "loss": total_loss.item(),
            **other_losses,
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
            target_one = batch[self.target_modality_one].float().to(device)
            target_two = batch[self.target_modality_two].float().to(device)
            input_modalities = batch[self.input_modality].float().to(device)
            mi_input_modalities = input_modalities.clone()
            miss_type = batch["miss_type"]

            labels = labels.to(device)

            trained_encoder = trained_model.get_encoder(self.target_modality_one)
            target_embd_one = trained_encoder(target_one)
            trained_encoder = trained_model.get_encoder(self.target_modality_two)
            target_embd_two = trained_encoder(target_two)

            rec_embd_one, rec_embd_two = self.forward(input_modalities)

            # prepare input for the pretrained model
            encoder_data = {
                str(self.input_modality)[0]: batch[self.input_modality].to(
                    device=device
                )
            }

            m_kwargs = {
                **encoder_data,
                f"{str(self.target_modality_one)[0]}": rec_embd_one.to(device=device),
                f"{str(self.target_modality_two)[0]}": rec_embd_two.to(device=device),
                f"is_embd_{str(self.target_modality_one)[0]}": True,
                f"is_embd_{str(self.target_modality_two)[0]}": True,
            }

            # Compute logits without torch.no_grad()
            logits = trained_model(**m_kwargs, device=device)
            predictions = logits.argmax(dim=1)

            pred_metrics = self.metric_recorder.calculate_metrics(predictions, labels)

            rec_one_loss_dict = cmam_criterion(
                predictions=rec_embd_one,
                targets=target_embd_one,
                originals=mi_input_modalities,
                reconstructed=rec_embd_one,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            rec_two_loss_dict = cmam_criterion(
                predictions=rec_embd_two,
                targets=target_embd_two,
                originals=mi_input_modalities,
                reconstructed=rec_embd_two,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            total_loss = (
                rec_one_loss_dict["total_loss"] * self.modality_weights[0]
                + rec_two_loss_dict["total_loss"] * self.modality_weights[1]
            )

            rec_one_other_losses = {
                k: v.item() for k, v in rec_one_loss_dict.items() if k != "total_loss"
            }
            rec_two_other_losses = {
                k: v.item() for k, v in rec_two_loss_dict.items() if k != "total_loss"
            }

            miss_type = np.array(miss_type)
            for m_type in set(miss_type):
                mask = miss_type == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                mask_metrics = self.metric_recorder.calculate_metrics(
                    predictions=mask_preds,
                    targets=mask_labels,
                    skip_metrics=["ConfusionMatrix"],
                )
                for k, v in mask_metrics.items():
                    pred_metrics[f"{k}_{m_type.replace('z', '').upper()}"] = v

            ## merge the two loss dicts and give them a prefix
            other_losses = {f"rec_{k}_one": v for k, v in rec_one_other_losses.items()}
            other_losses.update(
                {f"rec_{k}_two": v for k, v in rec_two_other_losses.items()}
            )

            if return_eval_data:
                return {
                    "loss": total_loss.item(),
                    **other_losses,
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd_one": rec_embd_one,
                    "rec_embd_two": rec_embd_two,
                    "target_embd_one": target_embd_one,
                    "target_embd_two": target_embd_two,
                    **pred_metrics,
                }

            return {
                "loss": total_loss.item(),
                **other_losses,
                **pred_metrics,
            }
