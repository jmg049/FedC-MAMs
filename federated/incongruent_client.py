from copy import deepcopy
import json
import os
from typing import Any, Optional, Tuple

from numpy import ndarray
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from modalities import Modality
from config.config import LoggingConfig
from config.federated_incongruent_config import FederatedIncongruentClientConfig
from federated import FederatedResult
from models import CMAMProtocol, MultimodalModelProtocol
from utils import (
    SafeDict,
    display_training_metrics,
    display_validation_metrics,
    get_logger,
)

logger = get_logger()


class FederatedIncongruentClient:
    def __init__(
        self,
        client_id: str | int,
        mm_model: MultimodalModelProtocol,
        optimizer: Optimizer,
        criterion: Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        config: FederatedIncongruentClientConfig,
        modality: Modality,
        epochs: int,
        run_id: int,
        cmam: Optional[CMAMProtocol] = None,
        cmam_logging: Optional[LoggingConfig] = None,
        cmam_optimizer: Optional[Optimizer] = None,
        print_fn: callable = print,
    ):
        self.client_id = client_id
        self.mm_model = mm_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.modality = modality
        self.cmam = cmam
        self.cmam_optimizer = cmam_optimizer
        self.print_fn = print_fn
        self.epochs = epochs
        self.run_id = run_id
        self.current_round = 0
        self.best_model_score = (
            float("inf") if config.logging.save_metric == "loss" else float("-inf")
        )
        self.best_model_info = {
            "path": None,
            "score": self.best_model_score,
            "epoch": 0,
            "round": 0,
        }
        self.best_cmam_info = {
            "path": None,
            "score": self.best_model_score,
            "epoch": 0,
            "round": 0,
        }
        self.cmam_logging = cmam_logging

    def get_model_parameters(
        self,
    ) -> Tuple[dict[str, torch.Tensor], Optional[dict[str, dict[str, torch.Tensor]]]]:
        mm_model_params = deepcopy(self.mm_model.state_dict())

        cmam_params = None
        if self.cmam:
            cmam_params = {self.modality: deepcopy(self.cmam.state_dict())}
        return mm_model_params, cmam_params

    def set_mm_model_parameters(self, params: dict[str, torch.Tensor]):
        self.mm_model.load_state_dict(params)

    def set_cmam_parameters(self, params: dict[str, dict[str, torch.Tensor]]):
        if self.cmam and self.modality in params:
            self.cmam.load_state_dict(params[self.modality])

    def train_round(self):
        self.current_round += 1
        self.mm_model.train()
        self.mm_model.to(self.device)
        if self.cmam:
            self.cmam.train()
            self.cmam.to(self.device)

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._update_best_model(val_metrics, epoch)

        self._load_best_model()
        # save best model
        best_model_path = os.path.join(
            self.config.logging.model_output_path,
            str(self.current_round),
            "best.pth",
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(self.mm_model.state_dict(), best_model_path)

        if self.test_loader:
            return self.evaluate_round(dataloader=self.test_loader)
        return self.evaluate_round(dataloader=self.validation_loader)

    def _train_epoch(self, epoch: int):
        self.print_fn(
            f"Training epoch {epoch}/{self.epochs} for client {self.client_id}"
        )

        epoch_metric_recorder = self.mm_model.metric_recorder.clone()
        for batch in tqdm(
            self.train_loader,
            desc=f"Client {self.client_id}: Training Batch",
            ascii=True,
        ):
            self.mm_model.train()
            if self.cmam:
                self.cmam.train()
            self.mm_model.freeze_irrelevant_parameters(self.modality)

            if self.cmam:
                # Use CMAM to assist in training the multimodal model
                results = self.cmam.incongruent_train_step(
                    batch=batch,
                    labels=batch.get("label"),
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    device=self.device,
                    mm_model=self.mm_model,
                )
            else:
                results = self.mm_model.train_step(
                    batch=batch,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    device=self.device,
                )
            epoch_metric_recorder.update_from_dict(results)

        train_metrics = epoch_metric_recorder.get_average_metrics()
        display_training_metrics(train_metrics)

    def _validate_epoch(self, epoch: int) -> dict:
        self.print_fn(
            f"Validating epoch {epoch}/{self.epochs} for client {self.client_id}"
        )
        self.mm_model.eval()
        if self.cmam:
            self.cmam.eval()

        epoch_metric_recorder = self.mm_model.metric_recorder.clone()
        with torch.no_grad():
            for batch in tqdm(
                self.validation_loader,
                desc=f"Client {self.client_id}: Validation Batch",
                ascii=True,
            ):
                if self.cmam:
                    results = self.cmam.incongruent_evaluate(
                        batch=batch,
                        labels=batch.get("label"),
                        mm_model=self.mm_model,
                        device=self.device,
                        criterion=self.criterion,
                    )
                else:
                    results = self.mm_model.evaluate(
                        batch=batch, criterion=self.criterion, device=self.device
                    )
                epoch_metric_recorder.update_from_dict(results)

        val_metrics = epoch_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                self.config.logging.metrics_path,
                f"{self.current_round}/validation",
            ),
            epoch=epoch,
        )

        display_validation_metrics(val_metrics)
        return val_metrics

    def _update_best_model(self, val_metrics: dict, epoch: int):
        save_metric = self.config.logging.save_metric.replace("_", "", 1)
        metric_value = val_metrics.get(save_metric)
        if metric_value is None:
            raise KeyError(
                f"Validation metrics do not contain '{save_metric}'. Available metrics: {val_metrics.keys()}"
            )

        is_better = (
            save_metric == "loss" and metric_value < self.best_model_info["score"]
        ) or (save_metric != "loss" and metric_value > self.best_model_info["score"])

        if is_better:
            self.best_model_info.update(
                {
                    "score": metric_value,
                    "epoch": epoch,
                    "round": self.current_round,
                    "path": os.path.join(
                        self.config.logging.model_output_path,
                        str(self.current_round),
                        "best.pth",
                    ),
                }
            )
            os.makedirs(os.path.dirname(self.best_model_info["path"]), exist_ok=True)
            torch.save(self.mm_model.state_dict(), self.best_model_info["path"])
            self.print_fn(
                f"Best model saved at {self.best_model_info['path']} with score: {self.best_model_info['score']}"
            )

    def _load_best_model(self):
        if self.best_model_info["path"]:
            self.mm_model.load_state_dict(torch.load(self.best_model_info["path"]))
            self.print_fn(f"Loaded best model from {self.best_model_info['path']}")

    def evaluate_round(self, dataloader: DataLoader) -> FederatedResult:
        self.print_fn(f"Evaluating the best model for client {self.client_id}")
        self.mm_model.eval()
        if self.cmam:
            self.cmam.eval()

        original_miss_types = self.train_loader.dataset.selected_missing_types

        missing_types = list(dataloader.dataset.base_dataset.MODALITY_LOOKUP.keys())

        metrics = {}

        for miss_type in missing_types:
            dataloader.dataset.set_selected_missing_types(miss_type)

            with torch.no_grad():
                epoch_metric_recorder = self.mm_model.metric_recorder.clone()
                for batch in tqdm(
                    dataloader,
                    desc=f"Client {self.client_id}: Evaluation Batch",
                    ascii=True,
                ):
                    self.mm_model.eval()

                    if self.cmam:
                        results = self.cmam.incongruent_evaluate(
                            batch=batch,
                            labels=batch.get("label"),
                            mm_model=self.mm_model,
                            device=self.device,
                            criterion=self.criterion,
                        )
                    else:
                        results = self.mm_model.evaluate(
                            batch=batch, criterion=self.criterion, device=self.device
                        )
                    epoch_metric_recorder.update_from_dict(results)

            test_metrics = epoch_metric_recorder.get_average_metrics()
            metrics[miss_type] = test_metrics
            display_validation_metrics(test_metrics)

            # Save metrics
            metrics_path = os.path.join(
                self.config.logging.metrics_path.format_map(
                    SafeDict(client_id=self.client_id, round=self.current_round)
                ),
                f"test_metrics_round_{self.current_round}_{miss_type}.json",
            )
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            for k, v in test_metrics.items():
                if isinstance(v, ndarray):
                    test_metrics[k] = v.tolist()

            with open(metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=4)
            self.print_fn(f"Saved test metrics to {metrics_path}")

        dataloader.dataset.set_selected_missing_types(original_miss_types)

        mm_params = self.mm_model.get_relevant_parameters(self.modality)
        if self.cmam and self.modality != Modality.MULTIMODAL:
            cmam_params = {k: v for k, v in self.cmam.state_dict().items()}
        else:
            cmam_params = None

        return FederatedResult(
            client_info=self.status(),
            model_params=(mm_params, {self.modality: cmam_params}),
            metrics=metrics,
            available_modality=[self.modality],
            metadata={},
        )

    def status(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "best_model_info": self.best_model_info,
            "modality": str(self.modality),
        }

    def __str__(self) -> str:
        return f"FederatedIncongruentClient(client_id={self.client_id}, current_round={self.current_round})"
