import json
import os
from typing import Any, Dict, Optional
from copy import deepcopy
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.federated_cmam_config import FederatedCMAMClientConfig
from federated import FederatedResult
from models import CMAMProtocol, MultimodalModelProtocol
from utils import (
    display_training_metrics,
    display_validation_metrics,
    get_logger,
)

logger = get_logger()


class FederatedCMAMClient:
    def __init__(
        self,
        client_id: str | int,
        model: MultimodalModelProtocol,
        cmam: CMAMProtocol,
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        fed_config: FederatedCMAMClientConfig,
        total_epochs: int,
        model_output_dir: str,
        metrics_output_dir: str,
        logging_output_dir: str,
        run_id: int,
        scheduler: Optional[_LRScheduler] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        print_fn: callable = print,
    ):
        assert not all(
            [train_loader is None, test_loader is None]
        ), "At least one of train_loader or test_loader should be provided."

        if val_loader is not None:
            assert (
                train_loader is not None
            ), "train_loader should be provided if val_loader is provided."

            logger.info(
                f"Client {client_id} has {len(train_loader)} training samples and {len(val_loader)} validation samples."
            )

        if test_loader is not None:
            logger.info(f"Client {client_id} has {len(test_loader)} test samples.")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.model = model
        self.cmam = cmam
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.fed_config = fed_config
        self.lr_scheduler = scheduler

        self.current_round = 0
        self.print_fn = print_fn
        self.current_epoch = 1
        self.total_epochs = total_epochs

        self.best_model_path = None
        self.best_model_score = (
            float("inf") if fed_config.target_metric == "loss" else float("-inf")
        )
        self.best_model_epoch = 0
        self.best_model_round = 0
        self.run_id = run_id

        self.model_output_dir = model_output_dir.format(
            run_id=run_id, client_id=client_id, save_metric=fed_config.target_metric
        )
        self.metrics_output_dir = metrics_output_dir.format(
            run_id=run_id, client_id=client_id
        )
        self.logging_output_dir = logging_output_dir.format(
            run_id=run_id, client_id=client_id
        )

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        ## return a deep copy of the state_dict
        return deepcopy(self.cmam.state_dict())

    def set_model_parameters(self, model_params: Dict[str, torch.Tensor]) -> None:
        self.cmam.load_state_dict(deepcopy(model_params))

    def status(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
            "best_model_epoch": self.best_model_epoch,
            "best_model_round": self.best_model_round,
        }

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = self.cmam.train_step(
            batch=batch,
            labels=batch["label"],
            optimizer=self.optimizer,
            cmam_criterion=self.criterion,
            device=self.device,
            trained_model=self.model,
        )
        return output

    def _evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = self.cmam.evaluate(
            batch=batch,
            labels=batch["label"],
            cmam_criterion=self.criterion,
            device=self.device,
            trained_model=self.model,
        )
        return output

    def evaluate_round(self, dataloader) -> FederatedResult:
        self.print_fn(
            f"Evaluating the best model on the test set on Client: {self.client_id}."
        )
        logger.info(
            f"Evaluating the best model on the test set on Client: {self.client_id}."
        )

        self.cmam.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        self.cmam.metric_recorder.reset()

        self.print_fn(f"Loaded best model from {self.best_model_path}")
        logger.info(f"Loaded best model from {self.best_model_path}")
        self.model.eval()
        self.cmam.eval()
        for batch in tqdm(dataloader, ascii=True, total=len(dataloader)):
            results = self._evaluate_step(batch)
            self.cmam.metric_recorder.update_from_dict(results)

        test_cm = self.cmam.metric_recorder.get("ConfusionMatrix", default=None)
        if test_cm is not None:
            test_cm = np.sum(test_cm, axis=0)
            tqdm.write(str(test_cm))
            del self.cmam.metric_recorder.results["ConfusionMatrix"]

        test_metrics = self.cmam.metric_recorder.get_average_metrics()

        self.print_fn("Test Metrics")
        display_validation_metrics(
            metrics=test_metrics,
            console=None,
        )

        logger.info(f"Test Metrics: {test_metrics}")

        self.print_fn(
            f"Evaluation of the best model on the test set completed on Client: {self.client_id}."
        )
        logger.info(
            f"Evaluation of the best model on the test set completed on Client: {self.client_id}."
        )

        with open(
            os.path.join(
                self.metrics_output_dir, f"test_metrics_round_{self.current_round}.json"
            ),
            "w",
        ) as f:
            json_str = json.dumps(test_metrics, indent=4)
            f.write(json_str)
        logger.info(
            f"Saved test metrics to {os.path.join(self.metrics_output_dir, f'test_metrics_round_{self.current_round}.json')}"
        )

        return FederatedResult(
            client_info=self.status(),
            model_params=self.get_model_parameters(),
            metrics=test_metrics,
            metadata={},
        )

    def train_round(self):
        self.current_epoch = 1
        self.current_round += 1
        self.cmam.flatten_parameters()
        self.cmam.train()
        self.print_fn(
            f"Training round {self.current_round} started on Client: {self.client_id}."
        )
        logger.info(
            f"Training round {self.current_round} started on Client: {self.client_id}."
        )
        self.cmam.metric_recorder.reset()
        epoch_metric_recorder = self.cmam.metric_recorder.clone()
        for epoch in range(1, self.total_epochs + 1):
            self.cmam.metric_recorder.reset()
            epoch_metric_recorder.reset()
            self.current_epoch = epoch
            self.print_fn(
                f"Epoch {epoch}/{self.total_epochs} started on Client: {self.client_id}."
            )
            logger.info(
                f"Epoch {epoch}/{self.total_epochs} started on Client: {self.client_id}."
            )
            for batch in tqdm(
                self.train_loader, ascii=True, total=len(self.train_loader)
            ):
                results = self._train_step(batch)
                epoch_metric_recorder.update_from_dict(results)

            logger.info(f"Train Metrics: {epoch_metric_recorder.results}")

            train_metrics = epoch_metric_recorder.get_average_metrics()
            self.print_fn("Train Metrics")
            display_training_metrics(
                metrics=train_metrics,
                console=None,
            )

            self.cmam.metric_recorder.reset()
            epoch_metric_recorder = self.cmam.metric_recorder.clone()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.print_fn(
                    f"Learning rate: {self.optimizer.param_groups[0]['lr']:7f}"
                )
                logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7f}")

            ##############################################################################
            logger.info(
                f"Epoch {epoch}/{self.total_epochs} completed on Client: {self.client_id}."
            )
            self.model.eval()

            self.cmam.eval()
            for batch in tqdm(self.val_loader, ascii=True, total=len(self.val_loader)):
                results = self._evaluate_step(batch)
                epoch_metric_recorder.update_from_dict(results)

            val_metrics = epoch_metric_recorder.get_average_metrics(
                save_to=os.path.join(
                    self.metrics_output_dir,
                    f"{self.current_round}/validation",
                ),
                epoch=self.current_epoch,
            )
            self.print_fn("Validation Metrics")
            display_validation_metrics(
                metrics=val_metrics,
                console=None,
            )

            logger.info(f"Validation Metrics: {val_metrics}")

            ## update best model
            if self.fed_config.target_metric == "loss":
                if val_metrics["loss"] < self.best_model_score:
                    self.best_model_score = val_metrics["loss"]
                    self.best_model_epoch = epoch
                    self.best_model_round = self.current_round
                    self.best_model_path = os.path.join(
                        self.model_output_dir,
                        str(self.current_round),
                        "best.pth",
                    )
                    os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
                    torch.save(self.cmam.state_dict(), self.best_model_path)
            else:
                if val_metrics[self.fed_config.target_metric] > self.best_model_score:
                    self.best_model_score = val_metrics[self.fed_config.target_metric]
                    self.best_model_epoch = epoch
                    self.best_model_round = self.current_round
                    self.best_model_path = os.path.join(
                        self.model_output_dir,
                        str(self.current_round),
                        "best.pth",
                    )
                    os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
                    torch.save(self.cmam.state_dict(), self.best_model_path)

                    self.print_fn(
                        f"Best model saved at {self.best_model_path} with score: {self.best_model_score}"
                    )
                    logger.info(
                        f"Best model saved at {self.best_model_path} with score: {self.best_model_score}"
                    )

            self.print_fn(f"Epoch {epoch}/{self.total_epochs} completed.")
            logger.info(f"Epoch {epoch}/{self.total_epochs} completed.")
        self.print_fn(
            f"Training round {self.current_round} completed on Client: {self.client_id}."
        )
        logger.info(
            f"Training round {self.current_round} completed on Client: {self.client_id}."
        )

        if self.test_loader is not None:
            logger.info(
                f"Starting evaluation of the best model on the test set on Client: {self.client_id}."
            )
            return self.evaluate_round(dataloader=self.test_loader)
        logger.info(
            f"Starting evaluation of the best model on the validation set on Client: {self.client_id}."
        )
        ## load the best model anyways and return the validation metrics
        self.cmam.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        return self.evaluate_round(dataloader=self.val_loader)

    def __str__(self) -> str:
        return f"FederatedMultimodalClient(client_id={self.client_id}, current_round={self.current_round}, total_rounds={self.total_rounds}, current_epoch={self.current_epoch}, total_epochs={self.total_epochs}, best_model_path={self.best_model_path}, best_model_score={self.best_model_score}, best_model_epoch={self.best_model_epoch}, best_model_round={self.best_model_round})"
