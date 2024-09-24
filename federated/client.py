import json
import os
from subprocess import Popen
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.federated_config import FederatedClientConfig
from federated import FederatedResult
from models import MultimodalModelProtocol
from utils import print_all_metrics_tables


class FederatedMultimodalClient:
    def __init__(
        self,
        client_id: str | int,
        model: MultimodalModelProtocol,
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        fed_config: FederatedClientConfig,
        run_id: int = -1,
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.fed_config = fed_config
        self.lr_scheduler = scheduler

        self.current_round = 0
        self.total_rounds = fed_config.n_rounds
        self.print_fn = print_fn
        self.current_epoch = 1
        self.total_epochs = fed_config.epochs

        self.best_model_path = None
        self.best_model_score = (
            float("inf") if fed_config.target_metric == "loss" else float("-inf")
        )
        self.best_model_epoch = 0
        self.best_model_round = 0
        self.run_id = run_id

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

    def status(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
            "best_model_epoch": self.best_model_epoch,
            "best_model_round": self.best_model_round,
        }

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = self.model.train_step(
            batch=batch,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )
        return output

    def _evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = self.model.evaluate(
            batch=batch,
            criterion=self.criterion,
            device=self.device,
        )
        return output

    def evaluate_round(self, dataloader) -> FederatedResult:
        self.print_fn(
            f"Evaluating the best model on the test set on Client: {self.client_id}."
        )

        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        self.model.eval()
        for batch in tqdm(dataloader, ascii=True, total=len(dataloader)):
            results = self._evaluate_step(batch)
            self.model.metric_recorder.update_from_dict(results)

        test_cm = self.model.metric_recorder.get("ConfusionMatrix", default=None)
        if test_cm is not None:
            test_cm = np.sum(test_cm, axis=0)
            tqdm.write(str(test_cm))
            del self.model.metric_recorder.results["ConfusionMatrix"]

        test_metrics = self.model.metric_recorder.get_average_metrics()

        self.print_fn("Test Metrics")
        print_all_metrics_tables(
            metrics=test_metrics,
            console=None,
            max_cols_per_row=15,
        )

        self.print_fn(
            f"Evaluation of the best model on the test set completed on Client: {self.client_id}."
        )

        with open(
            os.path.join(
                self.fed_config.output_dir,
                f"client_{self.client_id}/{self.best_model_round}/",
                "test_metrics.json",
            ),
            "w",
        ) as f:
            json_str = json.dumps(test_metrics, indent=4)
            f.write(json_str)

        return FederatedResult(
            client_info=self.status(),
            model_params=self.get_model_parameters(),
            metrics=test_metrics,
            metadata={},
        )

    def train_round(self):
        self.current_epoch = 1
        self.current_round += 1
        self.model.train()
        self.print_fn(
            f"Training round {self.current_round} started on Client: {self.client_id}."
        )
        self.model.metric_recorder.reset()
        epoch_metric_recorder = self.model.metric_recorder.clone()

        for epoch in range(1, self.total_epochs + 1):
            epoch_metric_recorder.reset()
            self.current_epoch = epoch
            self.print_fn(
                f"Epoch {epoch}/{self.total_epochs} started on Client: {self.client_id}."
            )
            for batch in tqdm(
                self.train_loader, ascii=True, total=len(self.train_loader)
            ):
                results = self._train_step(batch)
                epoch_metric_recorder.update_from_dict(results)
            train_cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
            if train_cm is not None:
                train_cm = np.sum(train_cm, axis=0)
                tqdm.write(str(train_cm))
                del epoch_metric_recorder.results["ConfusionMatrix"]

            train_metrics = epoch_metric_recorder.get_average_metrics()
            self.print_fn("Train Metrics")
            print_all_metrics_tables(
                metrics=train_metrics,
                console=None,
                max_cols_per_row=15,
            )

            epoch_metric_recorder.reset()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.print_fn(
                    f"Learning rate: {self.optimizer.param_groups[0]['lr']:7f}"
                )

            ##############################################################################

            self.model.eval()
            for batch in tqdm(self.val_loader, ascii=True, total=len(self.val_loader)):
                results = self._evaluate_step(batch)
                epoch_metric_recorder.update_from_dict(results)

            val_cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
            if val_cm is not None:
                val_cm = np.sum(val_cm, axis=0)
                tqdm.write(str(val_cm))
                del epoch_metric_recorder.results["ConfusionMatrix"]

            val_metrics = epoch_metric_recorder.get_average_metrics(
                save_to=os.path.join(
                    self.fed_config.output_dir,
                    f"client_{self.client_id}/{self.current_round}/",
                    epoch=self.current_epoch,
                )
            )
            self.print_fn("Validation Metrics")
            print_all_metrics_tables(
                metrics=val_metrics,
                console=None,
                max_cols_per_row=15,
            )

            ## update best model
            if self.fed_config.target_metric == "loss":
                if val_metrics["loss"] < self.best_model_score:
                    self.best_model_score = val_metrics["loss"]
                    self.best_model_epoch = epoch
                    self.best_model_round = self.current_round
                    self.best_model_path = os.path.join(
                        self.fed_config.output_dir,
                        f"client_{self.client_id}/{self.current_round}/",
                        "best.pth",
                    )
                    os.makedirs(
                        os.path.join(
                            self.fed_config.output_dir,
                            f"client_{self.client_id}/{self.current_round}/",
                        ),
                        exist_ok=True,
                    )
                    torch.save(self.model.state_dict(), self.best_model_path)
            else:
                if val_metrics[self.fed_config.target_metric] > self.best_model_score:
                    self.best_model_score = val_metrics[self.fed_config.target_metric]
                    self.best_model_epoch = epoch
                    self.best_model_round = self.current_round
                    self.best_model_path = os.path.join(
                        self.fed_config.output_dir,
                        f"client_{self.client_id}/{self.current_round}/",
                        "best.pth",
                    )
                    self.print_fn(
                        f"Best model saved at {self.best_model_path} with score: {self.best_model_score}"
                    )
                    os.makedirs(
                        os.path.join(
                            self.fed_config.output_dir,
                            f"client_{self.client_id}/{self.current_round}/",
                        ),
                        exist_ok=True,
                    )
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.print_fn(
                        f"Best model saved at {self.best_model_path} with score: {self.best_model_score}"
                    )
            self.print_fn(f"Epoch {epoch}/{self.total_epochs} completed.")
        self.print_fn(
            f"Training round {self.current_round} completed on Client: {self.client_id}."
        )

        if self.fed_config.do_validation_visualization:
            _vis_proc = Popen(
                [
                    "python",
                    "visualisation/validation_plotting.py",
                    "--root_dir",
                    self.fed_config.output_dir,
                    f"client_{self.client_id}/{self.current_round}/",
                    "--save_path",
                    self.fed_config.output_dir,
                    f"client_{self.client_id}/{self.current_round}/validation_plots",
                ]
            )

        if self.test_loader is not None:
            return self.evaluate_round(dataloader=self.test_loader)

        ## load the best model anyways and return the validation metrics
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        return self.evaluate_round(dataloader=self.val_loader)
