from collections import defaultdict
from copy import deepcopy
import json
import os
from typing import Literal, Optional, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

from config.federated_incongruent_config import FederatedIncongruentServerConfig
from federated import FederatedResult
from models import CMAMProtocol, MultimodalModelProtocol
from utils import (
    SafeDict,
    display_training_metrics,
    display_validation_metrics,
    get_logger,
)
from modalities import Modality
from federated.incongruent_client import FederatedIncongruentClient

logger = get_logger()


class FederatedIncongruentServer:
    def __init__(
        self,
        client_ids: list[int],
        device: torch.device,
        mm_model: MultimodalModelProtocol,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        test_dataloader: DataLoader,
        config: FederatedIncongruentServerConfig,
        mm_optimizer: Optimizer,
        mm_criterion: Module,
        *,
        cmams: Optional[dict[Modality, CMAMProtocol]] = None,
        cmam_optimizers: Optional[dict[Modality, Optimizer]] = None,
        cmam_criterion: Module = None,
        aggregation_strategy: Literal["fedavg", "fedprox"] = "fedavg",
        selection_strategy: Literal["all", "percent_dropout"] = "all",
        dropout_percent: float = 0.1,
    ):
        FederatedIncongruentServer.assert_preconditions(
            client_ids=client_ids,
            mm_model=mm_model,
            cmams=cmams,
            aggregation_strategy=aggregation_strategy,
            selection_strategy=selection_strategy,
            dropout_percent=dropout_percent,
        )

        self.client_ids = client_ids
        self.device = device
        self.mm_model = mm_model
        self.mm_optimizer = mm_optimizer
        self.mm_criterion = mm_criterion

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.config = config
        self.cmams = cmams
        self.cmam_optimizers = cmam_optimizers
        self.cmam_criterion = cmam_criterion

        self.aggregation_strategy = aggregation_strategy
        self.selection_strategy = selection_strategy
        self.dropout_percent = dropout_percent

        self.current_round = 0
        self.best_global_mm_metrics = None
        self.best_global_mm_state = None
        self.best_global_cmams_metrics = {} if cmams else None
        self.best_global_cmams_state = {} if cmams else None
        self.best_round = 0

    def train_round(
        self,
        mm_epochs: int,
        clients: List[FederatedIncongruentClient],
        *,
        print_fn: callable = print,
    ) -> List[FederatedResult]:

        self.current_round += 1
        print_fn(f"Starting round {self.current_round}")
        logger.info(f"Starting round {self.current_round}")
        # Step 2: Train global multimodal model and C-MAMs (if applicable)

        if self.current_round == 1:
            mm_epochs = 5

        self.train_global_models(mm_epochs, print_fn=print_fn)
        # Step 1: Evaluate global model (pre-aggregation)
        print_fn(
            f"Evaluating global models before training clients for round {self.current_round}"
        )
        pre_aggregation_results = self.evaluate_global_model(
            dataloader=self.validation_dataloader, clients=clients, print_fn=print_fn
        )

        # Step 3: Train clients
        client_results = []
        for client in clients:
            print_fn(f"Training client {client.client_id}")
            client_result = client.train_round()
            client_results.append(client_result)

        # Step 4: Aggregate model parameters from clients
        aggregated_params = self.aggregate(client_results)
        self.update_global_models(aggregated_params)

        # Step 5: Evaluate global model (post-aggregation)
        global_mm_eval_results = self.evaluate_global_model(
            dataloader=self.validation_dataloader, clients=clients, print_fn=print_fn
        )

        # save the model
        model_output_path = (
            self.config.server_config.cls_logging.model_output_path.format_map(
                SafeDict(round=self.current_round)
            )
        )

        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(self.mm_model.state_dict(), model_output_path)

        if self.cmams is not None:
            for cmam_name, cmam in self.cmams.items():
                cmam_output_path = self.config.server_config.cmam_logging[
                    cmam_name
                ].model_output_path.format_map(SafeDict(round=self.current_round))
                os.makedirs(os.path.dirname(cmam_output_path), exist_ok=True)
                torch.save(cmam.state_dict(), cmam_output_path)

        for result in global_mm_eval_results:
            display_validation_metrics(result.metrics, console=None)

        # Track the best round and metrics
        self._update_best_models(global_mm_eval_results)

        # Step 6: Evaluate global model on the test set
        print_fn(
            f"Evaluating global models on the test set for round {self.current_round}"
        )
        test_results = self.evaluate_global_model(
            dataloader=self.test_dataloader, clients=clients, print_fn=print_fn
        )

        print_fn(f"Round {self.current_round} complete")
        return global_mm_eval_results

    def evaluate_global_model(
        self,
        dataloader: DataLoader,
        clients: List[FederatedIncongruentClient],
        *,
        print_fn: callable = print,
    ) -> List[FederatedResult]:
        print_fn(f"Evaluating global model at round {self.current_round}")
        logger.info(f"Evaluating global model at round {self.current_round}")

        global_results = []

        missing_types = list(
            self.train_dataloader.dataset.base_dataset.MODALITY_LOOKUP.keys()
        )

        for miss_type in missing_types:
            dataloader.dataset.set_selected_missing_types(miss_type)
            for batch in tqdm(dataloader, desc="Global Model Evaluation", ascii=True):
                # Evaluate the multimodal model (MM)
                self.mm_model.eval()
                with torch.no_grad():
                    mm_eval_results = self.mm_model.evaluate(
                        batch, device=self.device, criterion=self.mm_criterion
                    )
                    display_validation_metrics(mm_eval_results, console=None)

                    for k, v in mm_eval_results.items():
                        if isinstance(v, np.ndarray):
                            mm_eval_results[k] = v.tolist()

                    metrics_path = self.config.server_config.cls_logging.metrics_path
                    save_metric = self.config.server_config.cls_logging.save_metric

                    p = metrics_path.format_map(
                        SafeDict(round=self.current_round, save_metric=save_metric)
                    )
                    p = os.path.join(p, f"{miss_type}.json")
                    os.makedirs(os.path.dirname(p), exist_ok=True)

                    with open(p, "w") as f:
                        json_str = json.dumps(mm_eval_results)
                        f.write(json_str)

                    print_fn(f"MM-Model Metrics saved to {p}")
                    global_results.append(FederatedResult(metrics=mm_eval_results))

                # Evaluate C-MAMs if available
                if self.cmams is not None:
                    for cmam_name, cmam in self.cmams.items():
                        cmam.eval()
                        with torch.no_grad():
                            cmam_eval_results = cmam.evaluate(
                                batch=batch,
                                labels=batch.get("label"),
                                device=self.device,
                                cmam_criterion=self.cmam_criterion,
                                trained_model=self.mm_model,
                            )
                            display_validation_metrics(cmam_eval_results, console=None)

                            for k, v in cmam_eval_results.items():
                                if isinstance(v, np.ndarray):
                                    cmam_eval_results[k] = v.tolist()

                            metrics_path = self.config.server_config.cmam_logging[
                                cmam_name
                            ].metrics_path
                            save_metric = self.config.server_config.cmam_logging[
                                cmam_name
                            ].save_metric

                            p = metrics_path.format_map(
                                SafeDict(
                                    round=self.current_round, save_metric=save_metric
                                )
                            )
                            p = os.path.join(p, f"{miss_type}.json")
                            os.makedirs(os.path.dirname(p), exist_ok=True)

                            print_fn(f"C-MAM {cmam_name} Metrics saved to {p}")
                            with open(p, "w") as f:
                                json_str = json.dumps(cmam_eval_results)
                                f.write(json_str)

                            global_results.append(
                                FederatedResult(metrics=cmam_eval_results)
                            )

        return global_results

    def train_global_models(self, mm_epochs: int, *, print_fn: callable = print):
        print_fn(f"Training global multimodal model for {mm_epochs} epochs")
        logger.info(f"Training global multimodal model for {mm_epochs} epochs")

        self.mm_model.train()
        self.mm_model.to(self.device)
        mm_epoch_metric_recorder = self.mm_model.metric_recorder.clone()

        for param in self.mm_model.parameters():
            param.requires_grad = True

        for epoch in range(1, mm_epochs + 1):
            self.mm_model.train()
            self.mm_model.to(self.device)
            self.mm_model.metric_recorder.reset()
            for batch in tqdm(
                self.train_dataloader,
                desc=f"Global MM Model Training Epoch {epoch}",
                ascii=True,
            ):
                mm_train_results = self.mm_model.train_step(
                    batch=batch,
                    optimizer=self.mm_optimizer,
                    criterion=self.mm_criterion,
                    device=self.device,
                )
                mm_epoch_metric_recorder.update_from_dict(mm_train_results)

            train_metrics = mm_epoch_metric_recorder.get_average_metrics()
            display_training_metrics(train_metrics)
            print_fn(f"Global MM Model - Train Metrics for epoch {epoch}")
            logger.info(f"Global MM Model - Train Metrics for epoch {epoch}")
            logger.info(train_metrics)

        if self.cmams is not None:
            for cmam_name, cmam in self.cmams.items():
                print_fn(f"Training C-MAM {cmam_name} for {mm_epochs} epochs")
                logger.info(f"Training C-MAM {cmam_name} for {mm_epochs} epochs")

                cmam.train()
                cmam.to(self.device)
                self.mm_model.eval()
                cmam_epoch_metric_recorder = cmam.metric_recorder.clone()

                for epoch in range(1, mm_epochs + 1):
                    cmam.train()
                    cmam.to(self.device)
                    cmam.metric_recorder.reset()
                    for batch in tqdm(
                        self.train_dataloader,
                        desc=f"C-MAM {cmam_name} Training Epoch {epoch}",
                        ascii=True,
                    ):
                        cmam_train_results = cmam.train_step(
                            batch=batch,
                            labels=batch.get("label"),
                            optimizer=self.cmam_optimizers[cmam_name],
                            cmam_criterion=self.cmam_criterion,
                            device=self.device,
                            trained_model=self.mm_model,
                        )
                        cmam_epoch_metric_recorder.update_from_dict(cmam_train_results)

                    train_metrics = cmam_epoch_metric_recorder.get_average_metrics()
                    display_training_metrics(train_metrics)
                    print_fn(f"C-MAM {cmam_name} - Train Metrics for epoch {epoch}")
                    logger.info(f"C-MAM {cmam_name} - Train Metrics for epoch {epoch}")
                    logger.info(train_metrics)

    def aggregate(
        self, client_results: List[FederatedResult]
    ) -> Tuple[dict[str, torch.Tensor], Optional[dict[str, dict[str, torch.Tensor]]]]:
        aggregated_mm_params = {}
        count_mm_updates = defaultdict(int)

        aggregated_cmam_params = {} if self.cmams else None
        count_cmam_updates = {} if self.cmams else None

        # Initialize aggregated parameters with zero tensors of appropriate shape
        for param_name, param_value in self.mm_model.state_dict().items():
            aggregated_mm_params[param_name] = torch.zeros_like(param_value)

        if self.cmams:
            for cmam_name, cmam in self.cmams.items():
                aggregated_cmam_params[cmam_name] = {}
                count_cmam_updates[cmam_name] = defaultdict(int)
                for param_name, param_value in cmam.state_dict().items():
                    aggregated_cmam_params[cmam_name][param_name] = torch.zeros_like(
                        param_value
                    )

        for result in client_results:
            client_mm_params, client_cmam_params = result.model_params

            # Aggregate multimodal model parameters
            for param_name, param_value in client_mm_params.items():
                if param_name in aggregated_mm_params:
                    aggregated_mm_params[param_name] += param_value
                    count_mm_updates[param_name] += 1

            # Aggregate C-MAM parameters if applicable
            if client_cmam_params and self.cmams:
                for cmam_name, cmam_param_values in client_cmam_params.items():
                    for param_name, param_value in cmam_param_values.items():
                        if param_name in aggregated_cmam_params[cmam_name]:
                            aggregated_cmam_params[cmam_name][param_name] += param_value
                            count_cmam_updates[cmam_name][param_name] += 1

        # Calculate average for multimodal model parameters
        for param_name in aggregated_mm_params:
            if count_mm_updates[param_name] > 0:
                if aggregated_mm_params[param_name].dtype == torch.float32:
                    aggregated_mm_params[param_name] /= count_mm_updates[param_name]
                else:
                    aggregated_mm_params[param_name] = (
                        aggregated_mm_params[param_name].float()
                        / count_mm_updates[param_name]
                    ).to(aggregated_mm_params[param_name].dtype)

        # Calculate average for C-MAM parameters if applicable
        if self.cmams:
            for cmam_name in aggregated_cmam_params:
                for param_name in aggregated_cmam_params[cmam_name]:
                    if count_cmam_updates[cmam_name][param_name] > 0:

                        if (
                            aggregated_cmam_params[cmam_name][param_name].dtype
                            == torch.float32
                        ):
                            aggregated_cmam_params[cmam_name][
                                param_name
                            ] /= count_cmam_updates[cmam_name][param_name]
                        else:
                            aggregated_cmam_params[cmam_name][param_name] = (
                                aggregated_cmam_params[cmam_name][param_name].float()
                                / count_cmam_updates[cmam_name][param_name]
                            ).to(aggregated_cmam_params[cmam_name][param_name].dtype)

        return aggregated_mm_params, aggregated_cmam_params

    def update_global_models(
        self,
        aggregated_params: Tuple[
            dict[str, torch.Tensor], Optional[dict[str, dict[str, torch.Tensor]]]
        ],
    ) -> None:
        mm_model_params = aggregated_params[0]
        self.mm_model.load_state_dict(mm_model_params)
        logger.debug("Updated MM model with aggregated parameters")

        if self.cmams is not None:
            cmam_params = aggregated_params[1]
            for cmam_name, cmam in self.cmams.items():
                if cmam_params and cmam_name in cmam_params:
                    cmam.load_state_dict(cmam_params[cmam_name])
                    logger.debug(
                        f"Updated CMAM model {cmam_name} with aggregated parameters"
                    )

    def distribute_model(
        self,
    ) -> Tuple[dict[str, torch.Tensor], Optional[dict[str, dict[str, torch.Tensor]]]]:
        mm_params = deepcopy(self.mm_model.state_dict())
        cmam_params = None

        if self.cmams is not None:
            cmam_params = {}
            for cmam_name, cmam in self.cmams.items():
                cmam_params[cmam_name] = deepcopy(cmam.state_dict())

        return (mm_params, cmam_params)

    def _select_all_clients(self) -> List[int]:
        return [i for i in range(len(self.client_ids))]

    def _select_clients_with_dropout(self, dropout_percent: float) -> List[int]:
        n_clients = int(len(self.client_ids) * (1 - dropout_percent))
        return np.random.choice(len(self.client_ids), n_clients, replace=False).tolist()

    def run_federated_learning(
        self,
        num_rounds: int,
        clients: List[FederatedIncongruentClient],
        mm_epochs: int,
        *,
        print_fn: callable = print,
    ) -> dict:
        results = {
            "round_results": [],
            "final_test_results": None,
            "best_round": 0,
            "best_global_performance": None,
        }
        for round_idx in range(1, num_rounds + 1):
            print_fn(
                f"\n--- Starting Federated Learning Round {round_idx}/{num_rounds} ---\n"
            )
            logger.info(
                f"\n--- Starting Federated Learning Round {round_idx}/{num_rounds} ---\n"
            )

            round_results = self.train_round(mm_epochs, clients, print_fn=print_fn)
            results["round_results"].append(round_results)

            # Track the best metrics and models
            self._update_best_models(round_results)

        # Evaluate on the global test set after all rounds are complete
        print_fn("\n--- Evaluating Final Global Model on Test Set ---\n")
        logger.info("\n--- Evaluating Final Global Model on Test Set ---\n")
        final_test_results = self.evaluate_global_model(
            dataloader=self.test_dataloader, clients=clients, print_fn=print_fn
        )
        # self.log_evaluation_results(final_test_results, prefix="Final Test Evaluation")
        results["final_test_results"] = {
            "metrics": [result.metrics for result in final_test_results]
        }

        results["best_round"] = self.best_round
        results["best_global_performance"] = self.best_global_mm_metrics

        print_fn("\n--- Federated Learning Complete ---\n")
        logger.info("\n--- Federated Learning Complete ---\n")

        return results

    def _update_best_models(self, global_mm_eval_results: List[FederatedResult]):
        for result in global_mm_eval_results:
            metrics = result.metrics
            save_metric = self.config.server_config.cls_logging.save_metric
            current_metric_value = metrics.get(save_metric)

            if current_metric_value is None:
                continue

            # Determine if current metrics are the best so far
            is_better = (
                save_metric == "loss"
                and (
                    self.best_global_mm_metrics is None
                    or current_metric_value < self.best_global_mm_metrics[save_metric]
                )
            ) or (
                save_metric != "loss"
                and (
                    self.best_global_mm_metrics is None
                    or current_metric_value > self.best_global_mm_metrics[save_metric]
                )
            )

            if is_better:
                self.best_global_mm_metrics = metrics
                self.best_global_mm_state = deepcopy(self.mm_model.state_dict())
                if self.cmams is not None:
                    self.best_global_cmams_metrics = {
                        cmam_name: deepcopy(cmam.metric_recorder.get_average_metrics())
                        for cmam_name, cmam in self.cmams.items()
                    }
                    self.best_global_cmams_state = {
                        cmam_name: deepcopy(cmam.state_dict())
                        for cmam_name, cmam in self.cmams.items()
                    }
                self.best_round = self.current_round

    @staticmethod
    def assert_preconditions(
        client_ids: list[int],
        mm_model: MultimodalModelProtocol,
        cmams: dict[str, CMAMProtocol] | None,
        aggregation_strategy: Literal["fedavg", "fedprox"],
        selection_strategy: Literal["all", "percent_dropout"],
        dropout_percent: float,
    ) -> None:
        assert len(client_ids) > 0, "Number of clients must be greater than 0"

        assert (
            hasattr(mm_model, "metric_recorder")
            and mm_model.metric_recorder is not None
        ), "Model must have its metric recorder set and not None"

        if cmams is not None:
            for c in cmams.values():
                assert (
                    hasattr(c, "metric_recorder") and c.metric_recorder is not None
                ), "CMAM model must have its metric recorder set and not None"

        assert aggregation_strategy in [
            "fedavg",
            "fedprox",
        ], "aggregation_strategy must be 'fedavg' or 'fedprox'"
        assert selection_strategy in [
            "all",
            "percent_dropout",
        ], "selection_strategy must be 'all' or 'percent_dropout'"
        assert 0 <= dropout_percent <= 1, "dropout_percent must be between 0 and 1"
