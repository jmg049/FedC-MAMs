from collections import defaultdict
from copy import deepcopy
import json
import os
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Any, List, Dict, Literal

from tqdm import tqdm
from config.federated_cmam_config import FederatedCMAMConfig
from federated import FederatedResult
from federated.client import FederatedMultimodalClient
from models import CMAMProtocol, MultimodalModelProtocol
from utils import SafeDict, display_training_metrics, display_validation_metrics

from utils import get_logger

logger = get_logger()


class FederatedCongruentCMAMServer:
    def __init__(
        self,
        model: MultimodalModelProtocol,
        cmam: CMAMProtocol,
        global_train_data: DataLoader,
        global_val_data: DataLoader,
        global_test_data: DataLoader,
        num_clients: int,
        device: torch.device,
        config: FederatedCMAMConfig,
        aggregation_strategy: Literal["fedavg"] = "fedavg",
        selection_strategy: Literal["all", "random"] = "all",
    ) -> "FederatedCongruentCMAMServer":
        assert (
            hasattr(model, "metric_recorder") and model.metric_recorder is not None
        ), "Model must have its metric recorder set and not None"
        self.global_model = model
        self.global_cmam = cmam
        self.aggregation_strategy = aggregation_strategy
        self.num_clients = num_clients
        self.current_round = 0
        self.client_selection_strategy = selection_strategy

        self.device = device

        self.global_train_data = global_train_data
        self.global_val_data = global_val_data
        self.global_test_data = global_test_data
        self.config = config

    def select_clients(self) -> List[int]:
        # assume all clients are selected for now
        return list(range(self.num_clients))

    def distribute_model(self) -> Dict[str, torch.Tensor]:
        logger.debug("Distributing global model to clients")
        return deepcopy(self.global_cmam.state_dict())

    def fed_avg(self, client_results: List[FederatedResult]) -> Dict[str, torch.Tensor]:
        """
        Implement the Federated Averaging (FedAvg) algorithm to aggregate client models.

        Args:
            client_results (List[FederatedResult]): List of client results containing model parameters.

        Returns:
            Dict[str, torch.Tensor]: Aggregated global model parameters.
        """
        # Initialize a dictionary to store the sum of parameters and their counts
        aggregated_params = {}
        param_counts = {}

        # Iterate through all client results
        for result in client_results:
            client_params = result.model_params

            for param_name, param_value in client_params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = torch.zeros_like(param_value)
                    param_counts[param_name] = 0

                # Sum up the parameters
                aggregated_params[param_name] += param_value
                param_counts[param_name] += 1

        # Calculate the average for each parameter
        for param_name in aggregated_params:
            if param_counts[param_name] > 0:
                if aggregated_params[param_name].dtype == torch.float32:
                    aggregated_params[param_name] /= param_counts[param_name]
                else:
                    aggregated_params[param_name] = (
                        aggregated_params[param_name] // param_counts[param_name]
                    )

        return aggregated_params

    def fed_prox(
        self, client_results: List[FederatedResult]
    ) -> Dict[str, torch.Tensor]:
        aggregated_params = {}
        return aggregated_params

    def aggregate_models(
        self, client_results: List[FederatedResult]
    ) -> Dict[str, torch.Tensor]:
        match self.aggregation_strategy:
            case "fedavg":
                logger.debug("Aggregating models using FedAvg")
                return self.fed_avg(client_results)
            case "fedprox":
                logger.debug("Aggregating models using FedProx")
                return self.fed_prox(client_results)
            case _:
                raise ValueError(
                    f"Invalid aggregation strategy: {self.aggregation_strategy}"
                )

    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        logger.debug("Updating global model with aggregated parameters")
        self.global_cmam.load_state_dict(aggregated_params)

    def evaluate_global_model(
        self, data: DataLoader, criterion: Module
    ) -> Dict[str, Any]:
        self.global_cmam.metric_recorder.reset()
        metric_recorder = self.global_cmam.metric_recorder.clone()
        for batch in tqdm(data, desc="Evaluating global model", ascii=True):
            eval_results = self.global_cmam.evaluate(
                batch=batch,
                device=self.device,
                cmam_criterion=criterion,
                labels=batch["label"],
                trained_model=self.global_model,
            )
            metric_recorder.update_from_dict(eval_results)
        return metric_recorder.get_average_metrics(
            save_to=self.config.server_config.logging.metrics_path.format_map(
                SafeDict(round=str(self.current_round))
            ),
            epoch="post_aggregation",
        )

    def train_round(
        self,
        epochs: int,
        optimizer: Optimizer,
        criterion: Module,
        clients: List[FederatedMultimodalClient],
        *,
        print_fn: callable = print,
    ) -> Dict[str, Any]:
        self.current_round += 1

        ## local training first
        best_eval_epoch = -1  # record the best eval epoch
        best_metrics = defaultdict(lambda: 0.0)
        best_eval_loss = 1e10

        self.global_cmam.to(self.device)
        self.global_cmam.train()
        self.global_model.to(self.device)
        self.global_model.eval()
        for epoch in tqdm(
            range(1, epochs + 1), desc="Global model training", ascii=True, total=epochs
        ):
            self.global_cmam.metric_recorder.reset()
            epoch_metric_recorder = self.global_cmam.metric_recorder.clone()

            for i, batch in tqdm(
                enumerate(self.global_train_data),
                desc=f"Epoch {epoch}",
                ascii=True,
                total=len(self.global_train_data),
            ):
                # batch["missing_index"] = mp[batch_size * i : batch_size * (i + 1), :]

                train_results = self.global_cmam.train_step(
                    batch=batch,
                    labels=batch["label"],
                    optimizer=optimizer,
                    cmam_criterion=criterion,
                    device=self.device,
                    trained_model=self.global_model,
                )
                epoch_metric_recorder.update_from_dict(train_results)

            global_train_metrics = epoch_metric_recorder.get_average_metrics()
            print_fn("Global model training metrics")
            display_training_metrics(
                metrics=global_train_metrics,
                console=None,
            )

            print_fn("Finished training global model")
            print_fn("Evaluating global model on validation data")
            self.global_cmam.eval()
            self.global_cmam.metric_recorder.reset()
            epoch_metric_recorder.reset()

            for batch in tqdm(
                self.global_val_data,
                desc="Validation",
                ascii=True,
                total=len(self.global_val_data),
            ):
                # batch["missing_index"] = mp
                val_results = self.global_cmam.evaluate(
                    batch=batch,
                    device=self.device,
                    cmam_criterion=criterion,
                    labels=batch["label"],
                    trained_model=self.global_model,
                )
                epoch_metric_recorder.update_from_dict(val_results)

            validation_metrics = epoch_metric_recorder.get_average_metrics(
                save_to=self.config.server_config.logging.metrics_path.format_map(
                    SafeDict(round=str(self.current_round))
                ),
                epoch="pre_aggregation",
            )
            print_fn("Global model validation metrics")
            display_validation_metrics(
                metrics=validation_metrics,
                console=None,
            )

            model_state_dict = self.global_cmam.state_dict()
            os.makedirs(
                os.path.dirname(
                    self.config.server_config.logging.model_output_path.format_map(
                        SafeDict(round=str(self.current_round))
                    )
                ),
                exist_ok=True,
            )
            torch.save(
                model_state_dict,
                self.config.server_config.logging.model_output_path.format_map(
                    SafeDict(round=str(self.current_round))
                ).replace(".pth", f"_{epoch}.pth"),
            )

            p = self.config.server_config.logging.model_output_path.format_map(
                SafeDict(round=str(self.current_round))
            ).replace(".pth", f"_{epoch}.pth")

            print_fn(f"Epoch {epoch} saved at {p}")

            save_metric = self.config.server_config.logging.save_metric.replace(
                "_", "", 1
            )
            # EARLY STOPPING - Based on the target metric

            if self.config.server_config.early_stopping:
                patience = (
                    self.config.server_config.early_stopping_patience
                )  # Number of epochs to wait for improvement
                min_delta = self.config.server_config.early_stopping_min_delta
                # Determine whether we are minimizing or maximizing the metric
                if save_metric == "loss":
                    improvement = (
                        best_metrics[save_metric] - validation_metrics[save_metric]
                    )
                    print_fn(
                        f"Improvement: {improvement}, Best: {best_metrics[save_metric]}, Current: {validation_metrics[save_metric]}"
                    )
                else:
                    improvement = (
                        validation_metrics[save_metric] - best_metrics[save_metric]
                    )  # Improvement means increase

                if epoch > patience:
                    # Check if there is significant improvement by at least `min_delta`
                    if improvement < min_delta:
                        print_fn(
                            f"No improvement for {patience} epochs, stopping training..."
                        )

                        break

            if save_metric == "loss":
                if validation_metrics["loss"] < best_eval_loss:
                    print_fn(f"New best model found at epoch {epoch}")
                    best_eval_epoch = epoch
                    best_eval_loss = validation_metrics["loss"]
                    best_metrics = validation_metrics
            else:
                target_metric = validation_metrics[save_metric]
                if target_metric > best_metrics[save_metric]:
                    print_fn(f"New best model found at epoch {epoch}")
                    best_eval_epoch = epoch
                    best_metrics = validation_metrics

        print_fn(
            f"Best eval epoch was {best_eval_epoch} with a {save_metric} of {best_metrics[save_metric]}"
        )

        print_fn("Global Model Training complete: Round: ", self.current_round)

        ## load the best model
        model_state_dict = torch.load(
            self.config.server_config.logging.model_output_path.format_map(
                SafeDict(round=str(self.current_round))
            ).replace(".pth", f"_{best_eval_epoch}.pth"),
            weights_only=True,
        )

        ## CLIENT TRAINING ##

        selected_clients = self.select_clients()
        global_params = self.distribute_model()
        client_results = []

        for client_id in selected_clients:
            client = clients[client_id]
            client.set_model_parameters(global_params)
            result = client.train_round()
            client_results.append(result)

        aggregated_params = self.aggregate_models(client_results)
        self.update_global_model(aggregated_params)

        eval_results = self.evaluate_global_model(
            data=self.global_test_data, criterion=criterion
        )

        return {
            "round": self.current_round,
            "global_model_performance": eval_results,
            "client_results": client_results,
        }

    def run_federated_learning(
        self,
        num_rounds: int,
        clients: List[FederatedMultimodalClient],
        epochs: int,
        optimizer: Optimizer,
        criterion: Module,
        print_fn: callable = print,
    ) -> dict[str, Any]:
        best_global_performance = None
        best_global_model_state = None
        best_round = 0

        for round in range(1, num_rounds + 1):
            print_fn(f"Starting round {round}/{num_rounds}")

            round_results = self.train_round(
                epochs=epochs,
                optimizer=optimizer,
                criterion=criterion,
                clients=clients,
                print_fn=print_fn,
            )

            global_performance = round_results["global_model_performance"]
            print_fn("Global model performance after aggregation:")
            display_validation_metrics(
                metrics=global_performance,
                console=None,
            )

            os.makedirs(
                self.config.server_config.logging.metrics_path.format_map(
                    SafeDict(round=str(self.current_round))
                ),
                exist_ok=True,
            )

            for k, v in global_performance.items():
                if isinstance(v, np.ndarray):
                    global_performance[k] = v.tolist()

            with open(
                os.path.join(
                    self.config.server_config.logging.metrics_path.format_map(
                        SafeDict(round=str(self.current_round))
                    ),
                    f"round_{round}_best_metrics.json",
                ),
                "w",
            ) as f:
                json_str = json.dumps(global_performance, indent=4)
                f.write(json_str)

            # Check if this round's performance is the best so far
            save_metric = self.config.server_config.logging.save_metric.replace("_", "")
            current_metric_value = global_performance[save_metric]
            best_metric_value = (
                best_global_performance[save_metric]
                if best_global_performance
                else None
            )
            minimising = True if save_metric == "loss" else False

            if (
                best_global_performance is None
                or (minimising and current_metric_value < best_metric_value)
                or (not minimising and current_metric_value > best_metric_value)
            ):
                best_global_performance = global_performance
                best_global_model_state = self.global_cmam.state_dict()
                best_round = self.current_round

                # Create the directory if it doesn't exist
                output_dir = os.path.dirname(
                    self.config.server_config.logging.model_output_path
                )
                os.makedirs(output_dir, exist_ok=True)

                # Create a filename that includes the round number and metric value
                filename = f"best_model_round_{best_round}_{save_metric}_{current_metric_value:.4f}.pth"
                full_path = os.path.join(output_dir, filename)

                # Save the best model
                torch.save(best_global_model_state, full_path)
                print_fn(
                    f"New best model saved at round {best_round} with {save_metric} of {current_metric_value:.4f}"
                )

            # Early stopping check
            if self.config.server_config.early_stopping:
                if (
                    round - best_round
                    >= self.config.server_config.early_stopping_patience
                ):
                    print_fn(
                        f"Early stopping triggered. No improvement for {self.config.server_config.early_stopping_patience} rounds."
                    )
                    break

        print_fn("Training complete")
        print_fn(f"Best round: {best_round} with {save_metric} of {best_metric_value}")
        display_validation_metrics(
            metrics=best_global_performance,
            console=None,
        )

        # Load the best model for final evaluation
        self.global_cmam.load_state_dict(best_global_model_state)

        model_path = self.config.server_config.logging.model_output_path.format_map(
            SafeDict(round=str(""))
        )

        model_path = model_path.replace(".pth", "_best.pth")
        torch.save(best_global_model_state, model_path)

        final_test_results = self.evaluate_global_model(
            self.global_test_data, criterion
        )

        print_fn("Final test results for the best global model:")
        display_validation_metrics(
            metrics=final_test_results,
            console=None,
        )

        return {
            "best_round": best_round,
            "best_global_performance": best_global_performance,
            "final_test_results": final_test_results,
        }

    def __str__(self) -> str:
        return f"FederatedCongruentServer: {self.global_cmam}\nAggregation Strategy: {self.aggregation_strategy}\nNumber of Clients: {self.num_clients}\nDevice: {self.device}\nCurrent Round: {self.current_round}"

    def __repr__(self) -> str:
        return self.__str__()
