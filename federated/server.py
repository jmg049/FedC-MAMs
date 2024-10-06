import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Literal

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.federated_config import FederatedConfig
from federated import FederatedResult
from federated.client import FederatedMultimodalClient
from models import MultimodalModelProtocol
from utils import (
    SafeDict,
    display_training_metrics,
    display_validation_metrics,
    get_logger,
)

logger = get_logger()


class FederatedCongruentServer:
    def __init__(
        self,
        model: MultimodalModelProtocol,
        global_train_data: DataLoader,
        global_val_data: DataLoader,
        global_test_data: DataLoader,
        num_clients: int,
        device: torch.device,
        config: FederatedConfig,
        aggregation_strategy: Literal["fedavg"] = "fedavg",
        selection_strategy: Literal["all", "random"] = "all",
    ):
        assert (
            hasattr(model, "metric_recorder") and model.metric_recorder is not None
        ), "Model must have its metric recorder set and not None"
        self.global_model = model
        self.aggregation_strategy = aggregation_strategy
        self.num_clients = num_clients
        self.current_round = 0
        # self.best_round = 0
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
        return deepcopy(self.global_model.state_dict())

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

    def aggregate_models(
        self, client_results: List[FederatedResult]
    ) -> Dict[str, torch.Tensor]:
        match self.aggregation_strategy:
            case "fedavg":
                return self.fed_avg(client_results)
            case _:
                raise ValueError(
                    f"Invalid aggregation strategy: {self.aggregation_strategy}"
                )

    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        self.global_model.load_state_dict(aggregated_params)

    def evaluate_global_model(
        self, data: DataLoader, criterion: Module
    ) -> Dict[str, Any]:
        self.global_model.metric_recorder.reset()
        metric_recorder = self.global_model.metric_recorder.clone()
        for batch in tqdm(data, desc="Evaluating global model", ascii=True):
            eval_results = self.global_model.evaluate(
                batch=batch, device=self.device, criterion=criterion
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

        print_fn(f"Starting round {self.current_round}")
        logger.info(f"Starting round {self.current_round}")
        for epoch in tqdm(
            range(1, epochs + 1), desc="Global model training", ascii=True, total=epochs
        ):
            self.global_model.metric_recorder.reset()
            epoch_metric_recorder = self.global_model.metric_recorder.clone()

            for i, batch in tqdm(
                enumerate(self.global_train_data),
                desc=f"Epoch {epoch}",
                ascii=True,
                total=len(self.global_train_data),
            ):
                train_results = self.global_model.train_step(
                    batch=batch,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=self.device,
                )
                epoch_metric_recorder.update_from_dict(train_results)

            global_train_metrics = epoch_metric_recorder.get_average_metrics()
            print_fn("Global model training metrics")
            display_training_metrics(
                metrics=global_train_metrics,
                console=None,
            )

            logger.info(f"Finished training global model for epoch {epoch}")
            logger.debug(f"Global model training metrics for epoch {epoch}")
            logger.debug(global_train_metrics)

            print_fn("Finished training global model")
            print_fn("Evaluating global model on validation data")
            self.global_model.eval()
            self.global_model.metric_recorder.reset()
            epoch_metric_recorder.reset()

            for batch in tqdm(
                self.global_val_data,
                desc="Validation",
                ascii=True,
                total=len(self.global_val_data),
            ):
                val_results = self.global_model.evaluate(
                    batch=batch, device=self.device, criterion=criterion
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

            model_state_dict = self.global_model.state_dict()
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

            print_fn(
                f"Saving model at epoch {epoch} to disk to {self.config.server_config.logging.model_output_path.format_map(SafeDict(round=str(self.current_round))).replace('.pth', f'_{epoch}.pth')}"
            )

            p = self.config.server_config.logging.model_output_path.format_map(
                SafeDict(round=str(self.current_round))
            ).replace(".pth", f"_{epoch}.pth")

            print_fn(f"Epoch {epoch} saved at {p}")
            # EARLY STOPPING - Based on the target metric
            # if self.config.server_config.early_stopping:
            #     patience = (
            #         self.config.server_config.early_stopping_patience
            #     )  # Number of epochs to wait for improvement
            #     min_delta = self.config.server_config.early_stopping_min_delta
            #     # Determine whether we are minimizing or maximizing the metric
            #     if self.config.server_config.logging.save_metric == "loss":
            #         improvement = (
            #             best_metrics[self.config.server_config.logging.save_metric]
            #             - validation_metrics[
            #                 self.config.server_config.logging.save_metric
            #             ]
            #         )
            #         print_fn(
            #             f"Improvement: {improvement}, Best: {best_metrics[self.config.server_config.logging.save_metric]}, Current: {validation_metrics[self.config.server_config.logging.save_metric]}"
            #         )
            #     else:
            #         improvement = (
            #             validation_metrics[
            #                 self.config.server_config.logging.save_metric
            #             ]
            #             - best_metrics[self.config.server_config.logging.save_metric]
            #         )  # Improvement means increase

            #     if epoch > patience:
            #         # Check if there is significant improvement by at least `min_delta`
            #         if improvement < min_delta:
            #             print_fn(
            #                 f"No improvement for {patience} epochs, stopping training..."
            #             )

            #             break

            if self.config.server_config.logging.save_metric.replace("_", "") == "loss":
                if validation_metrics["loss"] < best_eval_loss:
                    print_fn(f"New best model found at epoch {epoch}")
                    best_eval_epoch = epoch
                    best_eval_loss = validation_metrics["loss"]
                    best_metrics = validation_metrics
            else:
                target_metric = validation_metrics[
                    self.config.server_config.logging.save_metric.replace("_", "", 1)
                ]
                if (
                    target_metric
                    > best_metrics[
                        self.config.server_config.logging.save_metric.replace(
                            "_", "", 1
                        )
                    ]
                ):
                    print_fn(f"New best model found at epoch {epoch}")
                    best_eval_epoch = epoch
                    best_metrics = validation_metrics

        print_fn(
            f"Best eval epoch was {best_eval_epoch} with a {self.config.server_config.logging.save_metric.replace('_', '', 1)} of {best_metrics[self.config.server_config.logging.save_metric.replace('_','',1)]}"
        )
        logger.info(
            f"Best eval epoch was {best_eval_epoch} with a {self.config.server_config.logging.save_metric.replace('_', '', 1)} of {best_metrics[self.config.server_config.logging.save_metric.replace('_','', 1)]}"
        )

        print_fn("Global Model Training complete: Round: %d", self.current_round)
        logger.info("Global Model Training complete: Round: %d", self.current_round)

        ## load the best model
        model_state_dict = torch.load(
            self.config.server_config.logging.model_output_path.format_map(
                SafeDict(round=str(self.current_round))
            ).replace(".pth", f"_{best_eval_epoch}.pth"),
            weights_only=True,
        )

        print_fn(f"Loading best model from epoch {best_eval_epoch} for aggregation")
        self.global_model.load_state_dict(model_state_dict)

        ## CLIENT TRAINING ##

        selected_clients = self.select_clients()
        print_fn(f"Selected clients: {[c for c in selected_clients]}")
        logger.info(f"Selected clients: {[c for c in selected_clients]}")
        global_params = self.distribute_model()
        client_results = []

        for client_id in selected_clients:
            client = clients[client_id]
            client.set_model_parameters(global_params)
            result = client.train_round()
            client_results.append(result)

        aggregated_params = self.aggregate_models(client_results)
        self.update_global_model(aggregated_params)
        print_fn(f"Global model updated after round {self.current_round}")
        logger.info(f"Global model updated after round {self.current_round}")

        model_path = self.config.server_config.logging.model_output_path.format_map(
            SafeDict(round=str(self.current_round))
        )

        model_path = model_path.replace(".pth", "_aggregated.pth")

        torch.save(
            self.global_model.state_dict(),
            model_path,
        )

        print_fn(f"Global model saved at {model_path}")

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
    ):
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

            # with open(
            #     os.path.join(
            #         self.config.server_config.logging.metrics_path.format_map(
            #             SafeDict(round=str(self.current_round))
            #         ),
            #         f"round_{round}_best_metrics.json",
            #     ),
            #     "w",
            # ) as f:
            #     json_str = json.dumps(global_performance, indent=4)
            #     f.write(json_str)

            # Check if this round's performance is the best so far
            save_metric = self.config.server_config.logging.save_metric.replace(
                "_", "", 1
            )
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
                best_global_model_state = self.global_model.state_dict()
                best_round = self.current_round

                # Create the directory if it doesn't exist
                output_dir = os.path.dirname(
                    self.config.server_config.logging.model_output_path.format_map(
                        SafeDict(round=str(self.current_round))
                    )
                )
                os.makedirs(output_dir, exist_ok=True)

                filename = f"best_model_{save_metric}.pth"
                full_path = os.path.join(output_dir, filename)

                # Save the best model
                torch.save(best_global_model_state, full_path)
                print_fn(
                    f"New best model saved at round {best_round} with {save_metric} of {current_metric_value:.4f}"
                )
                logger.info(
                    f"New best model saved at round {best_round} with {save_metric} of {current_metric_value:.4f}"
                )

                print_fn(f"Best model saved at {full_path}")
                logger.info(f"Best model saved at {full_path}")

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

        print_fn(f"Best round: {best_round} with performance:")
        display_validation_metrics(
            metrics=best_global_performance,
            console=None,
        )

        # Load the best model for final evaluation
        self.global_model.load_state_dict(best_global_model_state)

        ## save the best model for future use
        model_path = self.config.server_config.logging.model_output_path.format_map(
            SafeDict(round="")
        )  ## empty string to save the best model to the /global/ directory

        model_path = model_path.replace(".pth", "_best.pth")
        torch.save(self.global_model.state_dict(), model_path)

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
        return f"FederatedCongruentServer: {self.global_model}\nAggregation Strategy: {self.aggregation_strategy}\nNumber of Clients: {self.num_clients}\nDevice: {self.device}\nCurrent Round: {self.current_round}"

    def __repr__(self) -> str:
        return str(self)
