import argparse
import json
import os

import torch
from modalities import add_modality
from rich.console import Console

from config import FederatedConfig, resolve_model_name
from data import (
    create_federated_dataloaders,
    create_federated_datasets,
)
from federated.client import FederatedMultimodalClient
from federated.server import FederatedCongruentServer
from models import MultimodalModelProtocol, check_protocol_compliance
from utils import SafeDict, print_all_metrics_tables
from utils.logger import get_logger
from utils.metric_recorder import MetricRecorder

add_modality("VIDEO")

console = Console()


def main(config_path: str, run_id: int = -1):
    # Load configuration
    config = FederatedConfig.load(config_path, run_id=run_id)
    console.print(
        config,
        style="bold green",
    )
    # Set up logging
    server_logger = get_logger(config.server_config.logging_config.log_path)
    server_logger.info(f"Loaded configuration from {config_path}")

    # Set device
    device = torch.device(config.experiment.device)

    num_clients = config.server_config.num_clients

    ## create all dataloaders
    (
        train_dataset,
        train_iid_metrics,
        val_dataset,
        val_iid_metrics,
        test_dataset,
        test_iid,
    ) = create_federated_datasets(
        num_clients=num_clients, data_config=config.data_config
    )
    console.rule("IID Metrics")
    iid_metrics = {
        "train": train_iid_metrics,
        "val": val_iid_metrics,
        "test": test_iid,
    }

    for split, metrics in iid_metrics.items():
        if metrics is not None:
            console.print(f"[bold]{split}[/bold]")
            print_all_metrics_tables(
                metrics, max_cols_per_row=10, max_width=20, console=console
            )
    if config.data_config.distribution_type == "non_iid":
        iid_metrics_output_path = os.path.join(
            config.server_config.logging_config.metrics_path.format_map(
                SafeDict(
                    run_id=config.experiment.run_id,
                    alpha=f"_{config.data_config.alpha}",
                    round="",
                )
            ),
            "iid_metrics.json",
        )
    else:
        iid_metrics_output_path = os.path.join(
            config.server_config.logging_config.metrics_path.format_map(
                SafeDict(run_id=config.experiment.run_id, round="")
            ),
            "iid_metrics.json",
        )
    os.makedirs(os.path.dirname(iid_metrics_output_path), exist_ok=True)
    ## write the iid metrics to a file under the metrics path
    with open(iid_metrics_output_path, "w") as f:
        json_str = json.dumps(iid_metrics, indent=4)
        f.write(json_str)
    console.print(f"IID Metrics saved to {iid_metrics_output_path}")

    federated_dataloaders = create_federated_dataloaders(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        },
        config.data_config,
    )

    global_train_loader = federated_dataloaders["train"]["global"]
    global_val_loader = federated_dataloaders["validation"]["global"]
    global_test_loader = federated_dataloaders["test"]["global"]

    global_model_cls = resolve_model_name(
        config.server_config.global_model_config["name"]
    )

    metric_recorder = MetricRecorder(config.metrics)

    global_model = global_model_cls(
        **config.server_config.global_model_config.kwargs,
        metric_recorder=metric_recorder,
    )
    global_model.to(device)
    check_protocol_compliance(global_model, MultimodalModelProtocol)

    global_optimizer = config._get_optimizer(global_model, is_global=True)
    global_criterion = config._get_criterion(is_global=True)

    # Initialize server
    server = FederatedCongruentServer(
        model=global_model,
        global_train_data=global_train_loader,
        global_val_data=global_val_loader,
        global_test_data=global_test_loader,
        num_clients=config.server_config.num_clients,
        device=device,
        aggregation_strategy=config.server_config.aggregation_strategy,
        config=config,
    )

    # Initialize clients
    clients = []
    for client_id in range(
        config.server_config.num_clients
    ):  # Start from 1 as 0 is for global
        client_config = config.get_client_config()

        # Initialize client model
        client_model = global_model_cls(
            **config.client_config.model_config.kwargs,
            metric_recorder=metric_recorder.clone(),
        )
        client_model.to(device)

        optimizer = config._get_optimizer(client_model, is_global=False)
        criterion = config._get_criterion(is_global=False)

        client_train_loader = federated_dataloaders["train"]["clients"][client_id]
        client_val_loader = federated_dataloaders["validation"]["clients"][client_id]
        client_test_loader = federated_dataloaders["test"]["clients"][client_id]
        client = FederatedMultimodalClient(
            client_id=client_id,
            model=client_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=client_train_loader,
            val_loader=client_val_loader,
            test_loader=client_test_loader,
            fed_config=client_config,
            total_epochs=client_config.local_epochs,
            run_id=config.experiment.run_id,
            model_output_dir=config.client_config.logging.model_output_path,
            logging_output_dir=config.client_config.logging.log_path,
            metrics_output_dir=config.client_config.logging.metrics_path,
        )
        clients.append(client)

    # Run federated learning
    results = server.run_federated_learning(
        num_rounds=config.server_config.rounds,
        clients=clients,
        epochs=config.server_config.epochs,
        optimizer=global_optimizer,
        criterion=global_criterion,
        print_fn=console.print,
    )

    # Print final results
    server_logger.info("Federated Learning completed")
    print_all_metrics_tables(results["final_test_results"], console=console)

    console.print(f"Best Round: {results["best_round"]}")
    console.print(f"Best Global Performance:\n{results["best_global_performance"]}")

    # Save results
    with open(config.server_config.logging_config.metrics_path, "w") as f:
        json_str = json.dumps(results["final_test_results"], indent=4)
        f.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument("--run_id", default=-1, type=int, help="Run ID for logging")
    args = parser.parse_args()

    main(args.config, args.run_id)
