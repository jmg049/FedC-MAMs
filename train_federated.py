import argparse

import torch

from config import FederatedConfig
from data import (
    FederatedDatasetWrapper,
    create_federated_dataloaders,
    create_federated_datasets,
    resolve_dataset_name,
)
from config import resolve_model_name
from data.label_functions import cmu_get_label_fn
from federated.client import FederatedMultimodalClient
from federated.server import FederatedCongruentServer
from models import MultimodalModelProtocol, check_protocol_compliance
from utils import print_all_metrics_tables
from utils.logger import get_logger

from torch.utils.data import DataLoader
from rich.console import Console

from modalities import add_modality

from utils.metric_recorder import MetricRecorder

add_modality("VIDEO")

console = Console()


def main(config_path: str, run_id: int = -1):
    # Load configuration
    config = FederatedConfig.load(config_path, run_id=run_id)

    # Set up logging
    logger = get_logger(config.logging.log_path)
    logger.info(f"Loaded configuration from {config_path}")

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
    global_model = global_model_cls(**config.server_config.global_model_config.kwargs)
    global_model.to(device)
    check_protocol_compliance(global_model, MultimodalModelProtocol)

    global_model.set_metric_recorder(metric_recorder=MetricRecorder(config.metrics))

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
    )

    # Initialize clients
    clients = []
    for client_id in range(
        config.server_config.num_clients
    ):  # Start from 1 as 0 is for global
        client_config = config.get_client_config()

        # Initialize client model
        client_model = global_model_cls(**config.client_config.model_config.kwargs)
        client_model.to(device)

        optimizer = config._get_optimizer(client_model, is_global=False)
        criterion = config._get_criterion(is_global=False)

        client_train_loader = federated_dataloaders["train"]["clients"][client_id]
        client_val_loader = federated_dataloaders["validation"]["clients"][client_id]
        client_test_loader = federated_dataloaders["test"]["clients"][client_id]
        client_model.set_metric_recorder(metric_recorder=MetricRecorder(config.metrics))
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
        )
        clients.append(client)

    # Run federated learning
    results = server.run_federated_learning(
        num_rounds=config.server_config.rounds,
        clients=clients,
        epochs=config.server_config.epochs,
        optimizer=global_optimizer,
        criterion=global_criterion,
        batch_size=128,
        config=config,
        print_fn=console.print,
    )

    # Print final results
    logger.info("Federated Learning completed")
    print_all_metrics_tables(results["final_test_results"], logger.info)

    # Save results
    # TODO: Implement result saving logic


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
