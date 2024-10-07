import argparse
import json
import logging
import os
from pprint import pformat

from numpy import ndarray
import torch
from modalities import add_modality
from rich.console import Console

import utils
from config import FederatedConfig, resolve_model_name
from data import (
    create_federated_dataloaders,
    create_federated_datasets,
)
from federated.client import FederatedMultimodalClient
from federated.server import FederatedCongruentServer
from utils import SafeDict, display_training_metrics, display_validation_metrics
from utils.metric_recorder import MetricRecorder

add_modality("VIDEO")

console = Console()


def main(config: FederatedConfig):
    logger.info("Starting Federated Learning")

    device = torch.device(config.experiment.device)
    console.print(f"Device: {device}")
    logger.debug(f"Device: {device}")

    metric_recorder = MetricRecorder(config.metrics)
    console.print(f"Metric Recorder: {metric_recorder}")
    logger.debug(f"Metric Recorder: {metric_recorder}")

    global_model_cls = resolve_model_name(
        config.server_config.global_model_config["name"]
    )
    global_model = global_model_cls(
        **config.server_config.global_model_config.kwargs,
        metric_recorder=metric_recorder,
    )
    global_model.to(device)

    console.print(f"Global Model: {global_model}")
    logger.debug(f"Global Model: {global_model}")

    global_optimizer = config._get_optimizer(global_model, is_global=True)
    global_criterion = config._get_criterion(is_global=True)

    console.print(f"Global Optimizer: {global_optimizer}")
    logger.debug(f"Global Optimizer: {global_optimizer}")

    console.print(f"Global Criterion: {global_criterion}")
    logger.debug(f"Global Criterion: {global_criterion}")

    num_clients = config.server_config.num_clients
    console.print(f"Number of Clients: {num_clients}")
    logger.info(f"Number of Clients: {num_clients}")

    # ## create all datasets
    (
        train_dataset,
        train_iid_metrics,
        val_dataset,
        val_iid_metrics,
        test_dataset,
        test_iid,
    ) = create_federated_datasets(
        num_clients=num_clients,
        data_config=config.data_config,
        indices_save_dir=os.path.dirname(config.logging.iid_metrics_path),
    )

    console.print("Federated Datasets created successfully")

    console.rule("IID Metrics")
    iid_metrics = {
        "train": train_iid_metrics,
        "val": val_iid_metrics,
        "test": test_iid,
    }

    for split, metrics in iid_metrics.items():
        if metrics is not None:
            console.print(f"[bold]{split}[/bold]")
            display_training_metrics(metrics, console=console)
    logger.debug(f"IID Metrics: {iid_metrics}")
    logger.debug(f"Saving iid metrics to {config.logging.iid_metrics_path}")

    os.makedirs(os.path.dirname(config.logging.iid_metrics_path), exist_ok=True)
    with open(config.logging.iid_metrics_path, "w") as f:
        json_str = json.dumps(iid_metrics, indent=4)
        f.write(json_str)

    logger.debug("IID Metrics saved successfully")

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

    console.print("Global Dataloaders created successfully")
    console.print(f"No. of Global Train Batches: {len(global_train_loader)}")
    console.print(f"No. of Global Validation Batches: {len(global_val_loader)}")
    console.print(f"No. of Global Test Batches: {len(global_test_loader)}")

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

    console.print("Server initialized successfully")
    logger.info("Server initialized successfully")
    console.print(f"{server}")

    # Initialize clients
    clients = []
    for client_id in range(
        config.server_config.num_clients
    ):  # Start from 1 as 0 is for global
        client_config = config.get_client_config()

        client_model_output_path = client_config.logging.model_output_path.format_map(
            SafeDict(client_id=client_id)
        )
        metrics_path = client_config.logging.metrics_path.format_map(
            SafeDict(client_id=client_id)
        )

        os.makedirs(os.path.dirname(client_model_output_path), exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)

        logger.info(
            f"Created directories: {client_config.logging.model_output_path} and {client_config.logging.metrics_path}"
        )

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

    logger.debug("Clients initialized successfully")
    for client in clients:
        logger.debug(f"{client}\n\n")

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
    console.rule("Final Results")
    display_validation_metrics(results["final_test_results"], console=console)

    console.print(f"Best Round: {results['best_round']}")
    console.print(
        f"Best Global Performance:\n{pformat(results['best_global_performance'])}"
    )
    console.rule("End of Federated Learning")
    # Save results

    final_metrics_path = os.path.join(
        config.server_config.logging.metrics_path.format_map(SafeDict(round="")),
        "best_test_metrics.json",
    )

    for k, v in results["final_test_results"].items():
        if isinstance(v, ndarray):
            results["final_test_results"][k] = v.tolist()

    with open(final_metrics_path, "w") as f:
        json_str = json.dumps(results["final_test_results"], indent=4)
        f.write(json_str)
    console.print(f"Metrics saved to {config.server_config.logging.metrics_path}")


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
    config_path = args.config
    run_id = args.run_id
    config = FederatedConfig.load(config_path, run_id=run_id)
    console.print(
        config,
        style="bold green",
    )

    console.print("Config loaded successfully", style="bold green")
    console.print("Configuring logger...", style="bold blue")

    level = logging.DEBUG if config.experiment.debug else logging.INFO

    utils.configure_logger(config)

    logger = utils.get_logger()

    ## This message should not be visible in the console
    logger.info(
        "Logger configured successfully. Logging to %s", config.logging.log_path
    )

    logger.info(f"{config.experiment.name} - Run ID: {run_id}")

    console.print(
        "Logger successfully configured and is logging to %s",
        config.logging.log_path + logger.handlers[0].baseFilename,
        style="bold blue",
    )

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(config)

    main(config)
