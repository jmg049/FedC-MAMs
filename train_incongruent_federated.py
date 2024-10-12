import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pprint import pformat
from typing import List
import numpy as np
from numpy import ndarray
import torch
from modalities import add_modality, Modality
from rich.console import Console

from config.federated_incongruent_config import FederatedIncongruentConfig
from federated.incongruent_client import FederatedIncongruentClient
from federated.incongruent_server import FederatedIncongruentServer
import utils
from config import FederatedConfig, resolve_model_name
from data import (
    create_federated_dataloaders,
    create_federated_datasets,
    resolve_dataset_name,
)
from federated.client import FederatedMultimodalClient
from utils import SafeDict, display_training_metrics, display_validation_metrics
from utils.metric_recorder import MetricRecorder
from torch.utils.data import DataLoader
from data.federated_dataset import (
    FederatedIncongruentDatasetWrapper,
)

try:
    add_modality("VIDEO")
except ValueError as ve:
    print(ve)

console = Console()


def sample_missing_types(
    num_clients: int, missing_types: List[str], strategy: str = "uniform", **kwargs
) -> dict[int, str]:
    """
    Sample a missing type for each client based on a given strategy.

    Args:
        num_clients (int): The number of clients.
        missing_types (List[str]): The list of possible missing types.
        strategy (str): The sampling strategy, one of ["uniform", "normal", "dirichlet"].
        kwargs: Additional parameters for each strategy (e.g., mean, variance for normal).

    Returns:
        Dict[int, str]: A dictionary where keys are client IDs and values are the assigned missing type.
    """
    client_missing_assignments = {}

    if strategy == "uniform":
        for client_id in range(num_clients):
            selected_type = np.random.choice(missing_types)
            client_missing_assignments[client_id] = selected_type

    elif strategy == "normal":
        mean = kwargs.get("mean", len(missing_types) / 2)
        std_dev = kwargs.get("std_dev", len(missing_types) / 4)
        num_missing_per_client = np.random.normal(
            loc=mean, scale=std_dev, size=num_clients
        ).astype(int)
        num_missing_per_client = np.clip(
            num_missing_per_client, 0, len(missing_types) - 1
        )

        for client_id, num_missing in enumerate(num_missing_per_client):
            selected_type = missing_types[num_missing]
            client_missing_assignments[client_id] = selected_type
    elif strategy == "custom":
        ## count the number of 'z' in each missing type
        ## weighted assignment based on the number of 'z' in the missing type
        ## Add 1 to the count to avoid zero probabilities for the fullly available case
        ## For two modalities this should result in a 40-40-20 split AZ, BZ, AB
        z_counter = defaultdict(int)
        for missing_type in missing_types:
            z_counter[missing_type] = missing_type.count("z") + 1

        for client_id in range(num_clients):
            selected_type = np.random.choice(
                missing_types,
                p=[
                    z_counter[missing_type] / sum(z_counter.values())
                    for missing_type in missing_types
                ],
            )
            client_missing_assignments[client_id] = selected_type

    elif strategy == "dirichlet":
        alpha = kwargs.get("alpha", 1.0)
        proportions = np.random.dirichlet(
            [alpha] * len(missing_types), size=num_clients
        )

        for client_id, client_proportions in enumerate(proportions):
            selected_type_idx = np.random.choice(
                len(missing_types), p=client_proportions
            )
            selected_type = missing_types[selected_type_idx]
            client_missing_assignments[client_id] = selected_type

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Use one of ['uniform', 'normal', 'dirichlet']"
        )

    return client_missing_assignments


def main(config: FederatedIncongruentConfig):
    logger.info("Starting Federated Learning")

    device = torch.device(config.experiment.device)
    console.print(f"Device: {device}")
    logger.debug(f"Device: {device}")

    metric_recorder = MetricRecorder(config.metrics)
    console.print(f"Metric Recorder: {metric_recorder}")
    logger.debug(f"Metric Recorder: {metric_recorder}")

    global_model_cls = resolve_model_name(config.server_config.model_config["name"])
    global_model = global_model_cls(
        **config.server_config.model_config.kwargs,
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
        federated_dataset_wrapper_cls=FederatedIncongruentDatasetWrapper,
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
    dataset_cls = resolve_dataset_name(config.data_config.datasets["train"]["dataset"])

    federated_dataloaders["train"][
        "global"
    ].dataset.selected_missing_types = dataset_cls.FULL_CONDITION
    federated_dataloaders["validation"][
        "global"
    ].dataset.selected_missing_types = dataset_cls.FULL_CONDITION
    federated_dataloaders["test"][
        "global"
    ].dataset.selected_missing_types = dataset_cls.FULL_CONDITION

    console.print(
        f"Train loader selected missing types: {federated_dataloaders['train']['global'].dataset.selected_missing_types}"
    )
    console.print(
        f"Validation loader selected missing types: {federated_dataloaders['validation']['global'].dataset.selected_missing_types}"
    )
    console.print(
        f"Test loader selected missing types: {federated_dataloaders['test']['global'].dataset.selected_missing_types}"
    )

    global_train_loader = federated_dataloaders["train"]["global"]
    global_val_loader = federated_dataloaders["validation"]["global"]
    global_test_loader = federated_dataloaders["test"]["global"]

    console.print("Global Dataloaders created successfully")
    console.print(f"No. of Global Train Batches: {len(global_train_loader)}")
    console.print(f"No. of Global Validation Batches: {len(global_val_loader)}")
    console.print(f"No. of Global Test Batches: {len(global_test_loader)}")

    cmams = None
    cmam_optimizers = None
    cmam_criterion = None
    ## create global cmams if configured
    if config.server_config.cmam_configs is not None:
        cmams = {}
        for cmam_config in config.server_config.cmam_configs:
            console.print(f"Creating CMAM: {cmam_config}")
            cmam_input_modality = Modality.from_str(cmam_config)
            console.print(f"CMAM Input Modality: {cmam_input_modality}")

            cmam_model_cls = resolve_model_name(
                config.server_config.cmam_configs[cmam_config].name
            )
            cmam = cmam_model_cls(
                input_encoder_info=config.server_config.cmam_configs[
                    cmam_config
                ].input_encoder_info,
                metric_recorder=metric_recorder.clone(),
                **config.server_config.cmam_configs[cmam_config].__dict__(),
            )
            cmams[cmam_input_modality] = cmam

        cmam_optimizers = {}
        for cmam_modality in cmams:
            cmam_optimizer = config._get_optimizer(cmams[cmam_modality], is_global=True)
            cmam_optimizers[cmam_modality] = cmam_optimizer

        cmam_criterion = config._get_criterion(is_global=True, cmam=True)

    # Initialize server
    server = FederatedIncongruentServer(
        mm_model=global_model,
        train_dataloader=global_train_loader,
        validation_dataloader=global_val_loader,
        test_dataloader=global_test_loader,
        client_ids=[i for i in range(config.server_config.num_clients)],
        device=device,
        aggregation_strategy=config.server_config.aggregation_strategy,
        selection_strategy=config.server_config.selection_strategy,
        config=config,
        cmams=cmams,
        mm_optimizer=global_optimizer,
        mm_criterion=global_criterion,
        cmam_optimizers=cmam_optimizers,
        cmam_criterion=cmam_criterion,
    )

    console.print("Server initialized successfully")
    logger.info("Server initialized successfully")
    console.print(f"{server}")

    # Obtain the list of all possible missing types using the dataset class method
    missing_types = (
        dataset_cls.get_missing_types()
    )  # Dynamically obtained missing types

    # Generate modality assignments for each client
    client_missing_assignments = sample_missing_types(
        num_clients=num_clients, missing_types=missing_types, strategy="custom"
    )

    ## write the missing types to a file along with the client assignments

    missing_types_path = os.path.join(
        config.server_config.cls_logging.metrics_path.replace("cls", "").format_map(
            SafeDict(round="")
        ),
        "missing_types.json",
    )
    os.makedirs(os.path.dirname(missing_types_path), exist_ok=True)

    with open(missing_types_path, "w") as f:
        json.dump(client_missing_assignments, f, indent=4)

    # Initialize clients
    clients = []
    for client_id in range(config.server_config.num_clients):
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

        criterion = config._get_criterion(is_global=False)

        # Get the client-specific missing types
        client_missing_types = client_missing_assignments[client_id]

        # Wrap the train dataset with the client-specific missing modalities
        federated_dataloaders["train"]["clients"][
            client_id
        ].dataset.selected_missing_types = client_missing_types

        federated_dataloaders["validation"]["clients"][
            client_id
        ].dataset.selected_missing_types = client_missing_types
        federated_dataloaders["test"]["clients"][
            client_id
        ].dataset.selected_missing_types = client_missing_types

        # Create dataloaders for the client
        cmam_modality = dataset_cls.get_modality(client_missing_types)

        if cmams is not None:
            # only one cmam per client
            if cmam_modality in cmams:
                cmam = cmams[cmam_modality]
                cmam_logging = config.server_config.cmam_logging[cmam_modality]
                optimizer = config._get_optimizer([client_model, cmam], is_global=False)
        else:
            cmam = None
            cmam_logging = None
            cmam_optimizer = None
            optimizer = config._get_optimizer(client_model, is_global=False)

        client_train_loader = federated_dataloaders["train"]["clients"][client_id]
        client_val_loader = federated_dataloaders["validation"]["clients"][client_id]
        client_test_loader = federated_dataloaders["test"]["clients"][client_id]

        client = FederatedIncongruentClient(
            client_id=client_id,
            mm_model=client_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=client_train_loader,
            validation_loader=client_val_loader,
            test_loader=client_test_loader,
            config=client_config,
            epochs=client_config.local_epochs,
            run_id=config.experiment.run_id,
            cmam=cmam,
            cmam_logging=cmam_logging,
            cmam_optimizer=cmam_optimizer,
            print_fn=console.print,
            modality=cmam_modality,
        )
        clients.append(client)

    logger.debug("Clients initialized successfully")
    for client in clients:
        logger.debug(f"{client}\n\n")

    # Run federated learning
    results = server.run_federated_learning(
        num_rounds=config.server_config.rounds,
        clients=clients,
        mm_epochs=config.server_config.mm_epochs,
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
        config.server_config.cls_logging.metrics_path.format_map(SafeDict(round="")),
        "best_test_metrics.json",
    )

    for k, v in results["final_test_results"].items():
        if isinstance(v, ndarray):
            results["final_test_results"][k] = v.tolist()

    os.makedirs(os.path.dirname(final_metrics_path), exist_ok=True)

    with open(final_metrics_path, "w") as f:
        json_str = json.dumps(results["final_test_results"], indent=4)
        f.write(json_str)
    console.print(f"Metrics saved to {config.server_config.cls_logging.metrics_path}")


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
    config = FederatedIncongruentConfig.load(config_path, run_id=run_id)
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
