import argparse
import logging

from rich.console import Console
from config.config import resolve_model_name
from config.federated_incongruent_config import FederatedIncongreuntConfig
import utils
import torch

from utils.metric_recorder import MetricRecorder

console = Console()


def main(config: FederatedIncongreuntConfig):
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
    config = FederatedIncongreuntConfig.load(config_path, run_id=run_id)
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
