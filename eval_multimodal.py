import argparse
import json
import os

import numpy as np
import torch
from modalities import add_modality
from rich.console import Console
from tqdm import tqdm

from config import StandardConfig, resolve_model_name
from data import build_dataloader
from models import MultimodalModelProtocol, check_protocol_compliance, kaiming_init
from utils import (
    clean_checkpoints,
    display_validation_metrics,
)
from utils.logger import get_logger
from utils.metric_recorder import MetricRecorder

add_modality("VIDEO")

console = Console()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument("--run_id", type=int, default=-1, help="Run ID")
    args = parser.parse_args()

    try:
        config = StandardConfig.load(args.config, args.run_id)
    except Exception as e:
        console.print(f"Error loading config:\n{e}")
        exit(1)

    logger_path = config.logging.log_path
    logger = get_logger(logger_path)
    console.print(f"Initalized logger. Logging to {logger_path}")
    result_dir = config.logging.metrics_path
    console.print(f"Results will be saved to {result_dir}")

    epochs = config.training.epochs
    batch_size = config.training.batch_size
    device = config.experiment.device
    run_idx = config.experiment.run_id

    verbose = config.experiment.debug

    _ = clean_checkpoints(os.path.dirname(config.logging.model_output_path), -1)

    dataloaders = {
        split: build_dataloader(
            data_config=config.data, target_split=split, batch_size=batch_size
        )
        for split in config.data.datasets.keys()
    }
    do_test = config.experiment.do_test

    tst_dataset = dataloaders["test"]

    logger.info("The number of test samples = %d" % len(tst_dataset.dataset))

    metric_recorder = MetricRecorder(config=config.metrics)

    model_cls = resolve_model_name(config.model.name)

    pretrained_path = config.model.pretrained_path

    model = model_cls(**config.model.kwargs, metric_recorder=metric_recorder)
    check_protocol_compliance(model, MultimodalModelProtocol)
    kaiming_init(
        model,
    )
    model.to(device)
    model.load_state_dict(torch.load(config.model.pretrained_path, weights_only=True))

    criterion = config.get_criterion(
        criterion_name=config.training.criterion,
        criterion_kwargs=config.training.criterion_kwargs,
    )

    console.print("Optimizer and criterion created")
    logger.info(f"Criterion created\n{criterion}")

    console.print("Testing model")

    model.metric_recorder.reset()
    epoch_metric_recorder = model.metric_recorder.clone()
    for batch in tqdm(tst_dataset, desc="Test", unit="Batches", colour="green"):
        results = model.evaluate(batch=batch, criterion=criterion, device=device)
        epoch_metric_recorder.update_from_dict(results)

    cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
    if cm is not None:
        cm = np.sum(cm, axis=0)
        tqdm.write(str(cm))
        del epoch_metric_recorder.results["ConfusionMatrix"]

    epoch_metrics = epoch_metric_recorder.get_average_metrics()

    console.rule("Test Metrics")
    display_validation_metrics(
        metrics=epoch_metrics,
        console=console,
    )

    file_name = os.path.join(result_dir, "test_metrics.json")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as f:
        json_str = json.dumps(epoch_metrics, indent=4)
        f.write(json_str)
