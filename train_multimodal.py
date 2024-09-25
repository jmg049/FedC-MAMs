import argparse
import json
import os
from subprocess import Popen
import time
from collections import defaultdict

import numpy as np
import torch
from modalities import add_modality
from rich.console import Console
from tqdm import tqdm

from config import StandardConfig, resolve_model_name
from data import build_dataloader
from missing_index import missing_pattern
from models import kaiming_init, check_protocol_compliance, MultimodalModelProtocol
from utils import clean_checkpoints, print_all_metrics_tables
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

    if do_test:
        train_dataset = dataloaders["train"]
        val_dataset = dataloaders["validation"]
        tst_dataset = dataloaders["test"]
    else:
        train_dataset = dataloaders["train"]
        val_dataset = dataloaders

    dataset_size = len(train_dataset.dataset)
    logger.info("The number of training samples = %d" % dataset_size)
    logger.info("The number of validation samples = %d" % len(val_dataset.dataset))
    if do_test:
        logger.info("The number of test samples = %d" % len(tst_dataset.dataset))

    total_iters = epochs * dataset_size

    console.print("Total iterations: ", total_iters)
    logger.info("Total iterations: %d" % total_iters)

    model_cls = resolve_model_name(config.model.name)
    model = model_cls(**config.model.kwargs)
    check_protocol_compliance(model, MultimodalModelProtocol)
    kaiming_init(
        model,
    )
    model.to(device)
    model.set_metric_recorder(MetricRecorder(config=config.metrics))
    total_iters = 0  # the total number of training iterations

    best_eval_epoch = -1  # record the best eval epoch
    best_metrics = defaultdict(lambda: 0.0)
    best_eval_loss = 1e10

    missing_rate = config.training.missing_rates
    mp = missing_pattern(config.training.num_modalities, dataset_size, missing_rate)

    console.print("Missing pattern created")
    console.print("Missing pattern shape: ", mp.shape)

    optimizer = config.get_optimizer(model)
    criterion = config.get_criterion(
        criterion_name=config.training.criterion,
        criterion_kwargs=config.training.criterion_kwargs,
    )

    console.print("Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{criterion}")

    if config.training.scheduler is not None:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")

    for epoch in tqdm(
        range(1, epochs + 1), desc="Epochs", unit="epoch", total=epochs, colour="yellow"
    ):
        model.metric_recorder.reset()

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration

        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        epoch_metric_recorder = model.metric_recorder.clone()

        for i, data in tqdm(
            enumerate(train_dataset),
            desc="Train",
            unit="Batches",
            colour="red",
            total=len(train_dataset),
        ):
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += batch_size

            data["missing_index"] = mp[batch_size * i : batch_size * (i + 1), :]

            results = model.train_step(
                batch=data, optimizer=optimizer, criterion=criterion, device=device
            )
            epoch_metric_recorder.update_from_dict(results)
            iter_data_time = time.time()

        train_cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
        if train_cm is not None:
            train_cm = np.sum(train_cm, axis=0)
            tqdm.write(str(train_cm))
            del epoch_metric_recorder.results["ConfusionMatrix"]

        train_metrics = epoch_metric_recorder.get_average_metrics()

        console.rule("Training Metrics")
        print_all_metrics_tables(
            metrics=train_metrics,
            console=console,
            max_cols_per_row=15,
            max_width=20,
        )

        logger.info(
            "End of training epoch %d / %d \t Time Taken: %d sec"
            % (epoch, epochs, time.time() - epoch_start_time)
        )

        scheduler.step()
        logger.info("Learning rate = %.7f" % optimizer.param_groups[0]["lr"])

        ########################################################################################

        model.metric_recorder.reset()
        epoch_metric_recorder = model.metric_recorder.clone()
        for batch in tqdm(
            val_dataset,
            desc="Validation",
            unit="Batches",
            colour="blue",
            total=len(val_dataset),
        ):
            # batch["missing_index"] = mp[len(data) * i : len(data) * (i + 1), :]
            results = model.evaluate(batch=batch, criterion=criterion, device=device)
            epoch_metric_recorder.update_from_dict(results)

        validation_cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
        if validation_cm is not None:
            validation_cm = np.sum(validation_cm, axis=0)
            tqdm.write(str(validation_cm))
            del epoch_metric_recorder.results["ConfusionMatrix"]

        validation_metrics = epoch_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                config.logging.metrics_path.rsplit(".", 1)[0], f"validation/{run_idx}"
            ),
            epoch=epoch,
        )

        console.rule("Validation Metrics")

        print_all_metrics_tables(
            metrics=validation_metrics,
            console=console,
            max_cols_per_row=15,
            max_width=15,
        )

        #########################################################################################

        # show test result for debugging
        if do_test and verbose and epoch % 5 == 0:
            model.metric_recorder.reset()
            epoch_metric_recorder = model.metric_recorder.clone()
            for batch in tqdm(tst_dataset, desc="Test", unit="Batches", colour="green"):
                # batch["missing_index"] = mp[batch_size * i : batch_size * (i + 1), :]
                # batch["missing_index"] = mp[len(data) * i : len(data) * (i + 1), :]

                results = model.evaluate(
                    batch=batch, criterion=criterion, device=device
                )
                epoch_metric_recorder.update_from_dict(results)

            test_cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
            if test_cm is not None:
                test_cm = np.sum(test_cm, axis=0)
                tqdm.write(str(test_cm))
                del epoch_metric_recorder.results["ConfusionMatrix"]

            epoch_metrics = epoch_metric_recorder.get_average_metrics()

            console.rule("Test Metrics")
            print_all_metrics_tables(
                metrics=epoch_metrics,
                console=console,
                max_cols_per_row=15,
                max_width=15,
            )

        if config.logging.save_metric == "loss":
            if validation_metrics["loss"] < best_eval_loss:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_eval_loss = validation_metrics["loss"]
                best_metrics = validation_metrics
        else:
            target_metric = validation_metrics[config.logging.save_metric]
            if target_metric > best_metrics[config.logging.save_metric]:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_metrics = validation_metrics

        # EARLY STOPPING - Based on the target metric
        if config.training.early_stopping:
            patience = (
                config.training.early_stopping_patience
            )  # Number of epochs to wait for improvement
            min_delta = config.training.early_stopping_min_delta
            # Determine whether we are minimizing or maximizing the metric
            if config.logging.save_metric == "loss":
                improvement = (
                    best_metrics[config.logging.save_metric]
                    - validation_metrics[config.logging.save_metric]
                )  # Improvement means decrease
            else:
                improvement = (
                    validation_metrics[config.logging.save_metric]
                    - best_metrics[config.logging.save_metric]
                )  # Improvement means increase

            if epoch > patience:
                # Check if there is significant improvement by at least `min_delta`
                if improvement < min_delta:
                    console.print(
                        f"No improvement for {patience} epochs, stopping training..."
                    )
                    logger.info(
                        f"Early stopping at epoch {epoch} due to no improvement in {config.logging.save_metric}"
                    )
                    break

        model_state_dict = model.state_dict()
        os.makedirs(os.path.dirname(config.logging.model_output_path), exist_ok=True)
        torch.save(
            model_state_dict,
            config.logging.model_output_path.replace(".pth", f"_{epoch}.pth"),
        )
        console.print(
            f"Epoch {epoch} saved at {config.logging.model_output_path.replace('.pth', f'_{epoch}.pth')}"
        )
    console.print("Training complete")
    console.print(
        f"Best eval epoch was {best_eval_epoch} with a {config.logging.save_metric} of {best_metrics[config.logging.save_metric]}"
    )
    logger.info(
        f"Best eval epoch was {best_eval_epoch} with a {config.logging.save_metric} of {best_metrics[config.logging.save_metric]}"
    )

    if config.training.do_validation_visualization:
        vis_proc = Popen(
            [
                "python",
                "visualisation/validation_plotting.py",
                "--root_dir",
                config.logging.metrics_path,
                "--save_path",
                os.path.join(config.logging.metrics_path, "validation_plots"),
            ]
        )
    to_store = clean_checkpoints(
        os.path.dirname(config.logging.model_output_path), best_eval_epoch
    )
    assert to_store is not None, "No model to store"

    console.print("Training complete")

    os.rename(to_store, to_store.replace(f"_{best_eval_epoch}.pth", "_best.pth"))
    console.print(
        f"Best model stored at {to_store.replace(f'_{best_eval_epoch}.pth', '_best.pth')}"
    )
    # test
    if do_test:
        console.print("Testing model")
        console.print(f"Loading best model found on val set: epoch-{best_eval_epoch}")
        logger.info(f"Loading best model found on val set: epoch-{best_eval_epoch}")

        model.load_state_dict(
            torch.load(
                config.logging.model_output_path.replace(".pth", "_best.pth"),
                weights_only=True,
            )
        )

        model.metric_recorder.reset()
        epoch_metric_recorder = model.metric_recorder.clone()
        for batch in tqdm(tst_dataset, desc="Test", unit="Batches", colour="green"):
            # batch["missing_index"] = mp[len(data) * i : len(data) * (i + 1), :]
            results = model.evaluate(batch=batch, criterion=criterion, device=device)
            epoch_metric_recorder.update_from_dict(results)

        cm = epoch_metric_recorder.get("ConfusionMatrix", default=None)
        if cm is not None:
            cm = np.sum(cm, axis=0)
            tqdm.write(str(cm))
            del epoch_metric_recorder.results["ConfusionMatrix"]

        epoch_metrics = epoch_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                config.logging.metrics_path.rsplit(".", 1)[0], f"test/{run_idx}"
            ),
        )

        console.rule("Test Metrics")
        print_all_metrics_tables(
            metrics=epoch_metrics,
            console=console,
            max_cols_per_row=15,
            max_width=15,
        )

        file_name = os.path.join(result_dir, f"test_metrics_{best_eval_epoch}.json")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, "w") as f:
            json_str = json.dumps(epoch_metrics, indent=4)
            f.write(json_str)
