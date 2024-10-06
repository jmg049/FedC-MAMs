import argparse
import json
import logging
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch
from modalities import add_modality
from rich.console import Console
from tqdm import tqdm

from config import CMAMConfig, resolve_model_name
from data import build_dataloader
from missing_index import missing_pattern
from models import CMAMProtocol, MultimodalModelProtocol, check_protocol_compliance
from models.cmams import DualCMAM
from utils import (
    call_latex_to_image,
    clean_checkpoints,
    prepare_path,
    display_training_metrics,
    display_validation_metrics,
    print_gpu_memory,
)
from utils.logger import configure_logger, get_logger
from utils.metric_recorder import MetricRecorder
import torch
torch.cuda.empty_cache()
add_modality("VIDEO")

console = Console()

# Set to False if your terminal does not support rendering an image
RENDER_LOSS_FN = False


def main(config: CMAMConfig) -> None:
    logger = get_logger()
    result_dir = config.logging.metrics_path
    console.print(f"Results will be saved to {result_dir}")

    epochs = config.training.epochs
    batch_size = config.training.batch_size
    device = config.experiment.device
    run_id = config.experiment.run_id

    os.makedirs(result_dir, exist_ok=True)

    dataloaders = {
        split: build_dataloader(
            data_config=config.data,
            target_split=split,
            batch_size=batch_size,
            print_fn=console.print,
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
        val_dataset = dataloaders["validation"]

    dataset_size = len(train_dataset.dataset)
    logger.info("The number of training samples = %d" % dataset_size)
    console.print(f"Training dataset {train_dataset.dataset}")
    logger.info("The number of validation samples = %d" % len(val_dataset.dataset))
    console.print(f"Validation dataset {val_dataset.dataset}")

    if do_test:
        logger.info("The number of test samples = %d" % len(tst_dataset.dataset))
        console.print(f"Test dataset {tst_dataset.dataset}")

    model_cls = resolve_model_name(config.model.name)
    model = model_cls(
        **config.model.kwargs, metric_recorder=MetricRecorder(config.prediction_metrics)
    )

    check_protocol_compliance(model, MultimodalModelProtocol)

    pretrained_path = config.model.pretrained_path.format(run_id=run_id)
    assert os.path.exists(
        pretrained_path
    ), f"Pretrained model not found at {pretrained_path}"
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    model.to(device)

    


    logger.info(f"Model loaded from {pretrained_path}")
    console.print(f"Model loaded from {pretrained_path}")

    logger.info(f"Classification Model\n{model}")

    best_eval_epoch = -1  # record the best eval epoch
    best_metrics = defaultdict(lambda: 0.0)
    best_eval_loss = 1e10

    missing_rate = config.training.missing_rates
    if missing_rate is not None:
        mp = missing_pattern(config.training.num_modalities, dataset_size, missing_rate)
    else:
        mp = missing_pattern(
            config.training.num_modalities,
            dataset_size,
            [0.0] * config.training.num_modalities,
        )

    console.print("Missing pattern created")
    console.print(f"Missing pattern shape: {mp.shape}")
    cmam_cfg = config.cmam

    input_encoder_info = cmam_cfg.input_encoder_info

    metric_recorder = MetricRecorder(config.prediction_metrics)

    cmam_cls = resolve_model_name(cmam_cfg["name"])
    # cmam_cfg.pop("name")  # safe since we just used it
    cmam = cmam_cls(
        input_encoder_info=input_encoder_info,
        metric_recorder=metric_recorder,
        **cmam_cfg.__dict__(),
    )


    check_protocol_compliance(cmam, CMAMProtocol)

    criterion_kwargs = config.training.criterion_kwargs

    ## print the GPU memory
    console.print(
        f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
    )

    ## calculate x-dim and z-dim for MI loss
    if "mi_weight" in criterion_kwargs and criterion_kwargs["mi_weight"] > 0:
        x_dims = [
            config.cmam.input_encoder_info[k].get_embedding_size()
            for k in config.cmam.input_encoder_info.keys()
        ]
        z_dim = config.cmam.assoc_net_output_size

        criterion_kwargs["x_dims"] = x_dims
        criterion_kwargs["z_dim"] = z_dim

    cmam_criterion = config.get_criterion(config.training.criterion, criterion_kwargs)
    cmam.to(device)

    

    console.print("CMAM model created")
    console.print(f"{cmam}")

    logger.info(f"CMAM model created\n{cmam}")

    optimizer = config._get_optimizer(cmam, optim_kwargs=config.training.optim_kwargs)

    if config.training.scheduler is not None:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")
    else:
        scheduler = None
    console.print("Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{cmam_cfg}")

    loss_fn_tex = cmam_criterion.to_latex()
    with open(os.path.join(result_dir, "loss_fn.tex"), "w") as f:
        f.write(loss_fn_tex)
    if RENDER_LOSS_FN:
        call_latex_to_image(
            latex_str=loss_fn_tex,
        )
    for epoch in tqdm(
        range(1, epochs + 1), desc="Epochs", unit="epoch", total=epochs, colour="yellow"
    ):
        cmam.reset_metric_recorders()

        epoch_rec_metric_recorder = cmam.metric_recorder.clone()

        for i, data in tqdm(
            enumerate(train_dataset),
            desc="Train",
            unit="Batches",
            colour="red",
            total=len(train_dataset),
        ):
            
            
            data["missing_index"] = mp[batch_size * i : batch_size * (i + 1), :]

            results = cmam.train_step(
                batch=data,
                labels=data["label"],
                optimizer=optimizer,
                device=device,
                cmam_criterion=cmam_criterion,
                trained_model=model,
            )
            epoch_rec_metric_recorder.update_from_dict(results)
        train_metrics = epoch_rec_metric_recorder.get_average_metrics()

        console.rule("Training Metrics")
        display_training_metrics(
            metrics=train_metrics,
            console=console,
        )
        logger.info("Epoch %d: Training Metrics: %s" % (epoch, train_metrics))

        ########################################################################################

        cmam.reset_metric_recorders()
        epoch_rec_metric_recorder = cmam.metric_recorder.clone()
        for batch in tqdm(
            val_dataset,
            desc="Validation",
            unit="Batches",
            colour="blue",
            total=len(val_dataset),
        ):
            results = cmam.evaluate(
                batch=batch,
                labels=batch["label"],
                device=device,
                cmam_criterion=cmam_criterion,
                trained_model=model,
            )
            epoch_rec_metric_recorder.update_from_dict(results)

        validation_metrics = epoch_rec_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                f"{config.logging.metrics_path.rsplit('.', 1)[0]}",
                "cmam_validation",
            ),
            epoch=epoch,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()
        logger.info("Learning rate = %.7f" % optimizer.param_groups[0]["lr"])
        console.rule("Validation Metrics")

        display_validation_metrics(
            metrics=validation_metrics,
            console=console,
        )

        # explicity print the loss terms as well

        #########################################################################################

        model_state_dict = cmam.state_dict()
        os.makedirs(os.path.dirname(config.logging.model_output_path), exist_ok=True)
        console.print(f"Model output path: {config.logging.model_output_path}")
        torch.save(
            model_state_dict,
            config.logging.model_output_path.replace(".pth", f"_{epoch}.pth"),
        )
        console.print(
            f"Epoch {epoch} saved at {config.logging.model_output_path.replace('.pth', f'_{epoch}.pth')}"
        )
        logger.info(f"Epoch {epoch} saved at {config.logging.model_output_path}")

        save_metric = config.logging.save_metric.replace("_", "", 1)

        # EARLY STOPPING - Based on the target metric
        if config.training.early_stopping:
            patience = (
                config.training.early_stopping_patience
            )  # Number of epochs to wait for improvement
            min_delta = config.training.early_stopping_min_delta
            # Determine whether we are minimizing or maximizing the metric
            if save_metric == "loss":
                improvement = (
                    best_metrics[save_metric] - validation_metrics[save_metric]
                )
                console.print(
                    f"Improvement: {improvement}, Best: {best_metrics[save_metric]}, Current: {validation_metrics[save_metric]}"
                )
            else:
                improvement = (
                    validation_metrics[save_metric] - best_metrics[save_metric]
                )  # Improvement means increase

            if epoch > patience:
                # Check if there is significant improvement by at least `min_delta`
                if improvement < min_delta:
                    console.print(
                        f"No improvement for {patience} epochs, stopping training..."
                    )
                    logger.info(
                        f"Early stopping at epoch {epoch} due to no improvement in {save_metric}"
                    )
                    break

        if hasattr(cmam, "modality_weights"):
            console.print(f"Modality weights: {cmam.modality_weights}")

        # record epoch with best result as per the target metric

        if save_metric == "loss":
            console.print(
                f"Current best model found at epoch {best_eval_epoch} with a {save_metric} of {best_eval_loss}"
            )
            if validation_metrics["loss"] < best_eval_loss:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_eval_loss = validation_metrics["loss"]
                best_metrics = validation_metrics
        else:
            target_metric = validation_metrics[save_metric]
            if target_metric > best_metrics[save_metric]:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_metrics = validation_metrics

    console.print("Training complete")
    console.print(
        f"Best eval epoch was {best_eval_epoch} with a {save_metric} of {best_metrics[save_metric]}"
    )
    logger.info(
        f"Best eval epoch was {best_eval_epoch} with a {save_metric} of {best_metrics[save_metric]}"
    )

    cmam.load_state_dict(
        torch.load(
            config.logging.model_output_path.replace(".pth", f"_{best_eval_epoch}.pth"),
            weights_only=True,
        )
    )

    ## save the model as the best model
    torch.save(
        model_state_dict,
        config.logging.model_output_path.replace(".pth", "_best.pth"),
    )

    # test
    if do_test:
        console.print("Testing model")
        console.print("Loading best model found on val set: epoch")
        logger.info("Loading best model found on val set: epoch")

        cmam.load_state_dict(
            torch.load(
                config.logging.model_output_path.replace(".pth", "_best.pth"),
                weights_only=True,
            )
        )
        cmam.reset_metric_recorders()
        epoch_rec_metric_recorder = cmam.metric_recorder.clone()
        preds = []
        labels = []
        rec_embds = []
        rec_embds_two = []
        target_embds = []
        target_embds_two = []
        for batch in tqdm(tst_dataset, desc="Test", unit="Batches", colour="green"):
            miss_type = batch["miss_type"]
            results = cmam.evaluate(
                batch=batch,
                labels=batch["label"],
                cmam_criterion=cmam_criterion,
                device=device,
                trained_model=model,
                return_eval_data=True,
            )

            miss_type = np.array(miss_type)

            target_miss_type_mask = miss_type == config.training.target_missing_type

            preds.append(results["predictions"][target_miss_type_mask])
            labels.append(results["labels"][target_miss_type_mask])

            if isinstance(cmam, DualCMAM):
                rec_embds.append(results["rec_embd_one"][target_miss_type_mask])
                rec_embds_two.append(results["rec_embd_two"][target_miss_type_mask])
                target_embds.append(results["target_embd_one"][target_miss_type_mask])
                target_embds_two.append(
                    results["target_embd_two"][target_miss_type_mask]
                )
                del results["rec_embd_one"]
                del results["rec_embd_two"]
                del results["target_embd_one"]
                del results["target_embd_two"]

            else:
                rec_embds.append(results["rec_embd"][target_miss_type_mask])
                target_embds.append(results["target_embd"][target_miss_type_mask])
                del results["rec_embd"]
                del results["target_embd"]
            # delete since they don't need to go to the metric recorder
            del results["predictions"]
            del results["labels"]

            epoch_rec_metric_recorder.update_from_dict(results)

        epoch_rec_metrics = epoch_rec_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                f"{config.logging.metrics_path.rsplit('.', 1)[0]}",
            ),
            epoch="test",
        )

        ## detach and convert to numpy
        preds = np.concatenate([p.detach().cpu().numpy() for p in preds], axis=0)
        labels = np.concatenate([l.detach().cpu().numpy() for l in labels], axis=0)

        if isinstance(cmam, DualCMAM):
            ## save the embeddings
            if isinstance(rec_embds[0], torch.Tensor):
                rec_embds = np.concatenate(
                    [r.detach().cpu().numpy() for r in rec_embds], axis=0
                )
                rec_embds_two = np.concatenate(
                    [r.detach().cpu().numpy() for r in rec_embds_two], axis=0
                )
                target_embds = np.concatenate(
                    [r.detach().cpu().numpy() for r in target_embds], axis=0
                )

                target_embds_two = np.concatenate(
                    [r.detach().cpu().numpy() for r in target_embds_two], axis=0
                )

                np.save(os.path.join(result_dir, "rec_embds_test.npy"), rec_embds)
                np.save(
                    os.path.join(result_dir, "rec_embds_two_test.npy"), rec_embds_two
                )
                np.save(os.path.join(result_dir, "target_embds_test.npy"), target_embds)
                np.save(
                    os.path.join(result_dir, "target_embds_two_test.npy"),
                    target_embds_two,
                )
            else:
                np.save(os.path.join(result_dir, "rec_embds_test.npy"), rec_embds)
                np.save(
                    os.path.join(result_dir, "rec_embds_two_test.npy"), rec_embds_two
                )
                np.save(os.path.join(result_dir, "target_embds_test.npy"), target_embds)
                np.save(
                    os.path.join(result_dir, "target_embds_two_test.npy"),
                    target_embds_two,
                )
        else:
            if isinstance(rec_embds[0], torch.Tensor):
                rec_embds = np.concatenate(
                    [r.detach().cpu().numpy() for r in rec_embds], axis=0
                )
                target_embds = np.concatenate(
                    [r.detach().cpu().numpy() for r in target_embds], axis=0
                )

                np.save(os.path.join(result_dir, "rec_embds_test.npy"), rec_embds)
                np.save(os.path.join(result_dir, "target_embds_test.npy"), target_embds)
            else:
                np.save(os.path.join(result_dir, "rec_embds_test.npy"), rec_embds)
                np.save(os.path.join(result_dir, "target_embds_test.npy"), target_embds)

        # ## save them to disk
        np.save(os.path.join(result_dir, "predictions_test.npy"), preds)
        np.save(os.path.join(result_dir, "labels_test.npy"), labels)

        console.rule("Test Metrics")
        display_validation_metrics(
            metrics=epoch_rec_metrics,
            console=console,
        )
        ## start a completely independent process to generate the visualizations
        input_modalities = "-".join(
            [str(k) for k in config.cmam.input_encoder_info.keys()]
        )

        if hasattr(config.cmam, "target_modality"):
            target_modality = str(config.cmam.target_modality)
        else:
            target_modality = (
                str(config.cmam.target_modality_one).title()
                + "-"
                + str(config.cmam.target_modality_two).title()
            )

        dataset_name = config.data.datasets["test"].dataset

        title = (
            rf"{dataset_name.upper()} | {input_modalities} $\mapsto$ {target_modality}"
        )
        output_path = os.path.join(
            result_dir, f"tsne_{input_modalities}_{target_modality}"
        )

        if config.training.do_tsne:
            command = f"""python visualisation/tsne.py \
                --tsne_config_path "{prepare_path(f'configs/{dataset_name}_tsne_config.yaml')}" \
                --rec_labels "{prepare_path(os.path.join(result_dir, 'predictions_test.npy'))}" \
                --gt_labels "{prepare_path(os.path.join(result_dir, 'labels_test.npy'))}" \
                --rec_embeddings "{prepare_path(os.path.join(result_dir, 'rec_embds_test.npy'))}" \
                --gt_embeddings "{prepare_path(os.path.join(result_dir, 'target_embds_test.npy'))}" \
                --output "{prepare_path(output_path)}" \
                --title "{title}"  """
            command = command.replace("\n", "")

            console.print(f"Running t-SNE visualization with command: {command}")

            # Wrap the Python command with nohup and redirect output
            full_command = f"nohup {command} > /dev/null 2>&1 &"

            # Use shell=True to interpret the command string
            process = subprocess.Popen(full_command, shell=True)
            console.print("Started t-SNE visualization process with PID: ", process.pid)

        file_name = os.path.join(result_dir, "test_metrics.json")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    clean_checkpoints(
        os.path.dirname(config.logging.model_output_path), best_eval_epoch
    )


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
    config = CMAMConfig.load(config_path, run_id=run_id)
    console.print(
        config,
        style="bold green",
    )

    console.print("Config loaded successfully", style="bold green")
    console.print("Configuring logger...", style="bold blue")

    level = logging.DEBUG if config.experiment.debug else logging.INFO

    configure_logger(config)

    logger = get_logger()

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
