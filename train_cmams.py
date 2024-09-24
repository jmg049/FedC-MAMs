import argparse
import json
import os
import shlex
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
from utils import (
    call_latex_to_image,
    clean_checkpoints,
    prepare_path,
    print_all_metrics_tables,
)
from utils.logger import get_logger
from utils.metric_recorder import MetricRecorder

add_modality("VIDEO")

console = Console()

# Set to False if your terminal does not support rendering an image
RENDER_LOSS_FN = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument("--run_id", type=int, default=-1, help="Run ID")
    args = parser.parse_args()

    try:
        config = CMAMConfig.load(args.config, args.run_id)
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
    run_id = config.experiment.run_id

    os.makedirs(result_dir, exist_ok=True)

    verbose = config.experiment.debug

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
    logger.info("The number of validation samples = %d" % len(val_dataset.dataset))
    if do_test:
        logger.info("The number of test samples = %d" % len(tst_dataset.dataset))
    console.print(f"Validation dataset {val_dataset.dataset}")

    total_iters = epochs * dataset_size

    console.print("Total iterations: ", total_iters)
    logger.info("Total iterations: %d" % total_iters)

    model_cls = resolve_model_name(config.model.name)
    model = model_cls(**config.model.kwargs)

    check_protocol_compliance(model, MultimodalModelProtocol)

    pretrained_path = config.model.pretrained_path.format(run_id=run_id)
    assert os.path.exists(
        pretrained_path
    ), f"Pretrained model not found at {pretrained_path}"
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    model.to(device)

    total_iters = 0  # the total number of training iterations

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
    target_modality = cmam_cfg.target_modality
    cmam_cls = resolve_model_name(cmam_cfg.name)
    # cmam_cfg.pop("name")  # safe since we just used it
    cmam = cmam_cls(
        input_encoder_info=input_encoder_info,
        **cmam_cfg.__dict__(),
    )
    check_protocol_compliance(cmam, CMAMProtocol)
    cmam.set_predictions_metric_recorder(MetricRecorder(config.prediction_metrics))
    cmam.set_rec_metric_recorder(MetricRecorder(config.rec_metrics))

    # kaiming_init(cmam)

    cmam.to(device)
    console.print("CMAM model created")
    logger.info(f"CMAM model created\n{cmam}")
    console.print(f"{cmam}")
    optimizer = config.get_optimizer(cmam)

    if config.training.scheduler is not None:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")

    criterion_kwargs = config.training.criterion_kwargs

    ## calculate x-dim and z-dim for MI loss

    x_dims = [
        config.cmam.input_encoder_info[k]["input_size"]
        for k in config.cmam.input_encoder_info.keys()
    ]
    z_dim = config.cmam.assoc_net_output_size

    criterion_kwargs["x_dims"] = x_dims
    criterion_kwargs["z_dim"] = z_dim

    cmam_criterion = config.get_criterion(config.training.criterion, criterion_kwargs)

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

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration

        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        # epoch_metric_recorder = model.metric_recorder.clone()
        epoch_rec_metric_recorder = cmam.rec_metric_recorder.clone()
        # epoch_cls_metric_recorder = cmam.predictions_metric_recorder.clone()

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

            results = cmam.train_step(
                batch=data,
                labels=data["label"],
                optimizer=optimizer,
                device=device,
                cmam_criterion=cmam_criterion,
                trained_model=model,
            )
            epoch_rec_metric_recorder.update_from_dict(results)
            # epoch_cls_metric_recorder.update_from_dict(results["cls_metrics"])
            iter_data_time = time.time()

        train_cm = epoch_rec_metric_recorder.get("ConfusionMatrix", default=None)
        if train_cm is not None:
            train_cm = np.sum(train_cm, axis=0)
            tqdm.write(str(train_cm))
            del epoch_rec_metric_recorder.results["ConfusionMatrix"]

        # train_metrics = epoch_metric_recorder.get_average_metrics()
        train_rec_metrics = epoch_rec_metric_recorder.get_average_metrics()
        # train_cls_metrics = epoch_cls_metric_recorder.get_average_metrics()

        console.rule("Training Metrics")
        print_all_metrics_tables(
            metrics=train_rec_metrics,
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

        cmam.reset_metric_recorders()
        epoch_rec_metric_recorder = cmam.rec_metric_recorder.clone()
        # epoch_cls_metric_recorder = cmam.predictions_metric_recorder.clone()
        for batch in tqdm(
            val_dataset,
            desc="Validation",
            unit="Batches",
            colour="blue",
            total=len(val_dataset),
        ):
            # batch["missing_index"] = mp[len(data) * i : len(data) * (i + 1), :]
            results = cmam.evaluate(
                batch=batch,
                labels=batch["label"],
                device=device,
                cmam_criterion=cmam_criterion,
                trained_model=model,
            )
            epoch_rec_metric_recorder.update_from_dict(results)

        validation_cm = epoch_rec_metric_recorder.get("ConfusionMatrix", default=None)
        if validation_cm is not None:
            validation_cm = np.sum(validation_cm, axis=0)
            tqdm.write(str(validation_cm))
            del epoch_rec_metric_recorder.results["ConfusionMatrix"]

        validation_rec_metrics = epoch_rec_metric_recorder.get_average_metrics()
        console.rule("Validation Metrics")

        print_all_metrics_tables(
            metrics=validation_rec_metrics,
            console=console,
            max_cols_per_row=15,
            max_width=15,
        )

        #########################################################################################

        # show test result for debugging
        if do_test and verbose and epoch % 5 == 0:
            cmam.reset_metric_recorders()

            epoch_rec_metric_recorder = cmam.rec_metric_recorder.clone()

            for batch in tqdm(tst_dataset, desc="Test", unit="Batches", colour="green"):
                results = cmam.evaluate(
                    batch=batch,
                    labels=batch["label"],
                    cmam_criterion=cmam_criterion,
                    device=device,
                    trained_model=model,
                )

                epoch_rec_metric_recorder.update_from_dict(results)
            test_cm = epoch_rec_metric_recorder.get("ConfusionMatrix", default=None)
            if test_cm is not None:
                test_cm = np.sum(test_cm, axis=0)
                tqdm.write(str(test_cm))
                del epoch_rec_metric_recorder.results["ConfusionMatrix"]

            epoch_rec_metrics = epoch_rec_metric_recorder.get_average_metrics(
                save_to=os.path.join(
                    f"{config.logging.metrics_path.rsplit('.', 1)[0]}_rec",
                    "cmam_validation",
                ),
                epoch=epoch,
            )

            console.rule("Test Metrics")
            print_all_metrics_tables(
                metrics=epoch_rec_metrics,
                console=console,
                max_cols_per_row=15,
                max_width=15,
            )

        # record epoch with best result as per the target metric

        if config.logging.save_metric == "loss":
            if validation_rec_metrics["loss"] < best_eval_loss:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_eval_loss = validation_rec_metrics["loss"]
                best_metrics = validation_rec_metrics
        else:
            target_metric = validation_rec_metrics[config.logging.save_metric]
            if target_metric > best_metrics[config.logging.save_metric]:
                console.print(f"New best model found at epoch {epoch}")
                best_eval_epoch = epoch
                best_metrics = validation_rec_metrics

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
    console.print("Training complete")
    console.print(
        f"Best eval epoch was {best_eval_epoch} with a {config.logging.save_metric} of {best_metrics[config.logging.save_metric]}"
    )
    logger.info(
        f"Best eval epoch was {best_eval_epoch} with a {config.logging.save_metric} of {best_metrics[config.logging.save_metric]}"
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
        epoch_rec_metric_recorder = cmam.rec_metric_recorder.clone()
        preds = []
        labels = []
        rec_embds = []
        target_embds = []
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
            rec_embds.append(results["rec_embd"][target_miss_type_mask])
            target_embds.append(results["target_embd"][target_miss_type_mask])
            # delete since they don't need to go to the metric recorder
            del results["predictions"]
            del results["labels"]
            del results["rec_embd"]
            del results["target_embd"]
            epoch_rec_metric_recorder.update_from_dict(results)
        test_cm = epoch_rec_metric_recorder.get("ConfusionMatrix", default=None)
        if test_cm is not None:
            test_cm = np.sum(test_cm, axis=0)
            tqdm.write(str(test_cm))
            del epoch_rec_metric_recorder.results["ConfusionMatrix"]

            ## save the confusion matrix
            np.save(
                os.path.join(result_dir, "confusion_matrix_test.npy"),
                test_cm,
            )

        epoch_rec_metrics = epoch_rec_metric_recorder.get_average_metrics(
            save_to=os.path.join(
                f"{config.logging.metrics_path.rsplit('.', 1)[0]}",
                "cmam_validation",
            ),
            epoch="test",
        )

        ## detach and convert to numpy
        preds = np.concatenate([p.detach().cpu().numpy() for p in preds], axis=0)
        labels = np.concatenate([l.detach().cpu().numpy() for l in labels], axis=0)
        rec_embds = np.concatenate(
            [r.detach().cpu().numpy() for r in rec_embds], axis=0
        )
        target_embds = np.concatenate(
            [t.detach().cpu().numpy() for t in target_embds], axis=0
        )
        ## save them to disk
        np.save(os.path.join(result_dir, "predictions_test.npy"), preds)
        np.save(os.path.join(result_dir, "labels_test.npy"), labels)
        np.save(os.path.join(result_dir, "rec_embds_test.npy"), rec_embds)
        np.save(
            os.path.join(result_dir, "target_embds_test.npy"),
            target_embds,
        )

        ## start a completely independent process to generate the visualizations
        input_modalities = "-".join(
            [str(k) for k in config.cmam.input_encoder_info.keys()]
        )
        target_modality = str(config.cmam.target_modality)

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
            console.print(f"Running t-SNE visualization with command: {command}")

            # Wrap the Python command with nohup and redirect output
            full_command = f"nohup {command} > /dev/null 2>&1 &"

            # Use shell=True to interpret the command string
            process = subprocess.Popen(full_command, shell=True)
            console.print("Started t-SNE visualization process with PID: ", process.pid)

        console.rule("Test Metrics")
        print_all_metrics_tables(
            metrics=epoch_rec_metrics,
            console=console,
            max_cols_per_row=15,
            max_width=15,
        )

        file_name = os.path.join(result_dir, "test_metrics.json")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        epoch_rec_metrics = {k: float(v) for k, v in epoch_rec_metrics.items()}

        with open(file_name, "w") as f:
            json_str = json.dumps(epoch_rec_metrics, indent=4)
            f.write(json_str)

    clean_checkpoints(
        os.path.dirname(config.logging.model_output_path), best_eval_epoch
    )
