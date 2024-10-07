import os
import time
from typing import Any, Dict

from torch.utils.data import DataLoader, Dataset

from config import DataConfig

# from config.federated_config import FederatedDataConfig
from data.avmnist import AVMNISTDataSet
from data.cmu_mosei_miss_dataset import cmumoseimissdataset
from data.federated_dataset import FederatedDataSplitter, FederatedDatasetWrapper
from utils import get_logger

logger = get_logger()


def create_federated_datasets(
    num_clients: int,
    data_config,
    indices_save_dir: str = None,
    indices_load_dir: str = None,
):
    """
    Create federated datasets for train, validation, and test splits.

    Args:
        num_clients (int): Number of clients in the federated setup.
        data_config (FederatedDataConfig): Configuration for datasets and federated parameters.

    Returns:
        A tuple containing:
            - federated_train_dataset (Dict): Contains global and client datasets for training.
            - federated_valid_dataset (Dict): Contains global and client datasets for validation.
            - federated_test_dataset (Dict): Contains global and client datasets for testing.
            - iid_metrics (Dict): Contains IID metrics for each data split.
    """

    dataloader_specific_args = [
        "batch_size",
        "shuffle",
        "num_workers",
        "pin_memory",
        "drop_last",
    ]

    federated_specific_args = [
        "distribution_type",
        "sampling_strategy",
        "alpha",
        "min_samples_per_client",
        "global_fraction",
        "get_label_fn",
        "random_seed",
        "client_proportions",
    ]

    # Dictionaries to hold datasets and IID metrics
    federated_datasets = {}
    iid_metrics = {}

    sampling_strategy = data_config.sampling_strategy
    alpha = data_config.alpha
    global_fraction = data_config.global_fraction
    # _min_samples_per_client = data_config.min_samples_per_client
    get_label_fn = data_config.get_label_fn
    random_seed = time.time()
    client_proportions = data_config.client_proportions

    for data_split in data_config.datasets:
        dataset_config = data_config.datasets[data_split]
        dataset_cls = resolve_dataset_name(dataset_config.dataset)

        base_dataset_args = {
            k: v
            for k, v in dataset_config
            if k not in dataloader_specific_args + federated_specific_args + ["dataset"]
        }

        # Ensure only one miss type for training
        if data_split == "train":
            base_dataset_args["selected_missing_types"] = [
                base_dataset_args["selected_missing_types"][0]
            ]

        base_dataset = dataset_cls(**base_dataset_args)
        print(f"{dataset_cls.__name__} dataset {data_split} created")
        logger.info(f"{dataset_cls.__name__} dataset {data_split} created")
        # Save indices
        if indices_save_dir is not None:
            _indices_save_dir = os.path.join(
                indices_save_dir, f"dataset_indices_{data_split}"
            )

        if indices_load_dir is not None:

            _indices_load_dir = os.path.join(
                indices_load_dir, f"dataset_indices_{data_split}"
            )
        else:
            _indices_load_dir = None

        splitter = FederatedDataSplitter(
            dataset=base_dataset,
            num_clients=num_clients,
            sampling_strategy=sampling_strategy,
            alpha=alpha,
            get_label_fn=get_label_fn,
            global_fraction=global_fraction,
            random_seed=random_seed,
            indices_load_dir=_indices_load_dir,
        )
        print(f"FederatedDataSplitter created for {data_split}")
        logger.info(f"FederatedDataSplitter created for {data_split}")

        if indices_load_dir is None:
            splitter.save_indices(_indices_save_dir)

        # Create global and client datasets
        global_dataset = FederatedDatasetWrapper(
            dataset=base_dataset,
            indices=splitter.get_global_indices(),
            get_label_fn=get_label_fn,
        )

        client_datasets = []
        for client_id in range(num_clients):
            client_dataset = FederatedDatasetWrapper(
                dataset=base_dataset,
                indices=splitter.get_client_indices(client_id),
                get_label_fn=get_label_fn,
            )
            client_datasets.append(client_dataset)

        federated_datasets[data_split] = {
            "global": global_dataset,
            "clients": client_datasets,
        }

        # Calculate IID metrics
        metrics = splitter.calculate_iid_metrics()

        # Store the datasets and metrics
        iid_metrics[data_split] = metrics

    # Extract datasets and metrics for each split
    fed_train_dataset = federated_datasets.get("train", None)
    fed_valid_dataset = federated_datasets.get("validation", None)
    fed_test_dataset = federated_datasets.get("test", None)

    train_iid_metrics = iid_metrics.get("train", None)
    valid_iid_metrics = iid_metrics.get("validation", None)
    test_iid_metrics = iid_metrics.get("test", None)

    return (
        fed_train_dataset,
        train_iid_metrics,
        fed_valid_dataset,
        valid_iid_metrics,
        fed_test_dataset,
        test_iid_metrics,
    )


def create_federated_dataloaders(
    federated_datasets: Dict[str, Dict[str, Any]],
    data_config,
):
    """
    Create data loaders for federated datasets.

    Args:
        federated_datasets (Dict[str, Dict[str, Any]]): The federated datasets for each data split.
        data_config (FederatedDataConfig): Configuration for datasets and data loaders.

    Returns:
        A dictionary containing data loaders for each data split, with separate loaders for
        the global dataset and each client's dataset.
    """

    dataloader_specific_args = [
        "batch_size",
        "shuffle",
        "num_workers",
        "pin_memory",
        "drop_last",
    ]

    federated_dataloaders = {}

    for data_split in federated_datasets.keys():
        dataset_config = data_config.datasets[data_split]

        # Extract dataloader specific arguments
        dataloader_args = {
            k: v for k, v in dataset_config.items() if k in dataloader_specific_args
        }

        # Set default values for dataloader arguments if not provided
        dataloader_args.setdefault("batch_size", 128)
        dataloader_args.setdefault("shuffle", True if data_split == "train" else False)
        dataloader_args.setdefault("num_workers", 0)
        dataloader_args.setdefault("pin_memory", False)
        dataloader_args.setdefault("drop_last", False)

        # Create global data loader
        global_dataset = federated_datasets[data_split]["global"]
        global_dataloader = DataLoader(
            global_dataset,
            **dataloader_args,
        )

        # Create client data loaders
        client_datasets = federated_datasets[data_split]["clients"]
        client_dataloaders = []
        for client_dataset in client_datasets:
            client_dataloader = DataLoader(
                client_dataset,
                **dataloader_args,
            )
            client_dataloaders.append(client_dataloader)

        # Store the data loaders
        federated_dataloaders[data_split] = {
            "global": global_dataloader,
            "clients": client_dataloaders,
        }

    return federated_dataloaders


def resolve_dataset_name(dataset_name: str) -> Dataset:
    try:
        match dataset_name.lower():
            case "mosi" | "mosei":
                return cmumoseimissdataset
            case "avmnist":
                return AVMNISTDataSet
            case _:
                raise ValueError(f"Invalid dataset: {dataset_name}")
    except Exception as e:
        print(f"Error: {e} - {dataset_name}")
        raise e


def build_dataloader(
    data_config: DataConfig, target_split: str, batch_size: int, print_fn=print
):
    """
    Build a DataLoader for the specified split.

    Args:
        target_split (str): The split to build the DataLoader for ('train', 'validation', or 'test').

    Returns:
        torch.utils.data.DataLoader: The constructed DataLoader.
    """

    dataloader_specific_args = [
        "batch_size",
        "shuffle",
        "num_workers",
        "pin_memory",
        "drop_last",
    ]

    dataset_config = data_config.datasets[target_split]
    dataset_cls = resolve_dataset_name(dataset_config.dataset)

    dataloader_cfg = {
        k: v
        for k, v in dataset_config
        if k in dataloader_specific_args and k != "dataset"
    }

    dataset_cfg = {
        k: v
        for k, v in dataset_config
        if k not in dataloader_specific_args and k != "dataset"
    }

    dataset = dataset_cls(**dataset_cfg)
    print_fn(f"{dataset_cls.__name__} dataset {target_split} created")

    return DataLoader(
        dataset,
        **dataloader_cfg,
        batch_size=batch_size,
        collate_fn=dataset.collate if hasattr(dataset, "collate") else None,
    )
