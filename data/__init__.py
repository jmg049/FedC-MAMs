from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Literal

import numpy as np
from scipy.stats import entropy
from torch.utils.data import DataLoader, Dataset

from config import DataConfig
from data.avmnist import AVMNISTDataSet
from data.cmu_mosei_miss_dataset import cmumoseimissdataset


class FederatedDatasetWrapper(Dataset):
    """
    A wrapper class for creating federated datasets from any PyTorch Dataset.

    This class splits a given dataset into subsets for federated learning scenarios.
    It supports various sampling strategies including stratified, random, IID, and non-IID data distribution.

    Args:
        dataset (Dataset): The base dataset to be split.
        client_id (int): The ID of the current client.
        num_clients (int): The total number of clients in the federated setup.
        sampling_strategy (str): The strategy for splitting the data. Options are
                                 "stratified", "random", "iid", or "non_iid".
        alpha (float): The alpha parameter for the Dirichlet distribution in non-IID sampling.
        get_label_fn (Callable): A function to extract the label from a dataset sample.

    Usage:
        base_dataset = YourDataset(...)
        fed_dataset = GenericFederatedDatasetWrapper(
            dataset=base_dataset,
            client_id=0,
            num_clients=10,
            sampling_strategy="stratified",
            get_label_fn=lambda x: x[1]  # Adjust based on your dataset structure (see label_functions.py for examples)
        )

    Note:
        The get_label_fn should be adjusted based on the structure of your dataset's samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        client_id: int,
        num_clients: int,
        sampling_strategy: Literal[
            "stratified", "random", "iid", "non_iid"
        ] = "stratified",
        alpha: float = 1.0,
        get_label_fn: Callable = None,
    ) -> "FederatedDatasetWrapper":
        self.base_dataset = dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.sampling_strategy = sampling_strategy
        self.alpha = alpha
        self.get_label_fn = get_label_fn or (
            lambda x: x[1]
        )  # Default assumes (data, label) format

        self.indices = self._split_data()

    def _split_data(self) -> Any:
        """
        Split the data according to the specified sampling strategy.

        Returns:
            np.array: An array of indices representing this client's subset of the data.
        """
        if self.sampling_strategy == "stratified":
            return self._stratified_split()
        elif self.sampling_strategy == "random":
            return self._random_split()
        elif self.sampling_strategy == "iid":
            return self._iid_split()
        elif self.sampling_strategy == "non_iid":
            return self._non_iid_split()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _stratified_split(self):
        """
        Perform a stratified split of the data.

        This method ensures that each client gets a proportional representation of each class.

        Returns:
            np.array: An array of indices for this client's stratified subset.
        """
        labels = [
            self.get_label_fn(self.base_dataset[i])
            for i in range(len(self.base_dataset))
        ]
        unique_labels = np.unique(labels)
        client_indices = []
        for label in unique_labels:
            label_indices = np.where(np.array(labels) == label)[0]
            client_label_indices = np.array_split(label_indices, self.num_clients)[
                self.client_id
            ]
            client_indices.extend(client_label_indices)
        return np.array(client_indices)

    def _random_split(self) -> Any:
        """
        Perform a random split of the data.

        This method randomly shuffles the data indices and then splits them among clients.

        Returns:
            np.array: An array of indices for this client's random subset.
        """
        all_indices = np.arange(len(self.base_dataset))
        np.random.shuffle(all_indices)
        return np.array_split(all_indices, self.num_clients)[self.client_id]

    def _iid_split(self) -> Any:
        """
        Perform an IID (Independent and Identically Distributed) split of the data.

        This method simply divides the dataset indices equally among clients.

        Returns:
            np.array: An array of indices for this client's IID subset.
        """
        return np.array_split(range(len(self.base_dataset)), self.num_clients)[
            self.client_id
        ]

    def _non_iid_split(self) -> Any:
        """
        Perform a non-IID split of the data.

        This method uses a Dirichlet distribution to create imbalanced class distributions across clients.

        Returns:
            np.array: An array of indices for this client's non-IID subset.
        """
        labels = [
            self.get_label_fn(self.base_dataset[i])
            for i in range(len(self.base_dataset))
        ]
        unique_labels = np.unique(labels)
        client_indices = []

        label_dirichlet = np.random.dirichlet([self.alpha] * len(unique_labels))
        for label, p in zip(unique_labels, label_dirichlet):
            label_indices = np.where(np.array(labels) == label)[0]
            label_client_sizes = np.random.multinomial(
                len(label_indices), [1 / self.num_clients] * self.num_clients
            )
            client_label_indices = np.array_split(
                label_indices, np.cumsum(label_client_sizes[:-1])
            )
            client_indices.extend(client_label_indices[self.client_id])

        return np.array(client_indices)

    def __len__(self) -> int:
        """
        Get the length of this client's dataset subset.

        Returns:
            int: The number of samples in this client's subset.
        """
        return len(self.indices)

    def __getitem__(self, idx) -> Any:
        """
        Get an item from this client's dataset subset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The dataset item at the specified index.
        """
        return self.base_dataset[self.indices[idx]]

    def get_labels(self) -> list:
        """
        Get all labels for this client's dataset subset.

        Returns:
            list: A list of labels for all samples in this client's subset.
        """
        return [self.get_label_fn(self.base_dataset[i]) for i in self.indices]

    def collate_fn(self, batch) -> Any | None:
        """
        Collate function for creating batches.

        If the base dataset has a collate_fn method, it will be used.
        Otherwise, the default PyTorch collate function will be used.

        Args:
            batch (list): A list of samples to be collated into a batch.

        Returns:
            Any: The collated batch.
        """
        if (
            hasattr(self.base_dataset, "collate_fn")
            and self.base_dataset.collate_fn is not None
        ):
            return self.base_dataset.collate_fn(batch)
        return None  # Default PyTorch collate_fn will be used

    def calculate_iid_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics to measure the level of IID in the dataset split.

        This method computes several metrics to quantify how close the data split
        is to being truly IID (Independent and Identically Distributed):

        1. Label Distribution Divergence: Average Jenson-Shannon divergence of client label
           distributions from the overall distribution.
        2. Client Data Size Variance: Normalized variance in the amount of data each client receives.
        3. Average KL Divergence: Average KL divergence of client label distributions
           from the overall distribution.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics.

        Note:
            Lower values for these metrics indicate a more IID-like split.
        """
        all_labels = [
            self.get_label_fn(self.base_dataset[i])
            for i in range(len(self.base_dataset))
        ]
        unique_labels = np.unique(all_labels)
        overall_distribution = np.bincount(
            all_labels, minlength=len(unique_labels)
        ) / len(all_labels)

        client_distributions = []
        client_sizes = []

        for client_id in range(self.num_clients):
            client_indices = self._get_client_indices(client_id)
            client_labels = [all_labels[i] for i in client_indices]
            client_distribution = np.bincount(
                client_labels, minlength=len(unique_labels)
            ) / len(client_labels)
            client_distributions.append(client_distribution)
            client_sizes.append(len(client_indices))

        # 1. Label Distribution Divergence (using Jensen-Shannon divergence)
        js_divergences = [
            self._jensen_shannon_divergence(overall_distribution, cd)
            for cd in client_distributions
        ]
        avg_js_divergence = np.mean(js_divergences)

        # 2. Client Data Size Variance
        client_size_variance = np.var(client_sizes) / (np.mean(client_sizes) ** 2)

        # 3. Average KL Divergence
        kl_divergences = [
            entropy(cd, overall_distribution) for cd in client_distributions
        ]
        avg_kl_divergence = np.mean(kl_divergences)

        return {
            "label_distribution_divergence": avg_js_divergence,
            "client_data_size_variance": client_size_variance,
            "average_kl_divergence": avg_kl_divergence,
        }

    def _get_client_indices(self, client_id: int) -> np.array:
        """Get indices for a specific client."""
        if self.sampling_strategy == "stratified":
            return self._stratified_split(client_id)
        elif self.sampling_strategy == "random":
            return self._random_split(client_id)
        elif self.sampling_strategy == "iid":
            return self._iid_split(client_id)
        elif self.sampling_strategy == "non_iid":
            return self._non_iid_split(client_id)

    @staticmethod
    def _jensen_shannon_divergence(p: np.array, q: np.array) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))


def create_federated_datasets(
    dataset_class: type,
    data_fp: str | Path | PathLike,
    client_id: int,
    num_clients: int,
    sampling_strategy: Literal["stratified", "random", "iid", "non_iid"] = "stratified",
    alpha: float = 1.0,
    get_label_fn: Callable = None,
    **dataset_kwargs,
):
    """
    Create federated datasets for train, validation, and test splits.

    Args:
        dataset_class (type): The class of the dataset to be used.
        data_fp (str): File path or other identifier for the dataset.
        client_id (int): The ID of the current client.
        num_clients (int): The total number of clients in the federated setup.
        sampling_strategy (str): The strategy for splitting the data.
        alpha (float): The alpha parameter for the Dirichlet distribution in non-IID sampling.
        get_label_fn (Callable): A function to extract the label from a dataset sample.
        **dataset_kwargs: Additional keyword arguments to be passed to the dataset class.

    Returns:
        tuple: A tuple containing the federated train, validation, and test datasets and their IID metrics.

    Usage:
        train_dataset, valid_dataset, test_dataset = create_federated_datasets(
            dataset_class=YourDatasetClass,
            data_fp="path/to/data",
            client_id=0,
            num_clients=10,
            sampling_strategy="stratified",
            get_label_fn=lambda x: x[1],
            custom_arg1=value1,
            custom_arg2=value2
        )
    """
    train_dataset = dataset_class(data_fp, split="train", **dataset_kwargs)
    valid_dataset = dataset_class(data_fp, split="valid", **dataset_kwargs)
    test_dataset = dataset_class(data_fp, split="test", **dataset_kwargs)

    fed_train_dataset = FederatedDatasetWrapper(
        train_dataset, client_id, num_clients, sampling_strategy, alpha, get_label_fn
    )
    fed_valid_dataset = FederatedDatasetWrapper(
        valid_dataset, client_id, num_clients, sampling_strategy, alpha, get_label_fn
    )
    fed_test_dataset = FederatedDatasetWrapper(
        test_dataset, client_id, num_clients, sampling_strategy, alpha, get_label_fn
    )

    train_iid_metrics = fed_train_dataset.calculate_iid_metrics()
    valid_iid_metrics = fed_valid_dataset.calculate_iid_metrics()
    test_iid_metrics = fed_test_dataset.calculate_iid_metrics()

    return (
        (fed_train_dataset, train_iid_metrics),
        (fed_valid_dataset, valid_iid_metrics),
        (fed_test_dataset, test_iid_metrics),
    )


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
