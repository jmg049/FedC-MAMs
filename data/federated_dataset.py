from collections import Counter
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, List, Literal
from scipy.stats import entropy
from tqdm import tqdm


from utils import get_logger

logger = get_logger()


class FederatedDataSplitter:

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        indices_load_dir: Optional[str] = None,
        sampling_strategy: Literal["stratified", "random", "non_iid"] = "stratified",
        alpha: float = 1.0,
        get_label_fn: Optional[Callable] = None,
        global_fraction: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        if random_seed is not None:
            np.random.seed(int(random_seed))
        self.dataset = dataset
        self.num_clients = num_clients
        self.sampling_strategy = sampling_strategy
        self.alpha = alpha
        self.get_label_fn = get_label_fn or (lambda x: x["label"])
        self.global_fraction = global_fraction

        # Only use core samples for splitting, ignoring missing types
        self.core_indices = np.arange(len(self.dataset.data))
        self.client_indices = {}
        self.global_indices = None
        self.indices_load_dir = indices_load_dir

        ## If indices_save_dir is provided, load indices from the directory, else partition and split the data (and save it)
        if self.indices_load_dir is not None:
            self.load_indices(self.indices_load_dir)
        else:
            self._partition_data()
            self._split_data()

    def _partition_data(self):
        num_global_samples = int(len(self.core_indices) * self.global_fraction)
        if num_global_samples > 0:
            self.global_indices = np.random.choice(
                self.core_indices, size=num_global_samples, replace=False
            )
            self.client_data_indices = np.setdiff1d(
                self.core_indices, self.global_indices
            )
        else:
            self.global_indices = np.array([])
            self.client_data_indices = self.core_indices.copy()

    def _split_data(self):
        if self.sampling_strategy == "stratified":
            self._stratified_split()
        elif self.sampling_strategy == "random":
            self._random_split()
        elif self.sampling_strategy == "non_iid":
            self._non_iid_split()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    # Existing splitting methods (_stratified_split, _random_split, _non_iid_split)

    def save_indices(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        # Save global indices
        if self.global_indices is not None:
            global_indices_path = os.path.join(save_dir, "global_indices.npy")
            np.save(global_indices_path, self.global_indices)
            logger.info(f"Global indices saved to {global_indices_path}")
        # Save client indices
        for client_id, indices in self.client_indices.items():
            client_indices_path = os.path.join(
                save_dir, f"client_{client_id}_indices.npy"
            )
            np.save(client_indices_path, indices)
            logger.info(f"Client {client_id} indices saved to {client_indices_path}")

    def load_indices(self, load_dir: str):
        # Load global indices
        global_indices_path = os.path.join(load_dir, "global_indices.npy")
        if os.path.exists(global_indices_path):
            self.global_indices = np.load(global_indices_path)
            logger.info(f"Global indices loaded from {global_indices_path}")
        else:
            raise FileNotFoundError(
                f"Global indices not found at {global_indices_path}"
            )
        # Load client indices
        self.client_indices = {}
        for client_id in range(self.num_clients):
            client_indices_path = os.path.join(
                load_dir, f"client_{client_id}_indices.npy"
            )
            if os.path.exists(client_indices_path):
                self.client_indices[client_id] = np.load(client_indices_path)
            else:
                raise FileNotFoundError(
                    f"Client indices not found at {client_indices_path}"
                )
            logger.info(f"Client {client_id} indices loaded from {client_indices_path}")

        print("Successfully loaded indices")

    def _split_data(self):
        if self.sampling_strategy == "stratified":
            self._stratified_split()
        elif self.sampling_strategy == "random":
            self._random_split()
        elif self.sampling_strategy == "non_iid":
            self._non_iid_split()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _stratified_split(self):
        labels = [self.get_label_fn(self.dataset[i]) for i in self.core_indices]
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        self.client_indices = {client_id: [] for client_id in range(self.num_clients)}

        for label, indices in label_indices.items():
            np.random.shuffle(indices)
            split_indices = np.array_split(indices, self.num_clients)
            for client_id, split in enumerate(split_indices):
                self.client_indices[client_id].extend(split)

        for client_id in self.client_indices:
            np.random.shuffle(self.client_indices[client_id])
            self.client_indices[client_id] = np.array(self.client_indices[client_id])

    def _random_split(self):
        shuffled_indices = self.client_data_indices.copy()
        np.random.shuffle(shuffled_indices)
        split_indices = np.array_split(shuffled_indices, self.num_clients)

        for client_id in range(self.num_clients):
            self.client_indices[client_id] = split_indices[client_id]

    def _non_iid_split(self):
        labels = [self.get_label_fn(self.dataset[i]) for i in self.client_data_indices]
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        class_indices = {
            label: self.client_data_indices[np.where(labels == label)[0]]
            for label in unique_labels
        }
        client_class_indices = {client_id: [] for client_id in range(self.num_clients)}

        for label in unique_labels:
            indices = class_indices[label]
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (proportions * len(indices)).astype(int)

            diff = len(indices) - np.sum(proportions)
            for i in range(abs(diff)):
                idx = i % self.num_clients
                proportions[idx] += np.sign(diff)

            start = 0
            for client_id, count in enumerate(proportions):
                end = start + count
                client_class_indices[client_id].extend(indices[start:end])
                start = end

        for client_id in range(self.num_clients):
            self.client_indices[client_id] = np.array(client_class_indices[client_id])

    def get_client_indices(self, client_id: int) -> np.array:
        return self.client_indices[client_id]

    def get_global_indices(self) -> np.array:
        return self.global_indices

    def calculate_iid_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics to measure the level of IID in the dataset split.

        Returns a dictionary with the following metrics:
        - label_distribution_divergence: Average Jensen-Shannon divergence of client
          label distributions from the overall distribution.
        - client_data_size_variance: Normalized variance in client data sizes.
        - average_kl_divergence: Average KL divergence of client label distributions
          from the overall distribution.

        Lower values indicate a more IID-like split. Higher values suggest more
        non-IID characteristics, which may lead to challenges in federated learning.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics.
        """
        # Compute overall label distribution
        all_labels = [self.get_label_fn(self.dataset[i]) for i in self.core_indices]
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        label_counts = np.bincount(all_labels, minlength=len(unique_labels))
        overall_distribution = label_counts / label_counts.sum()

        client_distributions = []
        client_sizes = []

        for client_id in range(self.num_clients):
            client_indices = self.get_client_indices(client_id)
            client_labels = [self.get_label_fn(self.dataset[i]) for i in client_indices]
            client_labels = np.array(client_labels)
            if client_labels.dtype != np.int64:
                client_labels = client_labels.astype(np.int64)
            client_label_counts = np.bincount(
                client_labels, minlength=len(unique_labels)
            )
            client_distribution = client_label_counts / client_label_counts.sum()
            client_distributions.append(client_distribution)
            client_sizes.append(len(client_indices))

        # 1. Label Distribution Divergence (using Jensen-Shannon divergence)
        js_divergences = [
            self._jensen_shannon_divergence(overall_distribution, cd)
            for cd in client_distributions
        ]
        avg_js_divergence = np.mean(js_divergences)

        # 2. Client Data Size Variance
        client_sizes = np.array(client_sizes)
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

    @staticmethod
    def _jensen_shannon_divergence(p: np.array, q: np.array) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        p = np.array(p)
        q = np.array(q)
        # Avoid division by zero and invalid values
        p = np.clip(p, 1e-12, 1)
        q = np.clip(q, 1e-12, 1)
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m)) + 0.5 * (entropy(q, m))


class FederatedDatasetWrapper(Dataset):
    """
    A wrapper class for creating federated datasets from any PyTorch Dataset.

    This class uses precomputed indices to provide the dataset subset for a specific client or global model.

    Args:
        dataset (Dataset): The base dataset.
        indices (np.array): Indices for this client's dataset subset.
        get_label_fn (Callable): A function to extract the label from a dataset sample.

    Usage:
        base_dataset = YourDataset(...)
        splitter = FederatedDataSplitter(
            dataset=base_dataset,
            num_clients=10,
            sampling_strategy="stratified",
            get_label_fn=lambda x: x[1],
            global_fraction=0.5  # Use 50% of data for global model
        )
        global_dataset = FederatedDatasetWrapper(
            dataset=base_dataset,
            indices=splitter.get_global_indices(),
            get_label_fn=lambda x: x[1]
        )
        client_datasets = []
        for client_id in range(10):
            client_indices = splitter.get_client_indices(client_id)
            client_dataset = FederatedDatasetWrapper(
                dataset=base_dataset,
                indices=client_indices,
                get_label_fn=lambda x: x[1]
            )
            client_datasets.append(client_dataset)
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: np.array,
        get_label_fn: Optional[Callable] = None,
    ) -> "FederatedDatasetWrapper":
        self.base_dataset = dataset
        self.indices = indices
        self.get_label_fn = get_label_fn or (lambda x: x[1])

    def __len__(self) -> int:
        if self.base_dataset.split == "train":
            return len(self.indices)
        else:
            return len(self.indices) * len(self.base_dataset.selected_missing_types)

    def __getitem__(self, idx) -> Any:
        if self.base_dataset.split == "train":
            return self.base_dataset[self.indices[idx]]
        else:
            base_idx = self.indices[
                idx // len(self.base_dataset.selected_missing_types)
            ]
            miss_type_idx = idx % len(self.base_dataset.selected_missing_types)
            return self.base_dataset[
                base_idx * len(self.base_dataset.selected_missing_types) + miss_type_idx
            ]

    # Other methods remain the same

    def get_labels(self) -> list:
        """
        Get all labels for this dataset subset.

        Returns:
            list: A list of labels for all samples in this subset.
        """
        return [self.get_label_fn(self.base_dataset[i]) for i in self.indices]

    def collate_fn(self, batch) -> Any:
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
