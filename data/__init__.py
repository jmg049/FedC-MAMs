from torch.utils.data import Dataset, DataLoader
from config import DataConfig
from data.cmu_mosei_miss_dataset import cmumoseimissdataset


def resolve_dataset_name(dataset_name: str) -> Dataset:
    try:
        match dataset_name.lower():
            case "mosi" | "mosei":
                return cmumoseimissdataset
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
