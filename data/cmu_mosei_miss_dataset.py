import os
from pathlib import Path
import random
from typing import Union
import torch
import numpy as np
from numpy import int64
from torch.nn.utils.rnn import pad_sequence
import pickle
from modalities import Modality
from torch.utils.data import Dataset

MASK_LOOKUP = {
    0: "avt",
    1: "azz",
    2: "zvz",
    3: "zzt",
    4: "avz",
    5: "azt",
    6: "zvt",
}

INDEX_LOOKUP = {
    "avt": "1,1,1",
    "azz": "1,0,0",
    "zvz": "0,1,0",
    "zzt": "0,0,1",
    "avz": "1,1,0",
    "azt": "1,0,1",
    "zvt": "0,1,1",
}


class cmumoseimissdataset(Dataset):
    @staticmethod
    def get_num_classes(is_classification: bool = True):
        return 3 if is_classification else 1

    def __init__(
        self,
        data_fp: Union[str, Path, os.PathLike],
        split: str,
        target_modality: Modality = Modality.MULTIMODAL,
        labels_key: str = "classification_labels",
        selected_missing_types: list[str] = None,  # New parameter
        **kwargs,
    ):
        super().__init__()
        assert split in [
            "train",
            "valid",
            "test",
        ], f"Invalid split: {split}, must be one of ['train', 'valid', 'test']"
        self.split = split
        self.data_fp = data_fp
        with open(data_fp, "rb") as f:
            data = pickle.load(f)

        assert labels_key in data[split], f"Invalid labels_key: {labels_key}"

        data_split = data[split]
        self.all_A = data_split["audio"]
        self.all_V = data_split["vision"]
        self.all_L = data_split["text"]
        self.label = data_split[labels_key]
        self.label = np.array(self.label, dtype=int64)

        if split != "train":  # val && tst
            if selected_missing_types is None:
                selected_missing_types = list(INDEX_LOOKUP.keys())
            valid_missing_indices = [
                INDEX_LOOKUP[miss_type] for miss_type in selected_missing_types
            ]
            valid_missing_indices = [
                [int(i) for i in m_index.split(",")]
                for m_index in valid_missing_indices
            ]

            self.selected_missing_types = selected_missing_types

            self.missing_index = torch.tensor(
                valid_missing_indices * len(self.label)
            ).long()
            self.miss_type = selected_missing_types * len(self.label)

        else:  # trn
            self.missing_index = [
                [1, 0, 0],  # AZZ
                [0, 1, 0],  # ZVZ
                [0, 0, 1],  # ZZL
                [1, 1, 0],  # AVZ
                [1, 0, 1],  # AZL
                [0, 1, 1],  # ZVL
            ]
            self.miss_type = list(INDEX_LOOKUP.keys())
        if not isinstance(target_modality, Modality):
            target_modality = Modality.from_str(target_modality)
        assert target_modality in [
            Modality.AUDIO,
            Modality.TEXT,
            Modality.VIDEO,
            Modality.MULTIMODAL,
        ], f"Invalid target_modality: {target_modality}, must be one of [{Modality.AUDIO}, {Modality.TEXT}, {Modality.VIDEO}, {Modality.MULTIMODAL}]"
        self.target_modality = target_modality
        # set collate function
        self.manual_collate_fn = True

    def __str__(self) -> str:
        return f"CMU-MOSEI Missing Dataset ({self.split}) - Target Modality: {self.target_modality} - {len(self)} samples"

    def __getitem__(self, index):
        if self.split != "train":
            feat_idx = index // len(self.selected_missing_types)
            missing_index = self.missing_index[index]
            miss_type = self.miss_type[index]
        else:
            feat_idx = index
            _miss_i = random.choice(list(range(len(self.missing_index))))

            missing_index = (
                torch.tensor(self.missing_index[_miss_i]).clone().detach().long()
            )

            miss_type = self.miss_type[_miss_i]
        label = torch.tensor(self.label[feat_idx])

        A = torch.from_numpy(self.all_A[feat_idx][()]).float()
        V = torch.from_numpy(self.all_V[feat_idx][()]).float()
        T = torch.from_numpy(self.all_L[feat_idx][()]).float()

        match self.target_modality:
            case Modality.AUDIO:
                return {
                    Modality.AUDIO: A,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.TEXT:
                return {
                    Modality.TEXT: T,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.VIDEO:
                return {
                    Modality.VIDEO: V,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.MULTIMODAL:
                return {
                    Modality.AUDIO: A,
                    Modality.VIDEO: V,
                    Modality.TEXT: T,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }

    def __len__(self):
        return len(self.missing_index) if self.split != "train" else len(self.label)

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std

    def collate_fn(self, batch):
        A = [sample["A"] for sample in batch]
        V = [sample["V"] for sample in batch]
        T = [sample["T"] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        T = pad_sequence(T, batch_first=True, padding_value=0)
        label = torch.tensor([sample["label"] for sample in batch])
        missing_index = torch.cat(
            [sample["missing_index"].unsqueeze(0) for sample in batch], axis=0
        )
        miss_type = [sample["miss_type"] for sample in batch]

        match self.target_modality:
            case Modality.AUDIO:
                return {
                    Modality.AUDIO: A,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.TEXT:
                return {
                    Modality.TEXT: T,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.VIDEO:
                return {
                    Modality.VIDEO: V,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.MULTIMODAL:
                return {
                    Modality.AUDIO: A,
                    Modality.VIDEO: V,
                    Modality.TEXT: T,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
