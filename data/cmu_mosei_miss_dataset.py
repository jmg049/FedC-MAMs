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
            self.missing_index = torch.tensor(
                [
                    [1, 0, 0],  # AZZ
                    [0, 1, 0],  # ZVZ
                    [0, 0, 1],  # ZZL
                    [1, 1, 0],  # AVZ
                    [1, 0, 1],  # AZL
                    [0, 1, 1],  # ZVL
                    [1, 1, 1],  # AVL
                ]
                * len(self.label)
            ).long()
            self.miss_type = ["azz", "zvz", "zzl", "avz", "azl", "zvl", "AVL"] * len(
                self.label
            )
        else:  # trn
            self.missing_index = [
                [1, 0, 0],  # AZZ
                [0, 1, 0],  # ZVZ
                [0, 0, 1],  # ZZL
                [1, 1, 0],  # AVZ
                [1, 0, 1],  # AZL
                [0, 1, 1],  # ZVL
            ]
            self.miss_type = ["azz", "zvz", "zzl", "avz", "azl", "zvl"]
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

    def __getitem__(self, index):
        if self.split != "train":
            feat_idx = index // 7  # totally 7 missing types
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

        # process A_feat
        A_feat = torch.from_numpy(self.all_A[feat_idx][()]).float()

        # process V_feat
        V_feat = torch.from_numpy(self.all_V[feat_idx][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[feat_idx][()]).float()

        match self.target_modality:
            case Modality.AUDIO:
                return {
                    "A_feat": A_feat,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.TEXT:
                return {
                    "L_feat": L_feat,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.VIDEO:
                return {
                    "V_feat": V_feat,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.MULTIMODAL:
                return {
                    "A_feat": A_feat,
                    "V_feat": V_feat,
                    "L_feat": L_feat,
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
        A = [sample["A_feat"] for sample in batch]
        V = [sample["V_feat"] for sample in batch]
        L = [sample["L_feat"] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample["label"] for sample in batch])
        missing_index = torch.cat(
            [sample["missing_index"].unsqueeze(0) for sample in batch], axis=0
        )
        miss_type = [sample["miss_type"] for sample in batch]

        match self.target_modality:
            case Modality.AUDIO:
                return {
                    "A_feat": A,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.TEXT:
                return {
                    "L_feat": L,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.VIDEO:
                return {
                    "V_feat": V,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.MULTIMODAL:
                return {
                    "A_feat": A,
                    "V_feat": V,
                    "L_feat": L,
                    "label": label,
                    "lengths": lengths,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
