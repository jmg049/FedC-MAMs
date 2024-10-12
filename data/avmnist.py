from matplotlib import cm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from modalities import Modality

from PIL import Image
from torchvision.transforms.v2 import ToDtype, PILToTensor


class AVMNISTDataSet(Dataset):
    NUM_CLASSES = 10
    MASK_LOOKUP = {
        0: "ai",
        1: "az",
        2: "zi",
    }

    INDEX_LOOKUP = {
        "ai": "1,1",
        "az": "1,0",
        "zi": "0,1",
    }

    MODALITY_LOOKUP = {
        "ai": Modality.MULTIMODAL,
        "az": Modality.AUDIO,
        "zi": Modality.IMAGE,
    }

    FULL_CONDITION = "ai"

    @classmethod
    def get_modality(cls, miss_type: str) -> Modality:
        return cls.MODALITY_LOOKUP[miss_type]

    @classmethod
    def get_missing_types(cls) -> list[str]:
        return list(cls.INDEX_LOOKUP.keys())

    def __init__(
        self,
        data_fp: str,
        split: str,
        target_modality: Modality = Modality.MULTIMODAL,
        audio_column: str = "audio",
        image_column: str = "image",
        labels_column: str = "label",
        selected_missing_types: list[str] = None,
        split_indices: list[int] = None,
        **kwargs,
    ):
        super(AVMNISTDataSet, self).__init__()

        assert split in [
            "train",
            "valid",
            "test",
        ], f"Invalid split: {split}, must be one of ['train', 'valid', 'test']"
        self.split = split
        self.data = pd.read_csv(data_fp)
        if split_indices is not None:
            self.data = self.data.iloc[split_indices]
        self.audio_column = audio_column
        self.image_column = image_column
        self.labels_column = labels_column

        self.pil_to_tensor = PILToTensor()
        self.scale = ToDtype(torch.float32, True)

        if selected_missing_types is None:
            selected_missing_types = ["ai"]
        self.selected_missing_types = selected_missing_types

        if not isinstance(target_modality, Modality):
            target_modality = Modality.from_str(target_modality)
        assert target_modality in [
            Modality.AUDIO,
            Modality.IMAGE,
            Modality.MULTIMODAL,
        ], f"Invalid target_modality: {target_modality}"

        self.target_modality = target_modality

    def __len__(self):
        return len(self.data) * len(self.selected_missing_types)

    def __getitem__(self, index):
        data_index = index // len(self.selected_missing_types)
        miss_type_index = index % len(self.selected_missing_types)
        miss_type = self.selected_missing_types[miss_type_index]

        label = torch.tensor(self.data.iloc[data_index][self.labels_column])

        A = self.data.iloc[data_index][self.audio_column]
        A = torch.load(A, weights_only=True)

        I = self.data.iloc[data_index][self.image_column]
        I = np.array(torch.load(I, weights_only=False))
        I = Image.fromarray(np.uint8(cm.gist_earth(I) * 255)).convert("L")
        I = self.pil_to_tensor(I)
        I = self.scale(I)

        missing_index = torch.tensor(
            [int(i) for i in self.INDEX_LOOKUP[miss_type].split(",")]
        ).long()

        result = {
            "label": label,
            "missing_index": missing_index,
            "miss_type": miss_type,
        }

        if self.target_modality in [Modality.AUDIO, Modality.MULTIMODAL]:
            result[Modality.AUDIO] = A
        if self.target_modality in [Modality.IMAGE, Modality.MULTIMODAL]:
            result[Modality.IMAGE] = I

        return result
