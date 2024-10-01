from matplotlib import cm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from modalities import Modality

from PIL import Image
from torchvision.transforms.v2 import ToDtype, PILToTensor

MASK_LOOKUP = {
    0: "ai",
    1: "az",
    2: "zi",
}


class AVMNISTDataSet(Dataset):
    NUM_CLASSES = 10

    def __init__(
        self,
        data_fp: str,
        split: str,
        target_modality: Modality | str = Modality.MULTIMODAL,
        audio_column: str = "audio",
        image_column: str = "image",
        labels_column: str = "label",
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
        self.modality = target_modality
        self.audio_column = audio_column
        self.image_column = image_column
        self.labels_column = labels_column

        assert (
            self.audio_column in self.data.columns
        ), f"{self.audio_column} not in data columns"
        assert (
            self.image_column in self.data.columns
        ), f"{self.image_column} not in data columns"
        assert (
            self.labels_column in self.data.columns
        ), f"{self.labels_column} not in data columns"

        self.pil_to_tensor = PILToTensor()
        self.scale = ToDtype(torch.float32, True)

        if split != "train":
            self.missing_index = torch.tensor(
                [
                    [1, 0],  # AZ
                    [0, 1],  # ZI
                    [1, 1],  # AI
                ]
                * len(self.data)
            ).long()
            self.miss_type = ["az", "zi", "ai"] * len(self.data)
        else:
            self.missing_index = [
                [1, 0],  # AZ
                [0, 1],  # ZI
                [1, 1],  # AI
            ]

            self.miss_type = ["az", "zi", "ai"]

        if not isinstance(target_modality, Modality):
            modality = Modality.from_str(target_modality)
        assert modality in [
            Modality.AUDIO,
            Modality.IMAGE,
            Modality.MULTIMODAL,
        ], f"Invalid modality: {modality}, must be one of [{Modality.AUDIO}, {Modality.IMAGE}, {Modality.MULTIMODAL}]"

        self.target_modality = modality

    def __len__(self):
        return len(self.missing_index) if self.split != "train" else len(self.data)

    def __getitem__(self, index):
        if self.split != "train":
            feat_idx = index // 3  # totally 3 missing types
            missing_index = self.missing_index[index]
            miss_type = self.miss_type[index]
        else:
            feat_idx = index
            _miss_i = np.random.choice(list(range(len(self.missing_index))))
            missing_index = (
                torch.tensor(self.missing_index[_miss_i]).clone().detach().long()
            )
            miss_type = self.miss_type[_miss_i]
        label = torch.tensor(self.data.iloc[feat_idx][self.labels_column])

        A = self.data.iloc[feat_idx][self.audio_column]
        A = torch.load(A, weights_only=True)

        I = self.data.iloc[feat_idx][self.image_column]
        I = np.array(torch.load(I, weights_only=False))
        I = Image.fromarray(np.uint8(cm.gist_earth(I) * 255)).convert("L")
        I = self.pil_to_tensor(I)
        I = self.scale(I)

        match self.target_modality:
            case Modality.AUDIO:
                return {
                    "Audio": A,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.IMAGE:
                return {
                    "Image": I,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }
            case Modality.MULTIMODAL:
                return {
                    "A": A,
                    "I": I,
                    "label": label,
                    "missing_index": missing_index,
                    "miss_type": miss_type,
                }


# if __name__ == "__main__":
#     dataset = AVMNISTDataSet("./AVMNIST/dataset/test.csv", modality=Modality.MULTIMODAL)
#     audio, image, label = dataset[0]
#     print(image.shape)
