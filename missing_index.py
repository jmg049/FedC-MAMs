import numpy as np
import torch


def missing_pattern(num_modality, num_sample, ratio):
    missing_matrix = np.ones((num_sample, num_modality))
    for i in range(0, num_modality):
        missing_index = np.random.choice(
            np.arange(num_sample), replace=False, size=int(num_sample * ratio[i])
        )
        missing_matrix[missing_index, i] = 0

    missing_matrix = torch.tensor(missing_matrix)
    return missing_matrix


if __name__ == "__main__":
    a = missing_pattern(3, 10, [0.2, 0.5, 0.8])
    print(a)
