from typing import Dict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def accuracy_3class(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy for 3-class classification."""
    y_pred_3 = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred_3)


def f1_score_3class(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate weighted F1-score for 3-class classification."""
    y_pred_3 = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred_3, average="weighted")


def accuracy_2class_has0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy for 2-class classification including neutral class."""
    y_pred_2 = np.argmax(np.array([[v[0], v[2]] for v in y_pred]), axis=1)
    y_true_2 = np.array([0 if v <= 1 else 1 for v in y_true])
    return accuracy_score(y_true_2, y_pred_2)


def f1_score_2class_has0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate weighted F1-score for 2-class classification including neutral class."""
    y_pred_2 = np.argmax(np.array([[v[0], v[2]] for v in y_pred]), axis=1)
    y_true_2 = np.array([0 if v <= 1 else 1 for v in y_true])
    return f1_score(y_true_2, y_pred_2, average="weighted")


def accuracy_2class_non0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy for 2-class classification excluding neutral class."""
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = np.argmax(y_pred[non_zeros][:, [0, 2]], axis=1)
    y_true_2 = y_true[non_zeros]
    return accuracy_score(y_true_2, y_pred_2)


def f1_score_2class_non0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate weighted F1-score for 2-class classification excluding neutral class."""
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = np.argmax(y_pred[non_zeros][:, [0, 2]], axis=1)
    y_true_2 = y_true[non_zeros]
    return f1_score(y_true_2, y_pred_2, average="weighted")


def round_dict_values(d: Dict[str, float], decimals: int = 4) -> Dict[str, float]:
    """Round all float values in a dictionary to a specified number of decimal places."""
    return {k: round(v, decimals) for k, v in d.items()}


def eval_mosi_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates model performance on MOSI (Multimodal Opinion Sentiment Intensity) dataset.

    This function performs three types of evaluations:
    1. Three-class classification: Negative (0), Neutral (1), Positive (2)
    2. Two-class classification (including neutral): Non-positive (<=0) vs Positive (>0)
    3. Two-class classification (excluding neutral): Negative (<0) vs Positive (>0)

    For each classification type, it calculates:
    - Accuracy
    - Weighted F1-score

    Parameters:
    y_pred: Model predictions as probabilities (after softmax)
    y_true: True labels

    Returns:
    Dictionary containing accuracies and F1-scores for each classification type.
    """

    results = {
        "Acc_3": accuracy_3class(y_true, y_pred),
        "F1_score_3": f1_score_3class(y_true, y_pred),
        "Has0_acc_2": accuracy_2class_has0(y_true, y_pred),
        "Has0_F1_score": f1_score_2class_has0(y_true, y_pred),
        "Non0_acc_2": accuracy_2class_non0(y_true, y_pred),
        "Non0_F1_score": f1_score_2class_non0(y_true, y_pred),
    }
    return round_dict_values(results)


## regression metrics


def cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute cosine similarity."""
    return np.dot(y_pred, y_true) / (np.linalg.norm(y_pred) * np.linalg.norm(y_true))


def multiclass_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param y_pred: Float array representing the predictions, dimension (N,)
    :param y_true: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


def clip_and_multiclass_acc(
    y_pred: np.ndarray, y_true: np.ndarray, min_val: float, max_val: float
) -> float:
    """Clip predictions and truth to a range and compute multiclass accuracy."""
    y_pred_clipped = np.clip(y_pred, a_min=min_val, a_max=max_val)
    y_true_clipped = np.clip(y_true, a_min=min_val, a_max=max_val)
    return multiclass_acc(y_pred_clipped, y_true_clipped)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.absolute(y_pred - y_true))


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(mse(y_pred, y_true))


def correlation(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute correlation coefficient."""
    return np.corrcoef(y_pred, y_true)[0][1]


def binary_accuracy_non_zero(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary accuracy for non-zero values."""
    test_preds = y_pred - 1
    test_truth = y_true - 1
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    if len(non_zeros) == 0:
        return 0.0
    try:
        non_zeros_binary_truth = test_truth[non_zeros] > 0
        non_zeros_binary_preds = test_preds[non_zeros] > 0
    except Exception as e:
        raise Exception(f"Error: {e} - {non_zeros} - {test_truth}")

    return accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)


def binary_f1_score_non_zero(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary F1 score for non-zero values."""
    test_preds = y_pred - 1
    test_truth = y_true - 1

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    if len(non_zeros) == 0:
        return 0.0
    non_zeros_binary_truth = test_truth[non_zeros] > 0
    non_zeros_binary_preds = test_preds[non_zeros] > 0
    return f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")


def binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary accuracy."""
    test_preds = y_pred - 1
    test_truth = y_true - 1

    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    return accuracy_score(binary_preds, binary_truth)


def binary_f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary F1 score."""
    test_preds = y_pred - 1
    test_truth = y_true - 1

    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    return f1_score(binary_truth, binary_preds, average="weighted")
