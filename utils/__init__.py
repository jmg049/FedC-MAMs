from collections import defaultdict
import os
import re
import subprocess
import sys
from typing import Any, Dict
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box
from rich.panel import Panel
from rich.columns import Columns
import numpy as np
from torch import Tensor
from numpy import ndarray
import torch
from .logger import get_logger, configure_logger, LoggerSingleton


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f}GB")


def de_device(x: Tensor | ndarray) -> ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def call_latex_to_image(
    latex_str,
    script_path="visualisation/latex.py",
    output_file="equation.png",
    display_image=True,
):
    """
    Calls the 'latex.py' script with the provided LaTeX string to generate an image.

    Args:
        latex_str (str): The LaTeX string to convert.
        script_path (str): Path to the 'latex.py' script.
        output_file (str): The filename to save the generated image.
        display_image (bool): Whether to display the image in the terminal.

    Returns:
        str: Path to the generated image file.

    Raises:
        RuntimeError: If the subprocess fails.
    """
    # Ensure the script_path exists
    if not os.path.isfile(script_path):
        raise RuntimeError(f"Script '{script_path}' does not exist.")

    # Build the command using the same Python interpreter
    cmd = [sys.executable, script_path, latex_str, "--output", output_file]

    try:
        # Run the subprocess
        result = subprocess.run(
            cmd,
            check=True,  # Raises CalledProcessError if returncode != 0
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Optionally, handle result.stdout if needed
        if result.stdout:
            print("Subprocess Output:", result.stdout)
        print("LaTeX image generated successfully.")

        if display_image:
            # Display the image using kitten icat
            display_cmd = ["kitten", "icat", output_file]
            display_result = subprocess.run(
                display_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if display_result.stdout:
                print("Display Output:", display_result.stdout)
            print("Image displayed in terminal.")

        return output_file
    except subprocess.CalledProcessError as e:
        # Handle errors from the subprocess
        error_message = (
            f"latex.py failed with return code {e.returncode}\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        )
        raise RuntimeError(error_message) from e
    except FileNotFoundError:
        # Handle the case where 'python' or 'latex.py' is not found
        raise RuntimeError(
            "Could not find 'python' executable or 'visualisation/latex.py' script."
        )
    except Exception as e:
        # Handle other exceptions
        raise RuntimeError(f"An unexpected error occurred: {e}") from e


def confusion_matrix_to_rich_table(
    confusion_matrix: np.ndarray, class_labels: list = None
):
    # Validate inputs
    if class_labels and len(class_labels) != confusion_matrix.shape[0]:
        raise ValueError(
            "The number of class labels must match the dimensions of the confusion matrix."
        )

    # Initialize Rich Console
    console = Console()

    # Define Table
    table = Table(title="Confusion Matrix", box=box.SIMPLE, row_styles=["dim", ""])

    # Add Header Row
    table.add_column("Predicted/Actual", justify="center")
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(confusion_matrix.shape[0])]

    for label in class_labels:
        table.add_column(label, justify="center")

    # Add Rows of Confusion Matrix
    for i, label in enumerate(class_labels):
        row = [label] + [
            str(confusion_matrix[i][j]) for j in range(confusion_matrix.shape[1])
        ]
        table.add_row(*row)

    # Print Table
    # console.print(table)
    return table


def display_training_metrics(metrics, console=None):
    if console is None:
        console = Console()
    # Extract the headers (keys) for the table
    metric_keys = list(metrics.keys())

    # Determine maximum number of columns based on console width
    max_columns = (
        console.width // 30
    )  # Adjust number of columns based on console width, assuming 30 chars per metric-value pair
    max_columns = max(
        1, max_columns // 2
    )  # Each metric-value pair occupies two columns

    # Initialize table
    table = Table(
        title="Training Metrics", show_header=True, header_style="bold magenta"
    )

    # Add column headers for each metric-value pair
    for i in range(max_columns):
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right", style="bold yellow")

    metric_pairs = []
    confusion_table = None
    # Prepare metric-value pairs
    for key in metric_keys:
        value = metrics[key]
        # Handle numpy data types and convert them to Python native types for display
        if isinstance(value, (np.generic, np.ndarray)):
            if isinstance(value, np.ndarray):
                # Display confusion matrix in a better formatted way
                confusion_table = confusion_matrix_to_rich_table(value)
                continue
            else:
                value = value.item()  # Convert np.generic to Python scalar
        # Format float values to 4 decimal places
        if isinstance(value, float):
            value = f"{value:.4f}"
        metric_pairs.append((str(key), str(value)))

    # Add rows to the table
    for i in range(0, len(metric_pairs), max_columns):
        row = []
        for j in range(max_columns):
            if i + j < len(metric_pairs):
                row.extend(metric_pairs[i + j])
            else:
                row.extend(["", ""])
        table.add_row(*row)

    # Print the table to the console
    console.print(table)

    # Print the confusion matrices as panels
    if confusion_table:
        console.print(confusion_table)


def display_validation_metrics(metrics, console=None):
    if console is None:
        console = Console()
    # Group metrics by condition suffix (e.g., A, V, L, AV, etc.)
    grouped_metrics = defaultdict(dict)
    loss_metrics = {}
    for key, value in metrics.items():
        if key == "loss":
            loss_metrics[key] = value
            continue
        match = re.match(r"(.+?)_([A-Z]+)$", key)
        if match:
            metric_name, condition = match.groups()
            grouped_metrics[condition][metric_name] = value

    # Sort conditions based on availability (e.g., AVL > AV > A)
    sorted_conditions = sorted(grouped_metrics.keys(), key=lambda x: (-len(x), x))

    # Prepare tables for each condition
    tables = []
    confusion_tables = []
    for condition in sorted_conditions:
        condition_metrics = grouped_metrics[condition]
        table = Table(
            title=f"Validation Metrics - Condition: {condition}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right", style="bold yellow")

        # Prepare metric-value pairs
        for key, value in condition_metrics.items():
            # Handle numpy data types and convert them to Python native types for display
            if isinstance(value, (np.generic, np.ndarray)):
                if isinstance(value, np.ndarray):
                    # Display confusion matrix in a better formatted way
                    confusion_table = confusion_matrix_to_rich_table(value)
                    tables.append(confusion_table)
                    continue
                else:
                    value = value.item()  # Convert np.generic to Python scalar
            # Format float values to 4 decimal places
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(str(key), str(value))

        tables.append(table)

    # Prepare loss metrics table
    if loss_metrics:
        loss_table = Table(
            title="Loss Metrics", show_header=True, header_style="bold magenta"
        )
        loss_table.add_column("Metric", style="bold cyan")
        loss_table.add_column("Value", justify="right", style="bold yellow")
        for key, value in loss_metrics.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            loss_table.add_row(str(key), str(value))
        tables.insert(0, loss_table)

    # Print tables side by side
    console.print(Columns(tables))


def clean_checkpoints(checkpoints_dir, store_epoch, print_fn=print) -> None | Any:
    to_store = None

    if not os.path.exists(checkpoints_dir):
        return None

    for checkpoint in os.listdir(checkpoints_dir):
        if checkpoint.endswith(".pth"):
            if checkpoint.endswith("best.pth"):
                continue
            try:
                if int(checkpoint.split("_")[-1].split(".")[0]) != store_epoch:
                    os.remove(os.path.join(checkpoints_dir, checkpoint))
                    print_fn(f"Removing {checkpoint}")
                else:
                    to_store = os.path.join(checkpoints_dir, checkpoint)
                    print_fn(f"Keeping {checkpoint}")
            except ValueError as _:
                # remove any file that does not have epoch number
                os.remove(os.path.join(checkpoints_dir, checkpoint))
                print_fn(f"Removing {checkpoint}")
    return to_store


def prepare_path(path):
    return os.path.abspath(path)


# if __name__ == "__main__":
#     example_metrics = {
#         "Accuracy": 0.5288,
#         "F1_Micro": 0.5288,
#         "F1_Macro": 0.3664,
#         "F1_Weighted": 0.4519,
#         "UAR": 0.4205,
#         "Precision_Macro": 0.3333,
#         "Recall_Macro": 0.4205,
#         "Precision_Weighted": 0.4031,
#         "Recall_Weighted": 0.5288,
#         "Precision_Micro": 0.5288,
#         "Recall_Micro": 0.5288,
#         "NonZeroAcc": 0.5893,
#         "NonZeroF1": 0.6131,
#         "HasZeroAcc": 0.6980,
#         "HasZeroF1": 0.6985,
#         "Accuracy_AVL": 0.5502,
#         "F1_Micro_AVL": 0.5502,
#         "F1_Macro_AVL": 0.3287,
#         "F1_Weighted_AVL": 0.4361,
#         "UAR_AVL": 0.4108,
#         "Precision_Macro_AVL": 0.3840,
#         "Recall_Macro_AVL": 0.4108,
#         "Precision_Weighted_AVL": 0.4775,
#         "Recall_Weighted_AVL": 0.5502,
#         "Precision_Micro_AVL": 0.5502,
#         "Recall_Micro_AVL": 0.5502,
#         "NonZeroAcc_AVL": 0.5502,
#         "NonZeroF1_AVL": 0.6541,
#         "HasZeroAcc_AVL": 0.7869,
#         "HasZeroF1_AVL": 0.8436,
#         "Accuracy_A": 0.5400,
#         "F1_Micro_A": 0.5400,
#         "F1_Macro_A": 0.3716,
#         "F1_Weighted_A": 0.4782,
#         "UAR_A": 0.4366,
#         "loss": 0.9629,
#     }

#     print_metrics_tables(example_metrics)
