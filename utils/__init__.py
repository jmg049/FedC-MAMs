import os
import subprocess
import sys
from typing import Any, Dict
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box

from .logger import get_logger, configure_logger, LoggerSingleton


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


def print_metrics_tables(
    metrics: Dict[str, Any],
    max_cols_per_row: int = 5,
    max_width: int = 20,
    console: Console = None,
    target_metric: str = None,
    generic_table: bool = False,
) -> None:
    """
    Print metrics tables with bordered formatting, grouped by conditions.

    :param metrics: Dictionary of metric names and values
    :param max_cols_per_row: Maximum number of columns per table row
    :param max_width: Maximum width for each column
    :param console: Rich Console object (if None, a new one will be created)
    """
    if console is None:
        console = Console()

    def get_condition(metric_name: str) -> str:
        parts = metric_name.split("_")
        return parts[-1] if len(parts) > 1 and parts[-1].isupper() else ""

    def sort_key(item):
        metric, _ = item
        condition = get_condition(metric)
        return (
            condition == "",  # No condition first
            -len(condition),  # Longer conditions before shorter ones
            condition,  # Alphabetical order for same-length conditions
            metric,  # Original order for metrics with the same condition
        )

    # Group metrics by condition
    grouped_metrics = {}
    for metric, value in sorted(metrics.items(), key=sort_key):
        condition = get_condition(metric)
        if condition:
            metric_name = "_".join(metric.split("_")[:-1])
        else:
            metric_name = metric
            condition = "No Condition" if not generic_table else ""

        if condition not in grouped_metrics:
            grouped_metrics[condition] = {}
        grouped_metrics[condition][metric_name] = value

    # Print tables for each condition
    for condition, condition_metrics in grouped_metrics.items():
        if target_metric is not None and target_metric != condition:
            if condition == "No Condition":
                ## try and extract the following mae, mse, cosine_sim, mmd
                loss_metrics = {}
                for condi in condition_metrics.keys():
                    if condi in [
                        "mae",
                        "mse",
                        "cosine_sim",
                        "mmd",
                        "cls_loss",
                        "moment_loss",
                    ]:
                        loss_metrics[condi] = condition_metrics[condi]
                if len(loss_metrics) > 0:
                    console.print("\n[bold]Table: Loss[/bold]")
                    table = Table(box=box.SQUARE, border_style="blue")
                    keys = list(loss_metrics.keys())
                    for i in range(0, len(keys), max_cols_per_row):
                        subtable_keys = keys[i : i + max_cols_per_row]

                        # Add columns
                        for k in subtable_keys:
                            wrapped_header = Text(k, style="bold")
                            wrapped_header.overflow = "fold"
                            table.add_column(
                                wrapped_header, justify="center", width=max_width
                            )

                        # Add data row
                        values = [loss_metrics[k] for k in subtable_keys]
                        formatted_values = [
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                            for v in values
                        ]
                        table.add_row(*formatted_values)

                        # Print the subtable
                        console.print(table)

                        # Reset the table for the next set of columns
                        table = Table(box=box.SQUARE, border_style="blue")
            continue
        console.print(f"\n[bold]Table: {condition}[/bold]")

        table = Table(box=box.SQUARE, border_style="blue")

        keys = list(condition_metrics.keys())
        for i in range(0, len(keys), max_cols_per_row):
            subtable_keys = keys[i : i + max_cols_per_row]

            # Add columns
            for k in subtable_keys:
                wrapped_header = Text(k, style="bold")
                wrapped_header.overflow = "fold"
                table.add_column(wrapped_header, justify="center", width=max_width)

            # Add data row
            values = [condition_metrics[k] for k in subtable_keys]
            formatted_values = [
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in values
            ]
            table.add_row(*formatted_values)

            # Print the subtable
            console.print(table)

            # Reset the table for the next set of columns
            table = Table(box=box.SQUARE, border_style="blue")

    # Check if 'loss' exists and print it last
    if "loss" in metrics:
        console.print("\n[bold]Loss:[/bold]", metrics["loss"])


def print_all_metrics_tables(
    metrics,
    max_cols_per_row=10,
    max_width=20,
    console=None,
    target_metric=None,
    generic_table: bool = False,
) -> None:
    if console is None:
        console = Console()
    console.print("[bold]Metrics Dashboard[/bold]")
    print_metrics_tables(metrics, max_cols_per_row, max_width, console, target_metric)


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
