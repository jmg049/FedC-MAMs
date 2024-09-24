import os
import subprocess
import sys
from typing import Any
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box


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
    metrics, max_cols_per_row=10, max_width=20, console=None
) -> None:
    """
    Print metrics tables with bordered formatting for improved readability.

    :param metrics: Dictionary of metric names and values
    :param max_cols_per_row: Maximum number of columns per table row
    :param max_width: Maximum width for each column
    :param console: Rich Console object (if None, a new one will be created)
    """
    if console is None:
        console = Console()

    n_tables = len(metrics) // max_cols_per_row
    if len(metrics) % max_cols_per_row != 0:
        n_tables += 1

    for i in range(n_tables):
        table = Table(
            title=f"Metrics Table {i+1}",
            show_header=True,
            header_style="bold",
            box=box.SQUARE,  # Changed to SQUARE for full borders
            border_style="blue",
        )

        keys = list(metrics.keys())[i * max_cols_per_row : (i + 1) * max_cols_per_row]
        values = [metrics[k] for k in keys]

        for k in keys:
            wrapped_header = Text(k, style="bold")
            wrapped_header.overflow = "fold"
            table.add_column(wrapped_header, justify="center", width=max_width)

        # Add data row
        formatted_values = [
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in values
        ]

        table.add_row(*formatted_values)

        console.print(table)
        console.print("")  # Add a blank line between tables


def print_all_metrics_tables(
    metrics, max_cols_per_row=10, max_width=20, console=None
) -> None:
    if console is None:
        console = Console()
    console.print("[bold]Metrics Dashboard[/bold]")
    print_metrics_tables(metrics, max_cols_per_row, max_width, console)


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
