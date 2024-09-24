import os
from typing import Any
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box


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
