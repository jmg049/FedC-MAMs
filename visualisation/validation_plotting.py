import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import numpy as np
import argparse
from typing import Optional

# Okabe-Ito color palette
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def load_metric_data(root_dir: str) -> pd.DataFrame:
    """
    Load metric data from JSON files in multiple subdirectories and calculate the average.

    Args:
    root_dir (str): Path to the main directory containing numbered subdirectories with JSON files.

    Returns:
    pd.DataFrame: DataFrame containing the averaged metric data across all runs.
    """
    all_run_data = []
    root_path = Path(root_dir)

    for run_dir in sorted(root_path.glob("*")):
        if run_dir.is_dir() and run_dir.name.isdigit():
            run_data = []
            for json_file in sorted(run_dir.glob("*.json"), key=lambda x: int(x.stem)):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                metrics["epoch"] = int(json_file.stem) + 1  # Offset epoch by 1
                run_data.append(metrics)
            all_run_data.append(pd.DataFrame(run_data))

    # Concatenate all runs and calculate the average
    if all_run_data:
        df_all = pd.concat(all_run_data, ignore_index=True)
        df_avg = df_all.groupby("epoch").mean().reset_index()
        return df_avg
    else:
        raise ValueError("No data found in the specified directory structure.")


def plot_metrics(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    fig_size: tuple = (10, 6),
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 10,
):
    """
    Plot metrics for a two-column ML conference paper with improved accessibility and layout.

    Args:
    df (pd.DataFrame): DataFrame containing the metrics data.
    save_path (Optional[str]): Path to save the plot. If None, the plot is displayed.
    fig_size (tuple): Figure size in inches.
    title_fontsize (int): Font size for the title.
    label_fontsize (int): Font size for axis labels.
    tick_fontsize (int): Font size for tick labels.
    legend_fontsize (int): Font size for legend.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Group metrics by their base name (F1, Precision, Recall, etc.)
    metric_groups = {}
    for col in df.columns:
        if col != "epoch":
            base_name = col.split("_")[0]

            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append(col)

    # Plot each group of metrics
    for base_metric, metrics in metric_groups.items():
        fig, ax = plt.subplots(figsize=fig_size, dpi=300)

        for i, metric in enumerate(metrics):
            color = OKABE_ITO[i % len(OKABE_ITO)]
            sns.lineplot(
                data=df, x="epoch", y=metric, label=metric, color=color, linewidth=1.5
            )

        plt.title(f"{base_metric} Metrics", fontsize=title_fontsize, fontweight="bold")
        plt.xlabel("Epoch", fontsize=label_fontsize)
        plt.ylabel("Value", fontsize=label_fontsize)

        # Set x-axis ticks and limits
        max_epoch = df["epoch"].max()
        x_ticks = list(range(0, max_epoch + 2, 2))
        plt.xticks(x_ticks, fontsize=tick_fontsize)
        plt.xlim(0, max_epoch + 1)

        # Set y-axis ticks between 0.0 and 1.0 if appropriate
        if df[metrics].min().min() >= 0 and df[metrics].max().max() <= 1:
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize=tick_fontsize)
        else:
            plt.yticks(fontsize=tick_fontsize)

        # Improve grid visibility
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add markers to lines for better distinction
        for line in ax.lines:
            line.set_marker("o")
            line.set_markersize(3)

        # Move legend below the plot
        plt.legend(
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            borderaxespad=0.0,
            ncol=min(4, len(metrics)),
            frameon=True,
        )

        # Adjust layout to prevent cutoff and accommodate legend
        plt.tight_layout()

        if save_path:
            plt.savefig(
                f"{save_path}_{base_metric}.pdf", format="pdf", bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from JSON files.")
    parser.add_argument(
        "--root_dir", type=str, help="Path to the directory containing JSON files."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the plot. If not provided, the plot is displayed.",
    )

    args = parser.parse_args()

    df = load_metric_data(args.root_dir)
    plot_metrics(df, save_path=args.save_path)
