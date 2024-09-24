from argparse import ArgumentParser
import numpy as np
import yaml
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openTSNE import TSNE
import uuid
import os

OKABE_ITO = [
    "#F0E442",
    "#56B4E9",
    "#000000",
    "#E69F00",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


@dataclass
class TSNEConfig:
    _config_path: str = "config/tsne_config.yaml"
    title_font_size: int = 26
    axis_label_font_size: int = 16
    tick_label_font_size: int = 14
    legend_font_size: int = 18
    marker_size: int = 16
    marker_styles: list = field(
        default_factory=lambda: ["o", "s"]
    )  # Changed from ["o", "x"]
    dpi: int = 300
    frame_on: bool = True
    grid: bool = False
    grid_color: str = "black"
    grid_alpha: float = 0.5
    grid_style: str = "--"
    perplexity: float = 30.0
    learning_rate: float | str = "auto"
    n_iter: int = 1000
    n_iter_without_progress: int = 300
    use_tex: bool = True
    label_mapping: dict = field(
        default_factory=lambda: {0: "Negative", 1: "Neutral", 2: "Positive"}
    )

    @staticmethod
    def load(yaml_path: str) -> "TSNEConfig":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        if config is None:
            print(f"Empty config file {yaml_path}. Using default")
            return TSNEConfig()
        return TSNEConfig(**config)

    def __post_init__(self):
        if self.use_tex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    "font.size": self.axis_label_font_size,
                    "axes.labelsize": self.axis_label_font_size,
                    "xtick.labelsize": self.tick_label_font_size,
                    "ytick.labelsize": self.tick_label_font_size,
                    "legend.fontsize": self.legend_font_size,
                    "figure.titlesize": self.title_font_size,
                    "text.latex.preamble": r"\usepackage{amsmath}",
                }
            )
        else:
            plt.rcParams.update(
                {
                    "text.usetex": False,
                    "font.size": self.axis_label_font_size,
                    "axes.labelsize": self.axis_label_font_size,
                    "xtick.labelsize": self.tick_label_font_size,
                    "ytick.labelsize": self.tick_label_font_size,
                    "legend.fontsize": self.legend_font_size,
                    "figure.titlesize": self.title_font_size,
                }
            )

        # Save the config to a yaml file with a UUID
        config_dir = os.path.dirname(self._config_path)
        config_name = os.path.splitext(os.path.basename(self._config_path))[0]
        new_config_path = os.path.join(config_dir, f"{config_name}_{uuid.uuid4()}.yaml")

        with open(new_config_path, "w") as f:
            yaml.dump(self.__dict__, f)
        print(f"Saved config to {new_config_path}")


def make_plot(
    rec_embeddings, gt_embeddings, title, config, rec_labels=None, gt_labels=None
):
    all_embeddings = np.vstack([rec_embeddings, gt_embeddings])

    tsne = TSNE(
        perplexity=config.perplexity,
        learning_rate=config.learning_rate,
        n_iter=config.n_iter,
        verbose=True,
        metric="cosine",
        initialization="pca",
    )
    tsne_embeddings = tsne.fit(all_embeddings)

    tsne_df = pd.DataFrame(tsne_embeddings, columns=["x", "y"])
    n_rec = rec_embeddings.shape[0]
    tsne_df["type"] = ["Reconstructed"] * n_rec + [
        "Ground Truth"
    ] * gt_embeddings.shape[0]

    if gt_labels is not None:
        # Use ground truth labels for both reconstructed and ground truth points
        all_labels = np.concatenate([gt_labels, gt_labels])
        tsne_df["label"] = [config.label_mapping[label] for label in all_labels]

    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=(16, 13))  # Slightly reduced height
    sns.set_style("white")
    sns.set_context("paper")

    scatter = sns.scatterplot(
        data=tsne_df,
        x="x",
        y="y",
        hue="label",
        style="type",
        palette=OKABE_ITO[: len(config.label_mapping)],
        markers=config.marker_styles,
        s=config.marker_size,
        ax=ax,
    )

    ax.set_title(title, fontsize=config.title_font_size, fontweight="bold")

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove the default legend
    ax.get_legend().remove()

    # Create custom legends
    handles, labels = scatter.get_legend_handles_labels()

    # Sentiment legend
    sentiment_handles = [
        h for h, l in zip(handles, labels) if l in config.label_mapping.values()
    ]
    sentiment_labels = [l for l in labels if l in config.label_mapping.values()]

    # Type legend
    type_handles = [
        h for h, l in zip(handles, labels) if l in ["Reconstructed", "Ground Truth"]
    ]
    type_labels = [l for l in labels if l in ["Reconstructed", "Ground Truth"]]

    # Create a new axes for the legends at the bottom, closer to the main plot
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.03], frameon=False)
    legend_ax.axis("off")

    # Add legends to the new axes
    sentiment_legend = legend_ax.legend(
        sentiment_handles,
        sentiment_labels,
        title="Labels",
        title_fontsize=config.legend_font_size,
        loc="center",
        bbox_to_anchor=(0, 0.55),
        ncol=len(sentiment_labels),
        borderaxespad=0,
        fontsize=int(config.legend_font_size * 0.8),
    )

    type_legend = legend_ax.legend(
        type_handles,
        type_labels,
        title="Type",
        title_fontsize=config.legend_font_size,
        loc="center",
        bbox_to_anchor=(0.95, 0.55),
        ncol=len(type_labels),
        borderaxespad=0,
        fontsize=int(config.legend_font_size * 0.8),
    )

    legend_ax.add_artist(sentiment_legend)
    legend_ax.add_artist(type_legend)

    # Adjust layout
    fig.subplots_adjust(
        top=0.95,
        left=-0.066,
        right=1.0,
        bottom=0.0,
    )  # Reduced bottom margin to bring legend closer to plot

    return fig


if __name__ == "__main__":
    parser = ArgumentParser("Visualise the embeddings using t-SNE")
    parser.add_argument(
        "--rec_embeddings", help="Path to the reconstructed embeddings file"
    )
    parser.add_argument(
        "--gt_embeddings", help="Path to the ground truth embeddings file"
    )
    parser.add_argument(
        "--rec_labels",
        help="Path to the reconstructed labels file",
        default=None,
    )
    parser.add_argument(
        "--gt_labels",
        help="Path to the ground truth labels file",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Path to the output file, if not provided, the plot is displayed. If provided the necessary directories are created.",
        default=None,
    )
    parser.add_argument(
        "--format",
        help="Format of the output file",
        default=".pdf",
        type=str,
        choices=[".pdf", ".png", ".svg"],
    )
    parser.add_argument("--title", help="Title of the plot", default="t-SNE Plot")
    parser.add_argument(
        "--tsne_config_path",
        help="Path to the t-SNE config file (optional)(font sizes etc.)",
    )
    args = parser.parse_args()

    config = (
        TSNEConfig.load(args.tsne_config_path)
        if args.tsne_config_path
        else TSNEConfig()
    )

    print(config)

    rec_embeddings = np.load(args.rec_embeddings)
    gt_embeddings = np.load(args.gt_embeddings)
    rec_labels = np.load(args.rec_labels)
    gt_labels = np.load(args.gt_labels)

    print(f"Reconstructed embeddings: {rec_embeddings.shape}")
    print(f"Ground truth embeddings: {gt_embeddings.shape}")
    print(f"Reconstructed labels: {rec_labels.shape}")
    print(f"Ground truth labels: {gt_labels.shape}")

    # Visualise the embeddings using t-SNE
    # Save the plot to the output directory
    # Use the title from the command line arguments
    # Use the font sizes from the config file
    # Use the marker size from the config file

    plot = make_plot(
        rec_embeddings,
        gt_embeddings,
        title=args.title,
        rec_labels=rec_labels,
        gt_labels=gt_labels,
        config=config,
    )

    os.makedirs(args.output, exist_ok=True)

    if args.output:
        plt.savefig(
            os.path.join(args.output, args.title + args.format),
            dpi=config.dpi,
            bbox_inches="tight",
        )
        print(f"Plot saved to {args.output}")
    else:
        plt.show()
