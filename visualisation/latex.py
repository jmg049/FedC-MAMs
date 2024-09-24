# visualisation/latex.py

from argparse import ArgumentParser
import subprocess
import matplotlib.pyplot as plt
from io import BytesIO


def display_latex(latex, fname="equation.png"):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    plt.figure(figsize=(6, 2))  # Adjusted size for better visibility
    # Wrap the LaTeX string in $...$ since it's a math expression
    plt.text(0.5, 0.5, f"${latex}$", fontsize=20, ha="center", va="center")
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    buf.seek(0)
    # Save to file
    with open(fname, "wb") as f:
        f.write(buf.getvalue())

    return buf.getvalue()


if __name__ == "__main__":
    args = ArgumentParser(description="Convert LaTeX string to image.")
    args.add_argument("latex", type=str, help="LaTeX string to convert to image")
    args.add_argument("--output", type=str, default=None, help="Output file (optional)")

    parsed_args = args.parse_args()

    image = display_latex(parsed_args.latex, fname=parsed_args.output or "equation.png")
    if not parsed_args.output:
        # Display the image in the terminal using kitty's icat, then remove it
        cmd = "kitten icat equation.png && rm equation.png"
        subprocess.run(cmd, shell=True)
