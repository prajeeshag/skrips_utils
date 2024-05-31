import logging
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
import typer

from .utils import vgrid_from_parm04
from enum import Enum


app = typer.Typer(add_completion=False)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


class FileType(str, Enum):
    png = "png"
    jpg = "jpg"
    eps = "eps"
    pdf = "pdf"


@app.command()
def main(
    nml: Path = typer.Option(
        default=Path("./data"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm `data` namelist file",
    ),
    saveas: FileType = typer.Option(
        default=FileType.png,
    ),
):
    """Read zi from data namelist and show MITgcm vertical grid"""
    z, zi, _ = vgrid_from_parm04(nml)

    print(zi)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot horizontal lines at each depth level
    for depth in zi:
        ax.axhline(y=depth, color="b", linestyle="--")

    # Annotate each line with the depth value
    for depth in zi:
        ax.text(0.5, depth, f"{depth} m", ha="right", va="center", color="black")

    # Setting labels and title
    ax.set_xlabel("Grid Cells")
    ax.set_ylabel("Depth Level (m)")
    ax.set_title("Vertical Grid")
    ax.set_yticks(zi)
    ax.invert_yaxis()  # Invert y-axis to have surface at the top

    plt.savefig(f"out.{saveas.value}", dpi=600)


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
