import logging
import math
from pathlib import Path

import cartopy.crs as ccrs
import f90nml
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
import xesmf as xe
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
from scipy import ndimage

from .utils import _get_bathy_from_nml, _grid_from_parm04

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False)


def plot_bathy(lat, lon, z, ax_in=None, gridlines=True, title=True):
    ax = ax_in
    if not ax:
        ax = plt.axes(projection=ccrs.PlateCarree())

    levels = list(np.linspace(-3000, -200, 10))[:-1] + list(np.linspace(-200, 0, 21))
    levels = [-0.0000001 if item == 0.0 else item for item in levels]

    cmap = plt.cm.jet
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    zcopy = z.copy()
    zcopy[zcopy >= 0] = np.nan
    cs = ax.pcolormesh(
        lon,
        lat,
        zcopy,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    if not ax_in:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        zneg = z[z < 0]
        zmin = np.min(zneg)
        zmax = np.max(zneg)
        ax.set_title(f"min={zmin:.2f}, max={zmax:.2f}", loc="right")
        plt.colorbar(cs)

    return cs


@app.command("plot")
def plot_bathymetry(
    data: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="Path of the namelist file containing the grid information of MITgcm",
    ),
    bathy_file: Path = typer.Option(
        None,
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="Path of bathymetry file",
    ),
    output: Path = typer.Option(
        "bathymetry.png",
        writable=True,
        file_okay=True,
        dir_okay=False,
        help="Path of the output figure",
    ),
):
    """
    Plot the bathymetry of MITgcm: given in `data` namelist of MITgcm.
    """

    z, lat, lon = _get_bathy_from_nml(data, bathy_file)

    plot_bathy(lat, lon, z)
    plt.savefig(output)
    print(f"Bathymetry figure is saved in file {output}")


@app.command()
def nc2bin(
    nc_bathy: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of input netcdf bathymetry file",
    ),
    bin_bathy: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path of output binary file",
    ),
):
    """
    Convert NetCdf bathymetry file to mitgcm compatible binary file
    """
    z = xr.open_dataset(nc_bathy)["z"]
    _da2bin(z, bin_bathy)


if __name__ == "__main__":
    # grid = xe.util.grid_global(5, 4)
    # print(grid["lon"][:, 0])
    # print(grid["lon_b"][:, 0])
    ds = xr.open_dataset("~/S2S/")
