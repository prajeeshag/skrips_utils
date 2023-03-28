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
from rich import print
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


def _da2bin(da: xr.DataArray, binfile: Path):
    """
    write xarray data array to with big-endian byte ordering
    as single-precision real numbers (which is NumPy float32 or
    equivalently, Fortran real*4 format)
    """
    da.values.astype(">f4").tofile(binfile)


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


@app.command("clip")
def clip_bathymetry(
    bathy_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        writable=True,
        dir_okay=False,
        file_okay=True,
        help="MITgcm bathymetry file",
    ),
    output: Path = typer.Option(
        None,
        exists=False,
        writable=True,
        dir_okay=False,
        file_okay=True,
        help="Output file",
    ),
    overwrite: bool = typer.Option(False, help="Overwrite the input file!!"),
    mindepth: float = typer.Option(
        None,
        help="Minimum depth (-ve downwards), z[z > mindepth]=landvalue",
    ),
    maxdepth: float = typer.Option(
        None,
        help="Maximum depth (-ve downwards), z[z < maxdepth]= maxdepth",
    ),
    landvalue: float = typer.Option(
        100.0,
        help="Land value",
    ),
):
    """
    Clip the bathymetry of MITgcm between min and max values.
    """
    if not any([mindepth, maxdepth]):
        raise typer.BadParameter("Neither mindepth or maxdepth is given.")

    if not output:
        if overwrite:
            output = bathy_file
        else:
            raise typer.BadParameter(
                "Provide a output file name or use option --overwrite"
            )

    z = np.fromfile(bathy_file, ">f4")

    clip_bathy(z, mindepth, maxdepth, landvalue)

    z.astype(">f4").tofile(output)


def clip_bathy(z, mindepth=None, maxdepth=None, landvalue=10.0):
    if mindepth:
        z[z > mindepth] = landvalue
    if maxdepth:
        z[z < maxdepth] = maxdepth




class Features:
    def __init__(self, mask) -> None:
        self.shown = False
        self.boxes = []
        self.mask = mask
        self._get_features()

    def _get_features(self):
        self.array, num_labels = ndimage.label(self.mask)
        self.labels = list(range(1, num_labels + 1))

    def max_points(self, n, array=None):
        labels = []
        if array is None:
            array = self.array.copy()
        for i in self.labels:
            npoints = np.count_nonzero(array == i)
            if npoints > 0 and npoints <= n:
                labels.append(i)
            else:
                array[array == i] = 0
        return array, labels

    def no_edge(self, array=None):
        if array is None:
            labeled_array = self.array
        labels = self.labels
        edge_pixels = np.concatenate(
            (
                labeled_array[0, :],
                labeled_array[-1, :],
                labeled_array[:, 0],
                labeled_array[:, -1],
            )
        )
        # get the labels that touch the edges
        edge_labels = np.intersect1d(labels, edge_pixels)
        # get the labels that do not touch the edges
        pool_labels = np.setdiff1d(labels, edge_labels)

        pools = labeled_array
        for i in edge_labels:
            pools[pools == i] = 0
        return pools, pool_labels

    def get_boxes(self, array, labels):
        boxes = []
        for i in labels:
            points = np.where(array == i)
            ny, nx = array.shape
            x = np.max([np.min(points[1]) - 1, 0])
            x2 = np.min([np.max(points[1]) + 1, nx])

            y = np.max([np.min(points[0]) - 1, 0])
            y2 = np.min([np.max(points[0]) + 1, ny])

            width = x2 - x + 1
            height = y2 - y + 1

            # recalibrate rectangle if it is too small
            rwidth = np.max([width, nx // 100])
            rheight = np.max([height, ny // 100])
            x = x - (rwidth - width) // 2
            y = y - (rheight - height) // 2
            width = rwidth
            height = rheight

            boxes.append(
                patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor="k", facecolor="none"
                )
            )
        return boxes


@app.command()
def del_islands(
    grid_nml: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    in_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "input bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
    out_file: Path = typer.Option(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="ouput bathymetry file",
    ),
    n_points: int = typer.Option(
        1,
        help="Islands having grid points <= `n_points` will be deleted",
    ),
    min_depth: int = typer.Option(
        None,
        help="depth value (if not given will try to find out from the input data)",
    ),
    show: bool = typer.Option(
        False,
        help="Show the islands which is going to be deleted",
    ),
):
    """Convert small land islands to ocean points"""

    def delete_islands(event):
        logger.info("Deleting islands")
        for i in labels:
            z[array == i] = min_depth
        logger.info(f"Saving bathymetry to file {out_file}")
        z.astype(">f4").tofile(out_file)
        plt.close()

    z, _, _ = _get_bathy_from_nml(grid_nml, bathy_file=in_file)
    if not min_depth:
        min_depth = np.amax(z[z < 0])
    mask = np.where(z >= 0, 1, 0)
    features = Features(mask)
    levels = list(np.linspace(-3000, -200, 10))[:-1] + list(np.linspace(-200, 0, 21))
    levels = [-0.0000001 if item == 0.0 else item for item in levels]
    cmap = plt.cm.jet
    cmap.set_over("white")
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    with plt.ioff():
        fig, ax = plt.subplots()
    mesh = ax.pcolormesh(z, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    array, labels = features.max_points(n_points)
    boxes = features.get_boxes(array, labels)
    for rect in boxes:
        ax.add_patch(rect)
    fig.canvas.draw()

    fig.subplots_adjust(bottom=0.2)
    # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axdel = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bndel = Button(axdel, "Delete")
    bndel.on_clicked(delete_islands)
    plt.colorbar(mesh)
    plt.show()


@app.command()
def del_ponds(
    grid_nml: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    in_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "input bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
    out_file: Path = typer.Option(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="ouput bathymetry file",
    ),
    n_points: int = typer.Option(
        1,
        help="Ponds having grid points <= `n_points` will be deleted",
    ),
):
    """Convert small ponds to land points"""

    def delete_islands(event):
        logger.info("Deleting ponds")
        for i in labels:
            z[array == i] = min_depth
        logger.info(f"Saving bathymetry to file {out_file}")
        z.astype(">f4").tofile(out_file)
        plt.close()

    z, _, _ = _get_bathy_from_nml(grid_nml, bathy_file=in_file)
    min_depth = 100.0
    mask = np.where(z < 0, 1, 0)
    features = Features(mask)
    levels = list(np.linspace(-3000, -200, 10))[:-1] + list(np.linspace(-200, 0, 21))
    levels = [-0.0000001 if item == 0.0 else item for item in levels]
    cmap = plt.cm.jet
    cmap.set_over("white")
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    with plt.ioff():
        fig, ax = plt.subplots()
    mesh = ax.pcolormesh(z, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    array, labels = features.max_points(n_points)
    boxes = features.get_boxes(array, labels)
    for rect in boxes:
        ax.add_patch(rect)
    fig.canvas.draw()

    fig.subplots_adjust(bottom=0.2)
    # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axdel = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bndel = Button(axdel, "Delete")
    bndel.on_clicked(delete_islands)
    plt.colorbar(mesh)
    plt.show()


@app.command()
def del_creeks(
    grid_nml: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    in_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "input bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
    out_file: Path = typer.Option(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="ouput bathymetry file",
    ),
    n_points: int = typer.Option(
        1,
        help="Ponds having grid points <= `n_points` will be deleted",
    ),
):
    """Convert small ponds to land points"""

    def delete_islands(event):
        logger.info("Deleting ponds")
        for i in labels:
            z[array == i] = min_depth
        logger.info(f"Saving bathymetry to file {out_file}")
        z.astype(">f4").tofile(out_file)
        plt.close()

    z, _, _ = _get_bathy_from_nml(grid_nml, bathy_file=in_file)
    min_depth = 100.0
    mask = np.where(z < 0, 1, 0)
    creeks = _creek_mask(mask)

    features = Features(creeks)
    levels = list(np.linspace(-3000, -200, 10))[:-1] + list(np.linspace(-200, 0, 21))
    levels = [-0.0000001 if item == 0.0 else item for item in levels]
    cmap = plt.cm.jet
    cmap.set_over("white")
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    with plt.ioff():
        fig, ax = plt.subplots()
    mesh = ax.pcolormesh(z, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    array, labels = features.max_points(n_points)
    boxes = features.get_boxes(array, labels)
    for rect in boxes:
        ax.add_patch(rect)
    fig.canvas.draw()
    fig.subplots_adjust(bottom=0.2)
    # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axdel = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bndel = Button(axdel, "Delete")
    bndel.on_clicked(delete_islands)
    plt.colorbar(mesh)

    plt.show()


def _creek_mask(image, n_neibhours=3):
    cimage = np.zeros(image.shape, dtype=int)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the pixel is black (i.e., part of a feature)
            if image[y, x] == 1:
                y1 = max(y - 1, 0)
                y2 = min(y + 2, image.shape[0] + 1)
                x1 = max(x - 1, 0)
                x2 = min(x + 2, image.shape[1] + 1)
                # Count the number of neighbors of the pixel
                # num_neighbors = np.sum(image[y1:y2, x1:x2]) - 1
                nyn = min(np.sum(image[y1:y2, x]) - 1, 1)
                nxn = min(np.sum(image[y, x1:x2]) - 1, 1)

                # If the pixel has one or two neighbors, label it
                if nxn + nyn <= 1:
                    cimage[y, x] = 1
    return cimage


class EditBathy:
    def __init__(self, z, out_file) -> None:
        self.z = z
        self.out_file = out_file
        self.min_depth = np.amax(z[z < 0])
        self.lnd_val = 100.0
        levels = list(np.linspace(-3000, -200, 10))[:-1] + list(
            np.linspace(-200, 0, 21)
        )
        levels = [-0.0000001 if item == 0.0 else item for item in levels]
        cmap = plt.cm.jet
        cmap.set_over("white")
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        with plt.ioff():
            self.fig, self.ax = plt.subplots()
        self.mesh = self.ax.pcolormesh(z, cmap=cmap, norm=norm, picker=True)
        self.ax.set_aspect("equal")
        plt.colorbar(self.mesh)

        # self.mesh = self.ax.pcolormesh(pools, cmap=self.cmap, norm=self.norm)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.subplots_adjust(bottom=0.2)
        self.axsave = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.bndel = Button(self.axsave, "Save")
        self.bndel.on_clicked(self.save_z)

        _ = zoom_factory(self.ax)
        _ = panhandler(self.fig, button=2)
        plt.show()

    def save_z(self, event):
        logger.info(f"Saving to file {self.out_file}")
        self.z.astype(">f4").tofile(self.out_file)

    def _on_pick(self, event):
        logger.debug("_on_pick")
        mouseevent = event.mouseevent
        if mouseevent.xdata is None or mouseevent.ydata is None:
            return

        logger.debug(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (
                "double" if mouseevent.dblclick else "single",
                mouseevent.button,
                mouseevent.x,
                mouseevent.y,
                mouseevent.xdata,
                mouseevent.ydata,
            )
        )
        _gx = int(math.floor(mouseevent.xdata))
        _gy = int(math.floor(mouseevent.ydata))

        if mouseevent.button is MouseButton.LEFT:
            # click left mouse button to make a land point
            self.z[_gy, _gx] = self.lnd_val
        elif mouseevent.button is MouseButton.RIGHT:
            # click right mouse button to make a ocean point
            self.z[_gy, _gx] = self.min_depth

        self.mesh.set_array(self.z)
        self.fig.canvas.draw()


@app.command("edit")
def edit_bathy(
    grid_nml: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    in_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "MITgcm bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
    out_file: Path = typer.Option(
        None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="ouput bathymetry file",
    ),
):
    """Opens up a GUI to click and edit Bathymetry"""

    if out_file is None:
        out_file = in_file

    Z, _, _ = _get_bathy_from_nml(grid_nml, bathy_file=in_file)
    Z[Z >= 0] = 100.0
    EditBathy(Z, out_file=out_file)


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


@app.command()
def match_wrf_lmask(
    wrf_geo: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="wrf geo_em file",
    ),
    grid_nml: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    in_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help=(
            "Input bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
    out_file: Path = typer.Option(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help=("Output file; \n"),
    ),
):
    """Edits MITgcm bathymetry to match the WRF land mask"""

    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)

    luindex = ds["LU_INDEX"].squeeze()

    iswater = ds.attrs["ISWATER"]
    ofrac = luindex.values
    if np.any(np.isnan(ofrac)):
        msg = "LU_INDEX contains NaN values"
        logger.error(msg)
        raise ValueError(msg)

    ofrac[ofrac != iswater] = 0.0  # 17 is water body in LULC of WRF
    ofrac[ofrac == iswater] = 1.0  # 17 is water body in LULC of WRF

    logger.info("Reading bathymetry file")
    z, zlat, zlon = _get_bathy_from_nml(grid_nml, bathy_file=in_file)

    if np.any(np.isnan(z)):
        msg = "Bathymetry contains NaN values"
        logger.error(msg)
        raise ValueError(msg)

    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac

    lpOcn = np.count_nonzero(ocnfrac)
    logger.info(f"Number of ocean points in bathymetry file: {lpOcn}")
    lpOcn = np.count_nonzero(ofrac)
    logger.info(f"Number of ocean points in {wrf_geo}: {lpOcn}")

    lpOcn = mismatch.size
    logger.info(f"Total number of points: {lpOcn}")
    lpOcn = np.count_nonzero(mismatch)
    logger.info(f"Number of mismatch points: {lpOcn}")
    lpOcn = np.count_nonzero(mismatch == 1)
    logger.info(f"Number of points were WRF:land,MITgcm:ocean : {lpOcn}")
    lpOcn = np.count_nonzero(mismatch == -1)
    logger.info(f"Number of points were WRF:ocean,MITgcm:land : {lpOcn}")
    lpOcn = np.count_nonzero(mismatch == 0)
    logger.info(f"Number of Matching points : {lpOcn}")

    mindepth = np.amax(z[z < 0])

    logger.info(f"Depth value used when creating a ocean point: {mindepth}")
    # If WRF says land put that point as land in MITgcm
    logger.info("Points were WRF:land,MITgcm:Ocean: to MITgcm:land")
    z[mismatch == 1] = 100.0
    # If WRF says ocean put that point as ocean in
    # MITgcm with minimum depth value of -5.0
    logger.info("Points were WRF:ocean,MITgcm:land: to MITgcm:ocean")
    z[mismatch == -1] = mindepth

    # Compute mismatch Again
    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac
    lpOcn = np.count_nonzero(mismatch)
    logger.info(f"Number of mismatch points: {lpOcn}")
    z.astype(">f4").tofile(out_file)


if __name__ == "__main__":
    # grid = xe.util.grid_global(5, 4)
    # print(grid["lon"][:, 0])
    # print(grid["lon_b"][:, 0])
    ds = xr.open_dataset("~/S2S/")
