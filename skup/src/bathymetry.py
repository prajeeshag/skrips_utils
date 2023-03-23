import math
from pathlib import Path

import cartopy.crs as ccrs
import f90nml
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
import numpy as np
import typer
import xarray as xr
import xesmf as xe
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import BoundaryNorm
from mpl_interactions import panhandler, zoom_factory
from rich import print
from scipy.ndimage import label
import matplotlib.patches as patches

import logging

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

    z, lat, lon = _get_bathy_info_from_data(data, bathy_file)

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
        10.0,
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


def _get_bathy_info_from_data(data, bathy_file=None):
    """
    Get z, lon, lat from the information provided by `data` namelist
    """
    nml = f90nml.read(data)
    idir = data.parents[0]
    usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
    if not usingsphericalpolargrid:
        raise NotImplementedError(
            "The bathymetry plotting is only implemented for spherical-polar grid"
        )
    bathyfile = bathy_file
    if not bathyfile:
        bathyfile = nml["parm05"]["bathyfile"]

    nx, ny, lon, lat = grid_from_parm04(nml["parm04"])

    z = np.fromfile(f"{idir}/{bathyfile}", ">f4").reshape(ny, nx)
    return z, lat, lon


def grid_from_parm04(nml):
    xgorigin = nml["xgorigin"]
    ygorigin = nml["ygorigin"]
    delx = nml["delx"]
    dely = nml["dely"]
    nx = len(delx)
    ny = len(dely)

    lon_bnd = np.zeros(nx + 1)
    lat_bnd = np.zeros(ny + 1)

    lon_bnd[0] = xgorigin
    lat_bnd[0] = ygorigin

    for i, dx in enumerate(delx):
        lon_bnd[i + 1] = lon_bnd[i] + dx

    for i, dy in enumerate(dely):
        lat_bnd[i + 1] = lat_bnd[i] + dy

    lon = (lon_bnd[1:] + lon_bnd[0:-1]) * 0.5
    lat = (lat_bnd[1:] + lat_bnd[0:-1]) * 0.5

    # lon, lat = np.meshgrid(lon, lat)
    # lon_bnd, lat_bnd = np.meshgrid(lon_bnd, lat_bnd)

    # grid_out = xr.Dataset(
    #     {
    #         "lat": (["nlon", "nlat"], lat, {"units": "degrees_north"}),
    #         "lon": (["nlon", "nlat"], lon, {"units": "degrees_east"}),
    #         "lat_b": (["nlonb", "nlatb"], lat_bnd, {"units": "degrees_north"}),
    #         "lon_b": (["nlonb", "nlatb"], lon_bnd, {"units": "degrees_east"}),
    #     }
    # )

    return nx, ny, lon, lat


@app.command(name="create")
def make_bathy(
    input_bathy: str = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of input bathymetry netcdf file",
    ),
    data: str = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of MITgcm `data` namelist",
    ),
    out_file: str = typer.Option(
        "bathymetry.bin",
        help="Path of output bathymetry file",
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
):
    """
    Create bathymetry file for MITgcm from the grid
    information taken from MITgcm `data` namelist
    """
    ds_input_bathy = xr.open_dataset(input_bathy)
    input_bathy = ds_input_bathy["z"]

    nml = f90nml.read(data)
    usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
    if not usingsphericalpolargrid:
        raise NotImplementedError(
            "bathymetry create is only implemented for spherical-polar grid"
        )

    nx, ny, lon, lat = grid_from_parm04(nml["parm04"])

    grid_out = xr.Dataset(
        {
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )
    regridder = xe.Regridder(ds_input_bathy, grid_out, "conservative")

    dr_out = regridder(input_bathy, keep_attrs=True)

    _da2bin(dr_out, out_file)


class Mask(np.ndarray):
    """Mask class"""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array: np.ndarray):
        pass

    def get_features(self):
        features, num_features = label(self)
        return features, list(range(1, num_features + 1))

    def get_pools(self):
        labeled_array, labels = self.get_features()
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


class Features:
    def __init__(self) -> None:
        self.shown = False
        self.boxes = []


class EditBathy:
    def __init__(self, z) -> None:
        self.z = z
        with plt.ioff():
            self.fig, self.ax = plt.subplots()

        self.omask = Mask(np.where(self.z >= 0, 0, 1))
        self.cmap = plt.colormaps["tab20"]
        self.cmap.set_under("white")
        pools, pool_labels = self.omask.get_pools()
        self.levels = pool_labels
        self.norm = BoundaryNorm(self.levels, ncolors=self.cmap.N)
        # self.mesh = self.ax.pcolormesh(
        #     self.z, cmap=self.cmap, norm=self.norm, picker=True
        # )
        self.mesh = self.ax.pcolormesh(pools, cmap=self.cmap, norm=self.norm)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.ax.set_aspect("equal")

        self.fig.subplots_adjust(left=0.2)
        # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        rax = self.fig.add_axes([0.05, 0.4, 0.1, 0.15])
        check = CheckButtons(ax=rax, labels=["Pools"])

        self.pools = Features()
        self.islands = Features()
        check.on_clicked(self._toggle_pools)
        plt.colorbar(self.mesh)
        # _ = zoom_factory(self.ax)
        _ = panhandler(self.fig, button=2)
        plt.show()

    def _check_box_callback(self, label):
        if label == "Pools":
            self._toggle_feature(self.pools)
        elif label == "Islands":
            self._toggle_feature(self.islands)

    def _toggle_feature(self, feature):
        if feature.shown:
            for rect in feature.boxes:
                rect.remove()
            feature.shown = False
        else:
            self.pool.boxes = self._get_boxes(*self.omask.get_pools())
            for rect in feature.boxes:
                self.ax.add_patch(rect)
            feature.shown = True
        self.fig.canvas.draw()

    def _get_boxes(self, features, feature_label):
        boxes = []
        for i in feature_label:
            points = np.where(features == i)
            ny, nx = features.shape
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

    def _on_pick(self, event):
        logger.debug("_on_pick")
        mouseevent = event.mouseevent
        if mouseevent.button is not MouseButton.LEFT:
            return
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
        self.z[_gy, _gx] = 25
        self.mesh.set_array(self.z)
        self.fig.canvas.draw()


@app.command("edit")
def edit_bathy(
    mitgcm_grid_nml: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="MITgcm namelist containing grid information",
    ),
    bathy_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help=(
            "MITgcm bathymetry file; \n"
            + "(if not given program will try to read it from namelist"
            + " parameter `&parm05:bathyfile`)"
        ),
    ),
):
    """Opens up a GUI to click and edit Bathymetry"""

    Z, _, _ = _get_bathy_info_from_data(mitgcm_grid_nml, bathy_file=bathy_file)
    Z[Z >= 0] = 100.0
    EditBathy(Z)


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


def compare_lnd_ocn_mask(wrf_geo: Path = typer.Option(None)):
    pass


if __name__ == "__main__":
    # grid = xe.util.grid_global(5, 4)
    # print(grid["lon"][:, 0])
    # print(grid["lon_b"][:, 0])
    ds = xr.open_dataset("~/S2S/")
