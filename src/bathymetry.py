import math
from pathlib import Path

import cartopy.crs as ccrs
import f90nml
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
import xesmf as xe
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import BoundaryNorm
from mpl_interactions import panhandler, zoom_factory
from rich import print

app = typer.Typer(pretty_exceptions_show_locals=False)


def _da2bin(da: xr.DataArray, binfile: Path):
    """
    write xarray data array to with big-endian byte ordering
    as single-precision real numbers (which is NumPy float32 or
    equivalently, Fortran real*4 format)
    """
    da.values.astype(">f4").tofile(binfile)


def plot_bathy(lat, lon, z, filepath="bathymetry.png"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    levels = list(np.linspace(-3000, -200, 10))[:-1] + list(np.linspace(-200, 0, 21))
    levels = [-0.0000001 if item == 0.0 else item for item in levels]

    cmap = plt.cm.jet
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

    cs = ax.contourf(
        lon,
        lat,
        z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        extend="min",
    )

    zneg = z[z < 0]
    zmin = np.min(zneg)
    zmax = np.max(zneg)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"min={zmin:.2f}, max={zmax:.2f}", loc="right")
    plt.colorbar(cs)
    plt.savefig(filepath)


@app.command("plot")
def plot_bathymetry(
    data: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="Path of the `data` namelist file of MITgcm",
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

    z, lat, lon = _get_bathy_info_from_data(data)

    plot_bathy(lat, lon, z, filepath=output)
    print(f"Bathymetry figure is saved in file {output}")


@app.command("clip")
def clip_bathymetry(
    data: Path = typer.Argument(
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
        help="Minimum depth (-ve downwards), z[z > mindepth] = landvalue",
    ),
    maxdepth: float = typer.Option(
        None,
        help="Maximum depth (-ve downwards), z[z < maxdepth] = maxdepth",
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
            output = data
        else:
            raise typer.BadParameter(
                "Provide a output file name or use option --overwrite"
            )

    z = np.fromfile(data, ">f4")

    clip_bathy(z, mindepth, maxdepth, landvalue)

    z.astype(">f4").tofile(output)


def clip_bathy(z, mindepth=None, maxdepth=None, landvalue=10.0):
    if mindepth:
        z[z > mindepth] = landvalue
    if maxdepth:
        z[z < maxdepth] = maxdepth


def _get_bathy_info_from_data(data):
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


def _on_pick(event):
    mouseevent = event.mouseevent
    if mouseevent.button is not MouseButton.LEFT:
        return
    artist = event.artist

    print(
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
    zz = artist.get_array()
    dims = [i - 1 for i in artist.get_coordinates().shape[:2]]
    ind = _gy * dims[1] + _gx
    zz[ind] = 25
    artist.set_array(zz)
    artist.get_figure().canvas.draw()


def edit_bathy(_on_pick):
    Z = xr.open_dataset("Bathymetry.nc")["z"]
    Z = Z.values
    Z[Z >= 0] = 100
    with plt.ioff():
        fig, ax = plt.subplots()

    cmap = plt.colormaps["jet"]
    cmap.set_over("white")
    levels = np.linspace(-1000, 0, 10)
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    mesh = ax.pcolormesh(Z, cmap=cmap, norm=norm, picker=True)
    fig.canvas.mpl_connect("pick_event", _on_pick)
    # plt.title('matplotlib.pyplot.pcolormesh() function Example', fontweight="bold")

    plt.colorbar(mesh)

    _ = zoom_factory(ax)
    _ = panhandler(fig, button=2)
    plt.show()


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
