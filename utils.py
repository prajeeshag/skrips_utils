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

app = typer.Typer()


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
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    plt.colorbar(cs)

    plt.savefig(filepath)


@app.command("plot")
def plot_bathymetry(
    ncfile: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of NetCdf Bathymetry file",
    ),
    data: Path = typer.Option(
        None,
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="Path of the `data` namelist file of MITgcm",
    ),
    out_file: Path = typer.Option(
        "bathymetry.png",
        writable=True,
        file_okay=True,
        dir_okay=False,
        help="Path of the output figure",
    ),
):
    """
    Plot the bathymetry of MITgcm:

        1. From the informations given in `data` namelist of MITgcm.
        2. From Netcdf bathymetry file
    """
    if not any([ncfile, data]) or all([ncfile, data]):
        raise typer.BadParameter("Provide either ncfile or data!")

    if ncfile:
        ds = xr.open_dataset(ncfile)
        z = ds["z"]
        lat = z["lat"]
        lon = z["lon"]
    elif data:
        z, lat, lon = _get_bathy_info_from_data(data)

    plot_bathy(lat, lon, z, filepath=out_file)


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
    xgorigin = nml["parm04"]["xgorigin"]
    ygorigin = nml["parm04"]["ygorigin"]
    delx = nml["parm04"]["delx"]
    dely = nml["parm04"]["dely"]
    nx = len(delx)
    ny = len(dely)

    lon = np.zeros(nx)
    lat = np.zeros(ny)

    lon[0] = xgorigin + delx[0] * 0.5
    lat[0] = ygorigin + dely[0] * 0.5

    for i in range(nx - 1):
        lon[i + 1] = lon[i] + (delx[i] + delx[i + 1]) * 0.5

    for i in range(ny - 1):
        lat[i + 1] = lat[i] + (dely[i] + dely[i + 1]) * 0.5

    z = np.fromfile(f"{idir}/{bathyfile}", ">f4").reshape(ny, nx)
    return z, lat, lon


@app.command(name="create")
def make_bathy(
    input_bathy: str = typer.Argument(..., help="Path of input bathymetry netcdf file"),
    wrf_geo: str = typer.Argument(..., help="Path of WRF `geo_em` file"),
    out_file: str = typer.Option("Bathymetry", help="Path of output bathymetry file"),
):
    """
    Create bathymetry file for MITgcm

    The coordinate information required for creating bathymetry can be \
        given in following ways:
        1) Coordinates taken from the WRF geo_em file.
           This will also create the relevant MITgcm namelist fields.
        2) Coordinates taken from MITgcm namelist. (Not Implemented)
    """

    ds_geo = xr.open_dataset(wrf_geo)
    ds_input_bathy = xr.open_dataset(input_bathy)
    input_bathy = ds_input_bathy["z"]

    XLAT_M = ds_geo["XLAT_M"][0, :, 0]
    XLONG_M = ds_geo["XLONG_M"][0, 0, :]

    print(XLAT_M.values)
    print(XLONG_M.values)

    grid_out = xr.Dataset(
        {
            "lat": (["lat"], XLAT_M.values, {"units": "degrees_north"}),
            "lon": (["lon"], XLONG_M.values, {"units": "degrees_east"}),
        }
    )
    regridder = xe.Regridder(ds_input_bathy, grid_out, "conservative")

    dr_out = regridder(input_bathy, keep_attrs=True)

    dr_out.to_netcdf(f"{out_file}.nc")

    plot_bathy(XLAT_M, XLONG_M, dr_out, filepath=f"{out_file}.png")


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
    bin_bathy: Path = typer.Option(
        None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path of output binary file",
    ),
):
    """
    Convert NetCdf bathymetry file to mitgcm compatible binary file
    """
    if bin_bathy is None:
        bin_bathy = Path(f"{nc_bathy.stem}.bin")

    z = xr.open_dataset(nc_bathy)["z"]
    _da2bin(z, bin_bathy)


def compare_lnd_ocn_mask(wrf_geo: Path = typer.Option(None)):
    pass


if __name__ == "__main__":
    ds = xr.open_dataset("~/S2S/Domain/geo_em.d01.nc")
    print(ds)
