import typer
from pathlib import Path
import xarray as xr
import f90nml
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs


from .bathymetry import _get_bathy_info_from_data, plot_bathy

app = typer.Typer()


@app.command()
def create_from_wrf(
    wrf_geo: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="WRF `wrf_geo` file.",
    ),
    out_file: Path = typer.Option(
        "mitgcm_grid.nml",
        writable=True,
        help="Output grid info nml file.",
    ),
):
    """
    Creates the grid parameters for MITgcm from WRf geo_em file.
    """
    ds = xr.open_dataset(wrf_geo)

    nml = get_parm04_from_geo(ds)
    nml.write(out_file, force=True)


def get_parm04_from_geo(ds):
    """Get the MITgcm PARM04 grid parameters from WRF geo_em file xarray dataset"""
    lat_bnd = ds["XLAT_V"][0, :, 0].values
    lon_bnd = ds["XLONG_U"][0, 0, :].values
    nml = f90nml.Namelist()
    delx = lon_bnd[1:] - lon_bnd[0:-1]
    dely = lat_bnd[1:] - lat_bnd[0:-1]
    nml["parm04"] = {
        "usingsphericalpolargrid": True,
        "xgorigin": lon_bnd[0],
        "ygorigin": lat_bnd[0],
        "delx": list(delx),
        "dely": list(dely),
    }

    return nml


@app.command()
def solve_lndmask_mismatch(
    wrf_geo: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="WRF geo_em file",
    ),
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
    ds = xr.open_dataset(wrf_geo)

    luindex = ds["LU_INDEX"].squeeze()
    lon = ds["XLONG_M"].squeeze()
    lat = ds["XLAT_M"].squeeze()

    iswater = ds.attrs["ISWATER"]
    ofrac = luindex.values
    print(ofrac)
    ofrac[ofrac != iswater] = 0.0  # 17 is water body in LULC of WRF
    ofrac[ofrac == iswater] = 1.0  # 17 is water body in LULC of WRF

    z, zlat, zlon = _get_bathy_info_from_data(mitgcm_grid_nml, bathy_file=bathy_file)

    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac

    print(np.count_nonzero(ocnfrac))
    print(np.count_nonzero(ofrac))

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(10, 10),
    )

    omasks = [ofrac, ocnfrac, mismatch]

    for i, mask in enumerate(omasks):
        axs[0, i].pcolor(
            lon,
            lat,
            mask,
            vmin=-1,
            vmax=1,
            cmap="seismic",
        )

    # If WRF says land put that point as land in MITgcm
    z[mismatch == 1] = 10.0
    # If WRF says ocean put that point as ocean in
    # MITgcm with minimum depth value of -5.0
    z[mismatch == -1] = -5.0

    # Compute mismatch Again
    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac
    omasks = [ofrac, ocnfrac, mismatch]

    for i, mask in enumerate(omasks):
        axs[1, i].pcolor(
            lon,
            lat,
            mask,
            vmin=-1,
            vmax=1,
            cmap="seismic",
        )

    # plt.colorbar(cs)
    fig.suptitle(
        "(1: MITgcm says ocean, WRF says land)\n"
        + "(-1: MITgcm says land, WRF says ocean)"
    )

    plt.savefig("lndmask.png")
    z.astype(">f4").tofile(bathy_file)
    plt.cla()
    plt.clf()
    plt.close()
