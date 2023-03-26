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
