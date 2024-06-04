import logging
from pathlib import Path
from typing import List

import tempfile
import numpy as np
import xarray as xr
import typer
from cdo import Cdo

from .utils import (
    load_grid,
    fill_missing3D,
    vgrid_from_parm04,
)


app = typer.Typer(add_completion=False)

BNDDEF = {
    "W": (slice(None), slice(0, 1)),
    "S": (slice(0, 1), slice(None)),
    "E": (slice(None), slice(-1, None)),
    "N": (slice(-1, None), slice(None)),
}

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


def gen_grid(grid_file: Path, nx: int, ny: int):

    logger.info("Reading grid info")
    gA = load_grid(grid_file, nx, ny)
    ds_out = xr.Dataset(
        {
            "lat": (
                ["y", "x"],
                gA["yC"][:-1, :-1],
                {"units": "degrees_north"},
            ),
            "lon": (
                ["y", "x"],
                gA["xC"][:-1, :-1],
                {"units": "degrees_east"},
            ),
            "rA": (
                ["y", "x"],
                gA["rA"][:-1, :-1],
                {"units": "m**2", "coordinates": "lat lon"},
            ),
        }
    )
    encoding = {var: {"_FillValue": None} for var in ds_out.variables}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpfile:
        logger.info(f"Writing grid file to {tmpfile.name}")
        ds_out.to_netcdf(tmpfile.name, encoding=encoding)
    return tmpfile.name


@app.command()
def main(
    varnm: str = typer.Option(help="Name of the variable in the input file"),
    input: str = typer.Option(
        help="""
        Input can be: \n
         1. A NetCDF file. \n 
         2. A valid cdo option which will generate a NetCDF file. \n
         e.g. "-mergetime input1.nc input2.nc input3.nc"
         """,
    ),
    nx: int = typer.Option(
        help="Number of points in x-direction",
    ),
    ny: int = typer.Option(
        help="Number of points in y-direction",
    ),
    nml: Path = typer.Option(
        default=Path("./data"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm `data` namelist file",
    ),
    mitgrid: Path = typer.Option(
        default=Path("./tile001.mitgrid"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm grid file",
    ),
    timestep: int = typer.Option(
        default=1,
        help="Timestep at which data to be used for initial condition",
    ),
    ovarnm: str = typer.Option(
        default=None,
        help="""
        Name of the variable in the filename. \n
        e.g. <ovarnm>_ini.bin. If None provided it will use the <varnm>
        """,
    ),
    addc: float = typer.Option(
        default=0.0,
        help="Add a constant to the input field",
    ),
    mulc: float = typer.Option(
        default=1.0,
        help="Multiply a constant to the input field",
    ),
):
    """
    Generate initial conditions for MITgcm from a CF-compliant Netcdf data
    It assume that the levels are in meters and is from top to bottom
    """

    cdo = Cdo(tempdir='tmp/')
    grid_nml = nml
    z, _, _ = vgrid_from_parm04(grid_nml)
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    gridFile = gen_grid(mitgrid, nx, ny)

    cdoOpr1 = f" -selvar,{varnm} -seltimestep,{timestep} {input} "
    cdoOpr2 = f" -setlevel,0 -sellevidx,1 " + cdoOpr1
    cdoOpr1 = f" -merge " + cdoOpr2 + cdoOpr1
    cdoOpr1 = f" -remapnn,{gridFile} " + cdoOpr1
    cdoOpr1 = f" -mulc,{mulc} -addc,{addc} " + cdoOpr1
    cdoOpr = f" -intlevel,{levels} " + cdoOpr1
    logger.info(f"CDO operation: {cdoOpr}")

    arr = cdo.fillmiss2(input=cdoOpr, returnXArray=varnm)
    arr = arr.squeeze()
    logger.info(f"Processing variable {varnm}; {arr.attrs}")

    if np.any(np.isnan(arr.values)):
        logger.info("Nan Values present in the initial conditions")
        logger.info("Trying to fill Nan Values")
        fill_missing3D(arr.values)

    if np.any(np.isnan(arr.values)):
        raise RuntimeError("Nan Values present in the initial conditions")

    if ovarnm is None:
        ovarnm = varnm

    out_file = f"{ovarnm}_ini.bin"

    logger.info(f"Shape of IC for {varnm} is {arr.shape}")
    logger.info(f"Maximum value of IC for {varnm} is {arr.values.max()}")
    logger.info(f"Minimum value of IC for {varnm} is {arr.values.min()}")
    logger.info(f"Writing IC for {varnm} to {out_file}")
    arr.values.astype(">f4").tofile(out_file)


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
