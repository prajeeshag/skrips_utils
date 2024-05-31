import logging
from pathlib import Path
from typing import List

import tempfile
import numpy as np
import xarray as xr
import typer
from cdo import Cdo

from utils import (
    fill_missing2D,
    load_grid,
    load_bathy,
    fill_missing3D,
    vgrid_from_parm04,
)


app = typer.Typer(pretty_exceptions_show_locals=False)


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


def gen_bnd_grid(mitgrid: Path, nx: int, ny: int, bathy_file: Path):
    """Generate MITgcm boundary grid files later to be used to interpolate the boundary condition"""

    logger.info("Reading bathymetry and grid info")

    gA = load_grid(mitgrid, nx, ny)
    z = load_bathy(bathy_file, nx, ny)
    lat = gA["yC"][:-1, :-1]
    lon = gA["xC"][:-1, :-1]

    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    bndAct = []
    for bnd in BNDDEF:
        bndMask = omask[BNDDEF[bnd]]
        isboundary = np.any(bndMask == 1)
        logger.info(f"{bnd}: {isboundary}")
        if not isboundary:
            continue
        latitude = lat[BNDDEF[bnd]]
        longitude = lon[BNDDEF[bnd]]
        ds_out = xr.Dataset(
            {
                "lat": (
                    ["y", "x"],
                    latitude,
                    {"units": "degrees_north"},
                ),
                "lon": (
                    ["y", "x"],
                    longitude,
                    {"units": "degrees_east"},
                ),
                "da": (
                    ["y", "x"],
                    bndMask,
                    {"units": "1", "coordinates": "lat lon"},
                ),
            }
        )
        encoding = {var: {"_FillValue": None} for var in ds_out.variables}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpfile:
            logger.info(f"Writing {bnd} boundary grid to file {tmpfile.name}")
            ds_out.to_netcdf(tmpfile.name, encoding=encoding)
        bndAct.append((bnd, tmpfile.name))

    return bndAct


@app.command()
def make_bc(
    varnm: str = typer.Option(help="Name of the variable in the input file"),
    input: str = typer.Option(
        help="""
        Input can be: \n
         1. A NetCDF file. 
         2. A valid cdo option which will generate a NetCDF file. 
         e.g. "-mergetime input1.nc input2.nc input3.nc"
         """,
    ),
    nx: int = typer.Option(
        help="Number of points in x-direction",
    ),
    ny: int = typer.Option(
        help="Number of points in y-direction",
    ),
    bathymetry: Path = typer.Option(
        default=Path("./bathymetry.bin"),
        exists=True,
        dir_okay=False,
        help="Path to the MITgcm bathymetry file",
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
        help="Path to the MITgcm data namelist file",
    ),
    ovarnm: str = typer.Option(
        default=None,
        help="Name of the variable in the boundary filenames i.e. obE_<ovarnm>.bin. [default: varnm]",
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
    Generate boundary conditions for MITgcm from a CF-compliant Netcdf data
    It assume that the levels are in meters and is from top to bottom
    """
    cdo = Cdo()
    if nml is None:
        nml = Path("data")
    z, _, _ = vgrid_from_parm04(nml)
    levels = ",".join(["{:.3f}".format(i) for i in z])

    # generate boundary grids
    bndAct = gen_bnd_grid(mitgrid, nx, ny, bathymetry)

    for bnd, gridfile in bndAct:
        logger.info(f"Processing {bnd} boundary")

        cdoOpr1 = f" -selvar,{varnm} {input} "
        cdoOpr2 = f" -setlevel,0 -sellevidx,1 " + cdoOpr1
        cdoOpr1 = f" -merge " + cdoOpr2 + cdoOpr1
        cdoOpr1 = f" -remapnn,{gridfile} " + cdoOpr1
        cdoOpr1 = f" -mulc,{mulc} -addc,{addc} " + cdoOpr1
        cdoOpr = f" -intlevel,{levels} " + cdoOpr1
        logger.info(f"CDO operation: {cdoOpr}")

        arr = cdo.fillmiss2(input=cdoOpr, returnXArray=varnm)

        arr = arr.squeeze()

        shape = arr.shape
        is2D = len(shape) == 2

        if np.any(np.isnan(arr.values)):
            logger.info(f"NaN Values present in the boundary {bnd} conditions")
            logger.info("Trying to fill NaN Values with Nearest Neighbhour")
            if is2D:
                fill_missing2D(arr.values)
            else:
                fill_missing3D(arr.values)

        if np.any(np.isnan(arr.values)):
            raise RuntimeError(f"Nan Values present in the boundary {bnd} conditions")

        if ovarnm is None:
            ovarnm = varnm
        out_file = f"ob{bnd}_{ovarnm}.bin"

        logger.info(f"Shape of {bnd} boundary for {varnm} is {arr.shape}")
        logger.info(
            f"Maximum value of {bnd} boundary for {varnm} is {arr.values.max()}"
        )
        logger.info(
            f"Minimum value of {bnd} boundary for {varnm} is {arr.values.min()}"
        )
        logger.info(f"Writing {bnd} boundary for {varnm} to {out_file}")

        arr.values.astype(">f4").tofile(out_file)


if __name__ == "__main__":
    app()
