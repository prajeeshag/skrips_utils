from pathlib import Path
from typing import List


from .utils import _get_bathy_from_nml, _vgrid_from_parm04
import numpy as np
import xarray as xr
import f90nml

import logging
import typer
from cdo import Cdo  # python version


logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False)

BNDDEF = {
    "S": (slice(0, 1), slice(None)),
    "E": (slice(None), slice(-1, None)),
    "N": (slice(-1, None), slice(None)),
    "W": (slice(None), slice(0, 1)),
}


@app.command()
def nxyz():
    data_nml = f90nml.read("data")
    n = len(data_nml['parm04']['delx'])
    print(f'nx = {n}')
    n = len(data_nml['parm04']['dely'])
    print(f'ny = {n}')
    n = len(data_nml['parm04']['delz'])
    print(f'nz = {n}')


@app.command()
def gen_bnd_grid():
    """Generate MITgcm boundary map nc files"""
    z, lat, lon = _get_bathy_from_nml(Path("data"))
    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    z, delz = _vgrid_from_parm04(f90nml.read("data"))
    bndAct = []
    for bnd in BNDDEF:
        bndMask = omask[BNDDEF[bnd]]
        isboundary = np.any(bndMask == 1)
        print(f"{bnd}: {isboundary}")
        if not isboundary:
            continue
        bndAct.append(bnd)
        latitude = lat[BNDDEF[bnd][0]]
        longitude = lon[BNDDEF[bnd][1]]
        ds_out = xr.Dataset(
            {
                "lat": (
                    ["lat"],
                    latitude,
                    {"units": "degrees_north"},
                ),
                "lon": (
                    ["lon"],
                    longitude,
                    {"units": "degrees_east"},
                ),
                "da": (
                    ["lat", "lon"],
                    bndMask,
                    {"units": "1"},
                ),
            }
        )
        encoding = {var: {"_FillValue": None} for var in ds_out.variables}
        ds_out.to_netcdf(f"bndGrid{bnd}.nc", encoding=encoding)
        with open("vGrid.txt", "wt") as foutput:
            foutput.write(",".join(["{:.3f}".format(i) for i in z]))

    return bndAct


@app.command()
def gen_bnd(
    varnm: str = typer.Option(...),
    addc: float = typer.Option(0.0),
    mulc: float = typer.Option(1.0),
    out_file: Path = typer.Option(None),
    infiles: List[Path] = typer.Argument(...),
):
    """Generate MITgcm boundary conditions"""

    grid_nml = Path("data")
    cdo = Cdo()
    z, delz = _vgrid_from_parm04(f90nml.read(grid_nml))
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    bndAct = gen_bnd_grid()
    for bnd in bndAct:
        file0 = infiles[0]
        # Generate cdo weights
        gridFile = f"bndGrid{bnd}.nc"
        wgts = cdo.gennn(gridFile, input=str(file0))

        cdoOpr1 = " "
        for file in infiles:
            cdoOpr1 += f" -remap,{gridFile},{wgts} -selvar,{varnm} {file}"
        cdoOpr1 = f" -mulc,{mulc} -addc,{addc} -intlevel,{levels} -mergetime " + cdoOpr1
        arr = cdo.fillmiss(input=cdoOpr1, returnXArray=varnm)
        logger.info(f"Processing variable {varnm}; {arr.attrs}")

        if np.any(np.isnan(arr.values)):
            raise RuntimeError("Nan Values present in the boundary conditions")

        time = arr["time"]
        delta_time = (time.values[1] - time.values[0]) / np.timedelta64(1, "s")
        startdate = time[0].dt.strftime("%Y%m%d-%H%M%S").values
        enddate = time[-1].dt.strftime("%Y%m%d-%H%M%S").values
        if out_file is None:
            out_file = f"ob{bnd}_{varnm}_{startdate}_{delta_time}_{enddate}.bin"
        arr.values.astype(">f4").tofile(out_file)


def gen_ini(grid_nml, bathy_file):
    """Generate initial conditions for MITgcm"""
    pass
