from pathlib import Path

import numpy as np
import typer
import xarray as xr
import matplotlib.pyplot as plt

import logging

from .utils import _get_bathy_from_nml

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def match_mitgcm_lmask(
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
            "bathymetry file; \n"
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
    """Edits WRF geo_em file to match the MITgcm land mask"""

    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)
    luindex = ds["LU_INDEX"].squeeze()
    luindexNew = luindex.values.copy()
    iswater = ds.attrs["ISWATER"]
    islake = ds.attrs["ISLAKE"]
    ofrac = luindex.values.copy()
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
    plt.pcolormesh(mismatch)
    plt.colorbar()
    plt.show()

    # If MITgcm says ocean put that point as Ocean in WRF
    logger.info("Converting points WRF:land x MITgcm:ocean: to WRF:ocean")
    luindexNew[mismatch == 1] = iswater
    # If WRF says ocean put that point as ocean in
    logger.info("Converting points WRF:ocean x MITgcm:land: to WRF:lake")
    luindexNew[mismatch == -1] = islake

    # Compute mismatch Again
    ofrac = luindexNew.copy()
    if np.any(np.isnan(ofrac)):
        msg = "LU_INDEX contains NaN values"
        logger.error(msg)
        raise ValueError(msg)

    ofrac[ofrac != iswater] = 0.0  # 17 is water body in LULC of WRF
    ofrac[ofrac == iswater] = 1.0  # 17 is water body in LULC of WRF

    mismatch = ocnfrac - ofrac
    lpOcn = np.count_nonzero(mismatch)
    logger.info(f"Number of mismatch points: {lpOcn}")
    if lpOcn != 0:
        raise RuntimeError("Mismatch points still exist!!!")
    ds["LU_INDEX"].values[0, :, :] = luindexNew
    logger.info(f"Saving to file: {out_file}")
    ds.to_netcdf(out_file)


if __name__ == "__main__":
    pass
