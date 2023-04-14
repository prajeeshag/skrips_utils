from pathlib import Path

import numpy as np
import typer
import xarray as xr
import matplotlib.pyplot as plt

import logging

from .utils import load_bathy

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
    geo: Path = typer.Option(
        "geo_em.d01.nc",
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="WRF geo_em.d??.nc file",
    ),
    bathy_file: Path = typer.Option(
        "bathymetry.bin",
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
        help="Bathymetry file",
    ),
):
    """Edits WRF geo_em file to match the MITgcm land mask"""

    wrf_geo = geo
    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)
    luindex = ds["LU_INDEX"].squeeze()
    landmask = ds["LANDMASK"].squeeze().values
    landusef = ds["LANDUSEF"].squeeze().values
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

    ny, nx = landmask.shape

    logger.info("Reading bathymetry file")
    z = load_bathy(bathy_file, nx, ny)
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
    if lpOcn == 0:
        logger.info(f"No mismatch points detected!!!")
        return
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
    for j, i in zip(*np.where(mismatch == 1)):
        luindexNew[j, i] = iswater
        landmask[j, i] = 0
        landusef[:, j, i] = 0.0
        landusef[iswater - 1, j, i] = 1.0

    # If WRF says ocean put that point as ocean in
    # LANDMASK does not change because we converting to LAKE
    logger.info("Converting points WRF:ocean x MITgcm:land: to WRF:lake")
    for j, i in zip(*np.where(mismatch == -1)):
        luindexNew[j, i] = islake
        landusef[:, j, i] = 0.0
        landusef[islake - 1, j, i] = 1.0

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
    logger.info("Editing LU_INDEX")
    ds["LU_INDEX"].values[0, :, :] = luindexNew
    logger.info("Editing LANDMASK")
    ds["LANDMASK"].values[0, :, :] = landmask
    logger.info("Editing LANDUSEF")
    ds["LANDUSEF"].values[0, :, :, :] = landusef
    out_file = f"mod_{wrf_geo}"
    logger.info(f"Saving to file: {out_file}")
    encode = {}
    for var in ds.data_vars:
        if var == "Times":
            encode[var] = {
                "char_dim_name": "DateStrLen",
                "zlib": True,
            }
            continue
        encode[var] = {"_FillValue": None}
    ds.to_netcdf(out_file, format="NETCDF4", encoding=encode)


if __name__ == "__main__":
    pass
