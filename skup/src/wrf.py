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
def match_mitgcm_lmask():
    """Edits WRF geo_em file to match the MITgcm land mask"""

    wrf_geo = Path("geo_em.d01.nc")
    grid_nml = Path("data")
    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)
    luindex = ds["LU_INDEX"].squeeze()
    landmask = ds["LANDMASK"].squeeze().values
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
    z, zlat, zlon = _get_bathy_from_nml(grid_nml)
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
    luindexNew[mismatch == 1] = iswater
    landmask[mismatch == 1] = 0

    # If WRF says ocean put that point as ocean in
    # LANDMASK does not change because we converting to LAKE
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
    logger.info(f"Editing LU_INDEX")
    ds["LU_INDEX"].values[0, :, :] = luindexNew
    logger.info(f"Editing LANDMASK")
    ds["LANDMASK"].values[0, :, :] = landmask
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
