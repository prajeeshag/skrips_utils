import glob
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

import f90nml
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
import xesmf as xe
from cdo import Cdo  # python version
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
from scipy import ndimage
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import itertools
from enum import Enum

from .utils import (
    _da2bin,
    _get_bathy_from_nml,
    _get_end_date_wps,
    _get_parm04_from_geo,
    _get_start_date_wps,
    _grid_from_parm04,
    _load_yaml,
    _vgrid_from_parm04,
    great_circle,
    quadrilateral_area_on_earth,
    load_bathy,
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = typer.Typer(pretty_exceptions_show_locals=False)

BNDDEF = {
    "W": (slice(None), slice(0, 1)),
    "S": (slice(0, 1), slice(None)),
    "E": (slice(None), slice(-1, None)),
    "N": (slice(-1, None), slice(None)),
}
GRID_VARS = [
    "xC",
    "yC",
    "dxF",
    "dyF",
    "rA",
    "xG",
    "yG",
    "dxV",
    "dyU",
    "rAz",
    "dxC",
    "dyC",
    "rAw",
    "rAs",
    "dxG",
    "dyG",
    "angleCosC",
    "angleSinC",
]


@app.command()
def ls_wrfgrid():
    nml = _get_parm04_from_geo()
    print(nml)


@app.command()
def nxyz():
    data_nml = f90nml.read("data")
    n = len(data_nml["parm04"]["delx"])
    print(f"nx = {n}")
    n = len(data_nml["parm04"]["dely"])
    print(f"ny = {n}")
    n = len(data_nml["parm04"]["delz"])
    print(f"nz = {n}")


def gen_grid():
    logger.info("Reading bathymetry and grid info")
    z, lat, lon = _get_bathy_from_nml(Path("data"))
    ds_out = xr.Dataset(
        {
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east"},
            ),
            "z": (
                ["lat", "lon"],
                z,
                {"units": "m"},
            ),
        }
    )
    encoding = {var: {"_FillValue": None} for var in ds_out.variables}
    logger.info("Writing grid file to Grid.nc")
    ds_out.to_netcdf("Grid.nc", encoding=encoding)
    z, delz = _vgrid_from_parm04(f90nml.read("data"))
    with open("vGrid.txt", "wt") as foutput:
        foutput.write(",".join(["{:.3f}".format(i) for i in z]))


def gen_grid1(grid_file: Path, nx: int, ny: int):
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
    logger.info("Writing grid file to Grid.nc")
    ds_out.to_netcdf("Grid.nc", encoding=encoding)


@app.command()
def gen_bnd_grid(
    grid_file: Path = None, nx: int = None, ny: int = None, bathy_file: Path = None
):
    """Generate MITgcm boundary map nc files"""

    logger.info("Reading bathymetry and grid info")

    if grid_file is not None:
        gA = load_grid(grid_file, nx, ny)
        z = load_bathy(bathy_file, nx, ny)
        lat = gA["yC"][:-1, :-1]
        lon = gA["xC"][:-1, :-1]
    else:
        z, lat, lon = _get_bathy_from_nml(Path("data"))
        lat, lon = np.meshgrid(lat, lon)

    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    bndAct = []
    for bnd in BNDDEF:
        bndMask = omask[BNDDEF[bnd]]
        isboundary = np.any(bndMask == 1)
        logger.info(f"{bnd}: {isboundary}")
        if not isboundary:
            continue
        bndAct.append(bnd)
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
        logger.info(f"Writing boundary grid file bndGrid{bnd}.nc")
        ds_out.to_netcdf(f"bndGrid{bnd}.nc", encoding=encoding)

    return bndAct


@app.command()
def gen_bnd(
    varnm: str,
    infiles: List[Path],
    addc=0.0,
    mulc=1.0,
    grid_file: Path = None,
    nx: int = None,
    ny: int = None,
    bathy_file: Path = None,
):
    """Generate MITgcm boundary conditions"""
    cdo = Cdo()
    grid_nml = Path("data")
    z, delz = _vgrid_from_parm04(f90nml.read(grid_nml))
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    bndAct = gen_bnd_grid(grid_file, nx, ny, bathy_file)

    for bnd in bndAct:
        logger.info(f"Processing {bnd} boundary")
        file0 = infiles[0]
        # Generate cdo weights
        gridFile = f"bndGrid{bnd}.nc"
        wgts = cdo.gennn(gridFile, input=str(file0))

        cdoOpr1 = " "
        for file in infiles:
            cdoOpr1 += f" -remap,{gridFile},{wgts} -selvar,{varnm} {file}"
        cdoOpr1 = f" -mulc,{mulc} -addc,{addc} -intlevel,{levels} -mergetime " + cdoOpr1
        arr = cdo.setmisstonn(input=cdoOpr1, returnXArray=varnm)
        arr = arr.squeeze()

        if np.any(np.isnan(arr.values)):
            logger.info("Nan Values present in the boundary conditions")
            logger.info("Trying to fill Nan Values")
            fill_missing3D(arr.values)

        if np.any(np.isnan(arr.values)):
            raise RuntimeError("Nan Values present in the boundary conditions")

        time = arr["time"]
        delta_time = (time.values[1] - time.values[0]) / np.timedelta64(1, "s")
        startdate = time[0].dt.strftime("%Y%m%d-%H%M%S").values
        enddate = time[-1].dt.strftime("%Y%m%d-%H%M%S").values
        out_file = None
        if out_file is None:
            out_file = f"ob{bnd}_{varnm}_{startdate}_{int(delta_time)}_{enddate}.bin"
        arr.values.astype(">f4").tofile(out_file)


@app.command()
def gen_ini(
    varnm: str,
    ifile: Path,
    addc=0.0,
    mulc=1.0,
    wts: Path = None,
    grid_file: Path = None,
    nx: int = None,
    ny: int = None,
):
    """Generate initial conditions for MITgcm"""
    cdo = Cdo()
    grid_nml = Path("data")
    z, delz = _vgrid_from_parm04(f90nml.read(grid_nml))
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    if grid_file is not None:
        gen_grid1(grid_file, nx, ny)
    else:
        gen_grid()
    gridFile = "Grid.nc"
    if wts is None:
        logger.info("Generating remaping weights")
        wgts = cdo.genbil(gridFile, input=str(ifile))
    elif wts.is_file():
        logger.info(f"Using remaping weights from file {wts}")
        wgts = wts
    else:
        logger.info(f"Generating remaping weights and saving it to {wts}")
        wgts = cdo.genbil(gridFile, input=str(ifile), output=str(wts))

    cdoOpr1 = " "
    cdoOpr1 += f" -remap,{gridFile},{wgts} -selvar,{varnm} -seltimestep,1 {ifile}"
    cdoOpr1 = f" -mulc,{mulc} -addc,{addc} -intlevel,{levels} -mergetime " + cdoOpr1
    logger.info(f"CDO operation: {cdoOpr1}")
    arr = cdo.setmisstonn(input=cdoOpr1, returnXArray=varnm, options="-P 8")
    arr = arr.squeeze()
    logger.info(f"Processing variable {varnm}; {arr.attrs}")

    if np.any(np.isnan(arr.values)):
        logger.info("Nan Values present in the initial conditions")
        logger.info("Trying to fill Nan Values")
        fill_missing3D(arr.values)

    if np.any(np.isnan(arr.values)):
        raise RuntimeError("Nan Values present in the initial conditions")

    out_file = None
    if out_file is None:
        out_file = f"{varnm}_ini.bin"
    logger.info(f"Saving to file {out_file}")
    arr.values.astype(">f4").tofile(out_file)


def fill_missing3D(arr):
    for i in range(arr.shape[0]):
        arr2D = arr[i, :, :]
        fill_missing_values(arr2D)
        arr[i, :, :] = arr2D


def fill_missing_values(arr):
    """
    Fill in missing values in a 2D NumPy array with their
    nearest non-missing neighbor.
    """

    # Get the indices of all missing values in the array
    missing_indices = np.argwhere(np.isnan(arr))

    # Get the shape of the array
    nrows, ncols = arr.shape

    # Define the directions to search for nearest neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Perform a spiral grid search to fill in missing values with nearest neighbors
    for r, c in missing_indices:
        for i in range(1, max(nrows, ncols)):
            for dr, dc in directions:
                nr, nc = r + i * dr, c + i * dc
                if (
                    nr >= 0
                    and nr < nrows
                    and nc >= 0
                    and nc < ncols
                    and not np.isnan(arr[nr, nc])
                ):
                    arr[r, c] = arr[nr, nc]
                    break
            if not np.isnan(arr[r, c]):
                break


def get_bnd_infiles(var, start_date, end_date, form):
    config_file = os.path.join(os.path.dirname(__file__), f"bndini/{form}.yml")
    logger.debug(f"Reading config file {config_file}")
    conf = _load_yaml(config_file)
    defaults = conf.get("defaults", {})
    return _parse_data_dir(conf[var]["data"], kwargs=defaults)


def _parse_data_dir(xlist, kwargs={}):
    data_list = []
    for x in xlist:
        logger.debug(f"Formating {x} with {kwargs}")
        xf = x.format(**kwargs)
        data_list += glob.glob(xf)
    return data_list


@app.command()
def mk_bathy(
    in_file: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of input bathymetry netcdf file",
    ),
    grid_file: Path = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="grid file",
    ),
    nx: int = typer.Option(
        None, help="no of grid points in x-dir (must be given if grid_file is given)"
    ),
    ny: int = typer.Option(
        None, help="no of grid points in y-dir (must be given if grid_file is given)"
    ),
    out_file: Path = typer.Option(
        None,
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="output bathymetry file (must be given if grid_file is given)",
    ),
):
    """
    Create bathymetry file for MITgcm:
    1. From the grid information taken from `data` namelist (Default)
    2. From the grid information from the `grid-file`, (--nx, --ny, --out-file) should be provided
    """
    BDATAVAR = {"z": "SRTM+", "elevation": "Gebco"}
    ds_input_bathy = xr.open_dataset(in_file)

    input_bathy = None
    for key in BDATAVAR:
        try:
            input_bathy = ds_input_bathy[key]
            dset = BDATAVAR[key]
            logger.info(f"Using variable `{key}`: assuming `{dset}` Bathymetry")
            break
        except KeyError:
            continue

    if input_bathy is None:
        keys = BDATAVAR.keys()
        raise KeyError(
            f"Bathymetry file doest not contain any variable with names {keys}"
        )

    if grid_file is not None:
        if (nx is None) or (ny is None):
            raise typer.BadParameter(
                "nx or ny not provided when grid_file was provided"
            )
        if out_file is None:
            raise typer.BadParameter("out_file was not when grid_file was provided")

        gD = load_grid(grid_file, nx, ny)
        grid_out = xr.Dataset(
            {
                "lat": (["y", "x"], gD["yC"][:-1, :-1], {"units": "degrees_north"}),
                "lon": (["y", "x"], gD["xC"][:-1, :-1], {"units": "degrees_east"}),
                "lat_b": (["y_b", "x_b"], gD["yG"][:, :], {"units": "degrees_north"}),
                "lon_b": (["y_b", "x_b"], gD["xG"][:, :], {"units": "degrees_east"}),
            }
        )
    else:
        logger.info("Reading `data`")
        nml = f90nml.read("data")
        usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
        if not usingsphericalpolargrid:
            raise NotImplementedError(
                "Not implemented for any other grid apart from spherical-polar grid"
            )

        logger.info("Generating grid from `data`")
        nx, ny, lon, lat = _grid_from_parm04(nml["parm04"])
        if out_file is None:
            out_file = nml["parm05"]["bathyfile"]

        grid_out = xr.Dataset(
            {
                "lat": (["lat"], lat, {"units": "degrees_north"}),
                "lon": (["lon"], lon, {"units": "degrees_east"}),
            }
        )

    logger.info("Creating regridder")
    regridder = xe.Regridder(ds_input_bathy, grid_out, "conservative")

    logger.info("Remapping Bathymery")
    dr_out = regridder(input_bathy, keep_attrs=True)

    logger.info(f"Writing to bathymetry to `{out_file}`")
    _da2bin(dr_out, out_file)


@app.command()
def clip_bathy(
    bathy_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        writable=True,
        dir_okay=False,
        file_okay=True,
        help="MITgcm bathymetry file",
    ),
    mindepth: float = typer.Option(
        -5,
        help="Minimum depth (-ve downwards), z[z > mindepth]=landvalue",
    ),
    maxdepth: float = typer.Option(
        None,
        help="Maximum depth (-ve downwards), z[z < maxdepth]= maxdepth",
    ),
    landvalue: float = typer.Option(
        100.0,
        help="Land value",
    ),
):
    """
    Clip the bathymetry of MITgcm between min and max values.
    """
    if not any([mindepth, maxdepth]):
        raise typer.BadParameter("Neither mindepth or maxdepth is given.")

    output = bathy_file

    logger.info(f"Reading bathymetry from {bathy_file}")
    z = np.fromfile(bathy_file, ">f4")
    logger.info("Clipping bathymetry")
    _clip_bathy(z, mindepth, maxdepth, landvalue)
    logger.info(f"Saving bathymetry to file {output}")
    z.astype(">f4").tofile(output)


def _clip_bathy(z, mindepth=None, maxdepth=None, landvalue=100.0):
    if mindepth:
        z[z > mindepth] = landvalue
    if maxdepth:
        z[z < maxdepth] = maxdepth


@app.command("match_wrf_lmask")
def match_wrf_lmask(
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
    """Edits MITgcm bathymetry to match the WRF land mask"""
    wrf_geo = geo
    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)
    luindex = ds["LU_INDEX"].squeeze()
    ny, nx = luindex.shape

    logger.info(f"Reading bathymetry file {bathy_file}")
    z = load_bathy(bathy_file, nx, ny)

    iswater = ds.attrs["ISWATER"]
    ofrac = luindex.values
    if np.any(np.isnan(ofrac)):
        msg = "LU_INDEX contains NaN values"
        logger.error(msg)
        raise ValueError(msg)

    ofrac[ofrac != iswater] = 0.0  # 17 is water body in LULC of WRF
    ofrac[ofrac == iswater] = 1.0  # 17 is water body in LULC of WRF

    if np.any(np.isnan(z)):
        msg = "Bathymetry contains NaN values"
        logger.error(msg)
        raise ValueError(msg)

    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac

    if np.count_nonzero(mismatch) == 0:
        logger.info("No mismatch points detected!!!")
        return

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

    mindepth = np.amax(z[z < 0])

    logger.info(f"Depth value used when creating a ocean point: {mindepth}")
    # If WRF says land put that point as land in MITgcm
    logger.info("Points were WRF:land,MITgcm:Ocean: to MITgcm:land")
    z[mismatch == 1] = 100.0
    # If WRF says ocean put that point as ocean in
    # MITgcm with minimum depth value of -5.0
    logger.info("Points were WRF:ocean,MITgcm:land: to MITgcm:ocean")
    z[mismatch == -1] = mindepth

    # Compute mismatch Again
    ocnfrac = z.copy()
    ocnfrac[ocnfrac >= 0.0] = 0.0
    ocnfrac[ocnfrac < 0.0] = 1.0
    mismatch = ocnfrac - ofrac
    lpOcn = np.count_nonzero(mismatch)
    logger.info(f"Number of mismatch points: {lpOcn}")
    if lpOcn != 0:
        raise RuntimeError("Mismatch points still exist..")
    logger.info(f"Saving bathymetry to file {bathy_file}")
    z.astype(">f4").tofile(bathy_file)


class FeatureName(Enum):
    ISLAND = "island"
    POND = "POND"
    CREEK = "CREEK"


class Features:
    def __init__(self, name: FeatureName, bathy_file: Path, nx: int, ny: int) -> None:
        self.name = name
        self.grid_nml = Path("data")
        self.out_file = bathy_file
        logger.info(f"Reading bathymetry from {self.out_file}")
        self.z = load_bathy(bathy_file, nx, ny)
        self.min_depth = np.amax(self.z[self.z < 0])
        logger.info(f"Minimum ocean depth: {self.min_depth}")
        self.min_height = 100.0

        if name == FeatureName.ISLAND:
            self.mask = np.where(self.z >= 0, 1, 0)
            self.cval = self.min_depth
        elif name == FeatureName.POND:
            self.mask = np.where(self.z < 0, 1, 0)
            self.cval = self.min_height
        elif name == FeatureName.CREEK:
            mask = np.where(self.z < 0, 1, 0)
            self._creek_mask(mask)
            self.cval = self.min_height
        else:
            raise NotImplementedError(f"Allowed feature names are {FeatureName}")

        self.shown = False
        self.boxes = []
        self._get_features()

    def _creek_mask(self, mask, n_neibhours=3):
        cmask = np.zeros(mask.shape, dtype=int)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                # Check if the pixel is black (i.e., part of a feature)
                if mask[y, x] == 1:
                    y1 = max(y - 1, 0)
                    y2 = min(y + 2, mask.shape[0] + 1)
                    x1 = max(x - 1, 0)
                    x2 = min(x + 2, mask.shape[1] + 1)
                    # Count the number of neighbors of the pixel
                    # num_neighbors = np.sum(mask[y1:y2, x1:x2]) - 1
                    nyn = min(np.sum(mask[y1:y2, x]) - 1, 1)
                    nxn = min(np.sum(mask[y, x1:x2]) - 1, 1)

                    # If the pixel has one or two neighbors, label it
                    if nxn + nyn <= 1:
                        cmask[y, x] = 1
        self.mask = cmask

    def _get_features(self):
        self.array, num_labels = ndimage.label(self.mask)
        self.labels = list(range(1, num_labels + 1))

    def max_points(self, n, array=None):
        labels = []
        if array is None:
            array = self.array.copy()
        for i in self.labels:
            npoints = np.count_nonzero(array == i)
            if npoints > 0 and npoints <= n:
                labels.append(i)
            else:
                array[array == i] = 0
        return array, labels

    def no_edge(self, array=None):
        if array is None:
            labeled_array = self.array
        labels = self.labels
        edge_pixels = np.concatenate(
            (
                labeled_array[0, :],
                labeled_array[-1, :],
                labeled_array[:, 0],
                labeled_array[:, -1],
            )
        )
        # get the labels that touch the edges
        edge_labels = np.intersect1d(labels, edge_pixels)
        # get the labels that do not touch the edges
        pool_labels = np.setdiff1d(labels, edge_labels)

        pools = labeled_array
        for i in edge_labels:
            pools[pools == i] = 0
        return pools, pool_labels

    def get_boxes(self, array, labels):
        boxes = []
        for i in labels:
            points = np.where(array == i)
            ny, nx = array.shape
            x = np.max([np.min(points[1]) - 1, 0])
            x2 = np.min([np.max(points[1]) + 1, nx])

            y = np.max([np.min(points[0]) - 1, 0])
            y2 = np.min([np.max(points[0]) + 1, ny])

            width = x2 - x + 1
            height = y2 - y + 1

            # recalibrate rectangle if it is too small
            rwidth = np.max([width, nx // 100])
            rheight = np.max([height, ny // 100])
            x = x - (rwidth - width) // 2
            y = y - (rheight - height) // 2
            width = rwidth
            height = rheight

            boxes.append(
                patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor="k", facecolor="none"
                )
            )
        return boxes

    def nn_average(self, _gx, _gy, avgtype="normal") -> bool:
        logger.info(f"Averaging ({avgtype}) for point {(_gy,_gx)}")
        ny, nx = self.z.shape
        y1, y2 = max(_gy - 1, 0), min(_gy + 2, ny + 1)
        x1, x2 = max(_gx - 1, 0), min(_gx + 2, nx + 1)
        _z = self.z[y1:y2, x1:x2]
        if avgtype == "deepen":
            _z = _z[_z < self.z[_gy, _gx]]
        else:
            _z = _z[_z < 0.0]
        if len(_z) < 1:
            return False
        self.z[_gy, _gx] = np.mean(_z)
        return True

    def edit_features(self, n_points: int = 1, min_depth=None):
        """Edit features"""

        def delete_islands(event):
            logger.info(f"Deleting {self.name}")

            if self.name == FeatureName.ISLAND:
                all_done = False
                while not all_done:
                    all_done = True
                    for label in labels:
                        idx = zip(*np.where(array == label))
                        for j, i in idx:
                            if self.z[j, i] < 0.0:  # if ocean
                                continue
                            all_done = all_done and self.nn_average(i, j)
            else:
                for label in labels:
                    self.z[array == label] = self.cval

            logger.info(f"Saving bathymetry to file {self.out_file}")
            self.z.astype(">f4").tofile(self.out_file)
            plt.close()

        levels = list(np.linspace(-3000, -200, 10))[:-1] + list(
            np.linspace(-200, 0, 21)
        )
        levels = [-0.0000001 if item == 0.0 else item for item in levels]
        cmap = plt.cm.jet
        cmap.set_over("white")
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        with plt.ioff():
            fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.z, cmap=cmap, norm=norm)
        ax.set_aspect("equal")
        array, labels = self.max_points(n_points)
        boxes = self.get_boxes(array, labels)
        nf = len(boxes)
        logger.info(f"Number of {self.name} with grid points <= {n_points}: {nf} ")
        if nf == 0:
            logger.info(f"No {self.name} detected with grid points <= {n_points}")
            return

        for rect in boxes:
            ax.add_patch(rect)
        fig.canvas.draw()

        fig.subplots_adjust(bottom=0.2)
        # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axdel = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bndel = Button(axdel, "Delete")
        bndel.on_clicked(delete_islands)
        plt.colorbar(mesh)
        plt.show()


@app.command("del_islands")
def del_islands(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.ISLAND, bathy_file, nx, ny).edit_features(n_points=n_points)


@app.command("del_ponds")
def del_ponds(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.POND, bathy_file, nx, ny).edit_features(n_points=n_points)


@app.command("del_creeks")
def del_creeks(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.CREEK, bathy_file, nx, ny).edit_features(n_points=n_points)


def remove_plt_keymaps():
    for item in plt.rcParams:
        if item.startswith("keymap."):
            for i in plt.rcParams[item]:
                plt.rcParams[item].remove(i)
                # print(f"{item} - {i}")


class EditBathy:
    def __init__(self, bathy_file, nx, ny) -> None:
        remove_plt_keymaps()
        self.out_file = bathy_file
        logger.info(f"Reading bathymetry from {self.out_file}")
        self.z = load_bathy(bathy_file, nx, ny)
        self.z[self.z >= 0] = 1000.0
        self.min_depth = np.amax(self.z[self.z < 0])
        logger.info(f"Minimum ocean depth {self.min_depth}")
        self.lnd_val = 100.0
        levels = list(np.linspace(-3000, -200, 10))[:-1] + list(
            np.linspace(-200, 0, 21)
        )
        levels = [-0.0000001 if item == 0.0 else item for item in levels]
        cmap = plt.cm.jet
        cmap.set_over("white")
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        with plt.ioff():
            self.fig, self.ax = plt.subplots()
        self.mesh = self.ax.pcolormesh(self.z, cmap=cmap, norm=norm, picker=True)
        self.ax.set_aspect("equal")
        plt.colorbar(self.mesh)

        # self.mesh = self.ax.pcolormesh(pools, cmap=self.cmap, norm=self.norm)
        self.fig.canvas.mpl_connect("button_press_event", self._on_pick)
        self.fig.subplots_adjust(bottom=0.2)
        self.axsave = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.bndel = Button(self.axsave, "Save")
        self.bndel.on_clicked(self.save_z)

        _ = zoom_factory(self.ax)
        _ = panhandler(self.fig, button=2)
        plt.show()

    def save_z(self, event):
        logger.info(f"Saving to file {self.out_file}")
        self.z.astype(">f4").tofile(self.out_file)

    def _on_pick(self, event):
        logger.debug("_on_pick")
        mouseevent = event
        key = event.key

        if mouseevent.xdata is None or mouseevent.ydata is None:
            return

        logger.debug(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, key=%s"
            % (
                "double" if mouseevent.dblclick else "single",
                mouseevent.button,
                mouseevent.x,
                mouseevent.y,
                mouseevent.xdata,
                mouseevent.ydata,
                key,
            )
        )
        if key is None:
            return

        if mouseevent.button is not MouseButton.LEFT:
            return

        _gx = int(math.floor(mouseevent.xdata))
        _gy = int(math.floor(mouseevent.ydata))

        if key == "a":
            if not self.nn_average(_gx, _gy):
                return
        elif key == "d":
            if not self.nn_average(_gx, _gy, avgtype="deepen"):
                return
        elif key == "l":
            # click left mouse button to make a land point
            self.z[_gy, _gx] = self.lnd_val
        elif key == "o":
            # click right mouse button to make a ocean point
            _z = self.z[_gy, _gx]
            if _z < 0:
                return
            self.z[_gy, _gx] = self.min_depth

        self.mesh.set_array(self.z)
        self.fig.canvas.draw()

    def nn_average(self, _gx, _gy, avgtype="normal") -> bool:
        logger.info(f"Averaging ({avgtype}) for point {(_gy,_gx)}")
        ny, nx = self.z.shape
        y1, y2 = max(_gy - 1, 0), min(_gy + 2, ny + 1)
        x1, x2 = max(_gx - 1, 0), min(_gx + 2, nx + 1)
        _z = self.z[y1:y2, x1:x2]
        if avgtype == "deepen":
            z = self.z[_gy, _gx]
            z_ = _z[_z < z]
        else:
            z_ = _z[_z < 0.0]
        logger.info(f"points {z_.shape}, {list(z_)} {(y1,y2,x1,x2)}")
        if len(z_) < 1:
            return False
        self.z[_gy, _gx] = np.mean(z_)
        return True


@app.command("edit_bathy")
def edit_bathy(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
):
    """Opens up a GUI to click and edit Bathymetry"""
    EditBathy(bathy_file, nx, ny)


@app.command()
def ini2nc():
    data_nml = Path("data")
    nml_data = f90nml.read(data_nml)
    nx, ny, lon, lat = _grid_from_parm04(nml_data["parm04"])
    depth, delz = _vgrid_from_parm04(nml_data)
    nz = len(depth)
    filenms = {
        "uvelinitfile": "uo",
        "vvelinitfile": "vo",
        "hydrogthetafile": "to",
        "hydrogsaltfile": "so",
    }
    parm05 = nml_data["parm05"]
    for filnm in filenms:
        fil = parm05.get(filnm, None)
        if fil is None:
            logger.warning(f"Could not find {filnm} in `data:parm05` namelist")
            continue
        if not Path(fil).is_file():
            logger.warning(f"The file {fil} for `data:parm05:{filnm}` does not exist")
            continue

        fdata = np.fromfile(fil, ">f4").reshape(nz, ny, nx)

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], lat, {"units": "degrees_north", "axis": "Y"}),
                "lon": (["lon"], lon, {"units": "degrees_east", "axis": "X"}),
                "depth": (
                    ["depth"],
                    depth,
                    {"units": "m", "positive": "down", "axis": "Z"},
                ),
                filenms[filnm]: (
                    ["depth", "lat", "lon"],
                    fdata,
                ),
            }
        )
        encoding = {var: {"_FillValue": None} for var in ds_out.variables}
        logger.info(f"Writing file {filnm}.nc")
        ds_out.to_netcdf(f"{filnm}.nc", encoding=encoding)


def _get_factors(n, maxq=1):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            if n / i < maxq:
                break
            factors.append(i)
    return factors


def _get_peslt(pelist, n):
    # Get all values less than or equal to n
    new_list = [val for val in pelist if val <= n]
    # Remove all values less than or equal to x from the original list
    old_list = [val for val in pelist if val > n]
    return new_list, old_list


@app.command()
def ls_decomp(
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    min_points: int = typer.Option(20),
    cpus_per_node: int = typer.Option(None),
):
    """List Possible decompositions for MITgcm given Nx and Ny"""
    decompAll = _all_decomp(nx, ny, min_points)

    pelist = list(decompAll.keys())
    pelist.sort()
    npes_per_node = 1
    if cpus_per_node:
        npes_per_node = cpus_per_node

    pelistCopy = pelist.copy()
    nodes = 1

    peNode = {}
    while pelistCopy:
        ncpus = int(nodes * npes_per_node)
        pelistNode, pelistCopy = _get_peslt(pelistCopy, ncpus)
        if not pelistNode:
            nodes += 1
            continue
        peNode[nodes] = max(pelistNode)
        nodes += 1

    nodeDecomp = {}
    print("# Nodes, Npes, Nx, Ny ")
    for node in peNode:
        npes = peNode[node]
        dlist = decompAll[npes]
        dcomp = _best_decomp(dlist)
        nodeDecomp[node] = dcomp
        nPx, nPy = dcomp
        nSx = int(nx / nPx)
        nSy = int(ny / nPy)
        print(f"{node} {npes} {nPx} {nPy} {nSx} {nSy}")


def _best_decomp(dlist):
    decomp0 = dlist[0]
    adiff0 = abs(decomp0[0] - decomp0[1])
    for decomp in dlist[1:]:
        adiff = abs(decomp[0] - decomp[1])
        if adiff < adiff0:
            decomp0 = decomp
            adiff = adiff0
    return decomp0


def _all_decomp(nx, ny, min_points):
    xfac = _get_factors(nx, min_points)
    yfac = _get_factors(ny, min_points)
    decompAll = {}
    for nxpes in xfac:
        for nypes in yfac:
            npes = int(nxpes * nypes)
            dlist = decompAll.get(npes, [])
            dlist.append((nxpes, nypes))
            decompAll[npes] = dlist
    return decompAll


@app.command()
def grid2nc(
    in_file: Path = typer.Option(...),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    out_file: Path = typer.Option(None),
) -> None:
    if out_file is None:
        out_file = Path(in_file).name
        out_file = out_file.replace(".bin", "")
        out_file = out_file + ".nc"

    logger.info(f"Loading grid from {in_file}")
    gdata = load_grid(in_file, nx, ny)
    logger.info(f"Writing grid to {out_file}")
    dump_grid_nc(gdata, out_file)


@app.command(hidden=True)
def test_grid(in_file: Path, fac: float = 1.0):
    ds = xr.open_dataset(in_file)
    xC, yC = ds["xC"].values, ds["yC"].values
    dxC, dyC = ds["dxC"].values, ds["dyC"].values
    xG, yG = ds["xG"].values, ds["yG"].values
    dxG, dyG = ds["dxG"].values, ds["dyG"].values
    rA, rAz = ds["rA"].values, ds["rAz"].values

    ny, nx = xC.shape
    JY, IX = np.meshgrid(range(ny), range(nx))
    print(f"{nx}, {ny}")
    # dxG
    print("testing dxG calculation!")
    X1, X2 = xG[:, 0:-1], xG[:, 1:]
    Y1, Y2 = yG[:, 0:-1], yG[:, 1:]
    D = dxG[:, 0:-1]

    for lon1, lon2, lat1, lat2, d in zip(*map(np.ravel, [X1, X2, Y1, Y2, D])):
        dc = great_circle(lon1, lat1, lon2, lat2)
        d1 = d * fac
        # print(f"{dc} - {d1}")
        assert math.isclose(dc, d1, rel_tol=0.00001), f"{dc} - {d1}"

    print("testing dyG calculation!")
    X1, X2 = xG[0:-1, :], xG[1:, :]
    Y1, Y2 = yG[0:-1, :], yG[1:, :]
    D = dyG[0:-1, :]
    for lon1, lon2, lat1, lat2, d in zip(*map(np.ravel, [X1, X2, Y1, Y2, D])):
        dc = great_circle(lon1, lat1, lon2, lat2)
        d1 = d * fac
        # print(f"{dc} - {d1}")
        assert math.isclose(dc, d1, rel_tol=0.00001)

    print("testing dyC calculation!")
    X1, X2 = xC[0:-1, :], xC[1:, :]
    Y1, Y2 = yC[0:-1, :], yC[1:, :]
    D = dyC[1:-1, :]
    for lon1, lon2, lat1, lat2, d in zip(*map(np.ravel, [X1, X2, Y1, Y2, D])):
        dc = great_circle(lon1, lat1, lon2, lat2)
        d1 = d * fac
        # print(f"{dc} - {d1}")
        assert math.isclose(dc, d1, rel_tol=0.00001)

    print("testing dxC calculation!")
    X1, X2 = xC[:-1, 0:-2], xC[:-1, 1:-1]
    Y1, Y2 = yC[:-1, 0:-2], yC[:-1, 1:-1]
    D = dxC[:-1, 1:-1]
    jy = JY[:-1, 1:-1]
    ix = IX[:-1, 1:-1]
    for lon1, lon2, lat1, lat2, d, i, j in zip(
        *map(np.ravel, [X1, X2, Y1, Y2, D, jy, ix])
    ):
        dc = great_circle(lon1, lat1, lon2, lat2)
        d1 = d * fac
        assert math.isclose(
            dc, d1, rel_tol=0.00001
        ), f"{dc} - {d1} - {j},{i},{lon1},{lon2},{lat1},{lat2}"

    print("testing rA calculation!")
    Ax, Ay = xG[:-1, :-1], yG[:-1, :-1]
    Bx, By = xG[:-1, 1:], yG[:-1, 1:]
    Cx, Cy = xG[1:, 1:], yG[1:, 1:]
    Dx, Dy = xG[1:, :-1], yG[1:, :-1]
    Area = rA[:-1, :-1] * fac * fac
    AreaCp = quad_area_a(Ax, Bx, Cx, Dx, Ay, By, Cy, Dy)
    for ax, ay, bx, by, cx, cy, dx, dy, area, areaCp in zip(
        *map(np.ravel, [Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Area, AreaCp])
    ):
        msg = f"{area} - {areaCp} : {[ay,ax,by,bx,cy,cx,dy,dx]}"
        assert math.isclose(area, areaCp, rel_tol=0.00001), msg

    print("testing rAz calculation!")
    Ax, Ay = xC[:-2, :-2], yC[:-2, :-2]
    Bx, By = xC[:-2, 1:-1], yC[:-2, 1:-1]
    Cx, Cy = xC[1:-1, 1:-1], yC[1:-1, 1:-1]
    Dx, Dy = xC[1:-1, :-2], yC[1:-1, :-2]
    Area = rAz[1:-1, 1:-1] * fac * fac
    AreaCp = quad_area_a(Ax, Bx, Cx, Dx, Ay, By, Cy, Dy)
    for ax, ay, bx, by, cx, cy, dx, dy, area, areaCp in zip(
        *map(np.ravel, [Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Area, AreaCp])
    ):
        msg = f"{area} - {areaCp} : {[ay,ax,by,bx,cy,cx,dy,dx]}"
        assert math.isclose(area, areaCp, rel_tol=0.00001), msg


@app.command()
def wrfgrid(
    in_file: Path = typer.Option(
        Path("geo_em.d01.nc"), readable=True, exists=True, dir_okay=False
    )
):
    """
    Create MITgcm grid from geo_em.d??.nc file of WRF
    """
    """
    xG  -> (j=0, i=0), (j=0, i=nx  ), (j=ny,  i=nx  ), (j=ny,  i=0)
    yG  -> (j=0, i=0), (j=0, i=nx  ), (j=ny,  i=nx  ), (j=ny,  i=0)
    xC  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    yC  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    """

    def print_info(var):
        min_val = np.amin(gA[var][:, :])
        max_val = np.amax(gA[var][:, :])
        logger.info(f"{var} min = {min_val}")
        logger.info(f"{var} max = {max_val}")

    sl = slice(None, None, None)
    sl1 = slice(None, -1, None)
    vMap = {
        "xG": ["XLONG_C", sl, sl],
        "yG": ["XLAT_C", sl, sl],
        "xC": ["XLONG_M", sl1, sl1],
        "yC": ["XLAT_M", sl1, sl1],
        "xU": ["XLONG_U", sl1, sl],
        "yU": ["XLAT_U", sl1, sl],
        "xV": ["XLONG_V", sl, sl1],
        "yV": ["XLAT_V", sl, sl1],
    }

    geo_ds = xr.open_dataset(in_file)
    XLAT_C = geo_ds["XLAT_C"].squeeze().values.astype(np.float64)
    nyp1, nxp1 = XLAT_C.shape

    gA = {}
    for var in GRID_VARS + ["xU", "yU", "xV", "yV"]:
        gA[var] = np.zeros_like(XLAT_C)

    for var in vMap:
        vin = vMap[var][0]
        dy = vMap[var][1]
        dx = vMap[var][2]
        fld = geo_ds[vin].squeeze().values.astype(np.float64)
        gA[var][dy, dx] = fld

    # dxG
    # dxG -> (j=0, i=0), (j=0, i=nx-1), (j=ny,  i=nx-1), (j=ny,  i=0)
    logger.info("computing dxG")
    X1 = gA["xG"][:, :-1]
    Y1 = gA["yG"][:, :-1]
    X2 = gA["xG"][:, 1:]
    Y2 = gA["yG"][:, 1:]
    gA["dxG"][:, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxG"][:, -1] = gA["dxG"][:, -2]
    print_info("dxG")

    # dyG -> (j=0, i=0), (j=0, i=nx  ), (j=ny-1,i=nx  ), (j=ny-1,i=0)
    logger.info("computing dyG")
    X1 = gA["xG"][:-1, :]
    Y1 = gA["yG"][:-1, :]
    X2 = gA["xG"][1:, :]
    Y2 = gA["yG"][1:, :]
    gA["dyG"][:-1, :] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyG"][-1, :] = gA["dyG"][-2, :]
    print_info("dyG")

    # dxC -> (j=0, i=1), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing dxC")
    X1 = gA["xC"][:-1, :-2]
    Y1 = gA["yC"][:-1, :-2]
    X2 = gA["xC"][:-1, 1:-1]
    Y2 = gA["yC"][:-1, 1:-1]
    gA["dxC"][:-1, 1:-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxC"][:, 0] = gA["dxC"][:, 1]
    gA["dxC"][:, -1] = gA["dxC"][:, -2]
    gA["dxC"][-1, :] = gA["dxC"][-2, :]
    print_info("dxC")

    # dyC -> (j=1, i=0), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dyC")
    X1 = gA["xC"][:-2, :-1]
    Y1 = gA["yC"][:-2, :-1]
    X2 = gA["xC"][1:-1, :-1]
    Y2 = gA["yC"][1:-1, :-1]
    gA["dyC"][1:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyC"][0, :] = gA["dyC"][1, :]
    gA["dyC"][-1, :] = gA["dyC"][-2, :]
    gA["dyC"][:, -1] = gA["dyC"][:, -2]
    print_info("dyC")

    # dxF -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dxF")
    X1 = gA["xU"][:-1, :-1]
    Y1 = gA["yU"][:-1, :-1]
    X2 = gA["xU"][:-1, 1:]
    Y2 = gA["yU"][:-1, 1:]
    gA["dxF"][:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxF"][:, -1] = gA["dxF"][:, -2]
    gA["dxF"][-1, :] = gA["dxF"][-2, :]
    print_info("dxF")

    # dyF -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing dyF")
    X1 = gA["xV"][:-1, :-1]
    Y1 = gA["yV"][:-1, :-1]
    X2 = gA["xV"][1:, :-1]
    Y2 = gA["yV"][1:, :-1]
    gA["dyF"][:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyF"][:, -1] = gA["dyF"][:, -2]
    gA["dyF"][-1, :] = gA["dyF"][-2, :]
    print_info("dxF")

    # dxV -> (j=0, i=1), (j=0, i=nx-1), (j=ny,  i=nx-1), (j=ny,  i=1)
    logger.info("computing dxV")
    X1 = gA["xV"][:-1, :-2]
    Y1 = gA["yV"][:-1, :-2]
    X2 = gA["xV"][:-1, 1:-1]
    Y2 = gA["yV"][:-1, 1:-1]
    gA["dxV"][:-1, 1:-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dxV"][:, 0] = gA["dxV"][:, 1]
    gA["dxV"][:, -1] = gA["dxV"][:, -2]
    gA["dxV"][-1, :] = gA["dxV"][-2, :]
    print_info("dxV")

    # dyU -> (j=1, i=0), (j=1, i=nx  ), (j=ny-1,i=nx  ), (j=ny-1,i=0)
    logger.info("computing dyU")
    X1 = gA["xU"][:-2, :-1]
    Y1 = gA["yU"][:-2, :-1]
    X2 = gA["xU"][1:-1, :-1]
    Y2 = gA["yU"][1:-1, :-1]
    gA["dyU"][1:-1, :-1] = great_circle_a(X1, X2, Y1, Y2)
    gA["dyU"][0, :] = gA["dyU"][1, :]
    gA["dyU"][-1, :] = gA["dyU"][-2, :]
    gA["dyU"][:, -1] = gA["dyU"][:, -2]
    print_info("dyU")

    # rA  -> (j=0, i=0), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing rA")
    X1, Y1 = gA["xG"][:-1, :-1], gA["yG"][:-1, :-1]
    X2, Y2 = gA["xG"][:-1, 1:], gA["yG"][:-1, 1:]
    X3, Y3 = gA["xG"][1:, 1:], gA["yG"][1:, 1:]
    X4, Y4 = gA["xG"][1:, :-1], gA["yG"][1:, :-1]
    gA["rA"][:-1, :-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rA"][-1, :] = gA["rA"][-2, :]
    gA["rA"][:, -1] = gA["rA"][:, -2]
    print_info("rA")

    # rAz -> (j=1, i=1), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing rAz")
    X1, Y1 = gA["xC"][:-2, :-2], gA["yC"][:-2, :-2]
    X2, Y2 = gA["xC"][:-2, 1:-1], gA["yC"][:-2, 1:-1]
    X3, Y3 = gA["xC"][1:-1, 1:-1], gA["yC"][1:-1, 1:-1]
    X4, Y4 = gA["xC"][1:-1, :-2], gA["yC"][1:-1, :-2]
    gA["rAz"][1:-1, 1:-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAz"][0, :] = gA["rAz"][1, :]
    gA["rAz"][:, 0] = gA["rAz"][:, 1]
    gA["rAz"][-1, :] = gA["rAz"][-2, :]
    gA["rAz"][:, -1] = gA["rAz"][:, -2]
    print_info("rAz")

    # rAw -> (j=0, i=1), (j=0, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=1)
    logger.info("computing rAw")
    X1, Y1 = gA["xV"][:-1, :-2], gA["yV"][:-1, :-2]
    X2, Y2 = gA["xV"][:-1, 1:-1], gA["yV"][:-1, 1:-1]
    X3, Y3 = gA["xV"][1:, 1:-1], gA["yV"][1:, 1:-1]
    X4, Y4 = gA["xV"][1:, :-2], gA["yV"][1:, :-2]
    gA["rAw"][:-1, 1:-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAw"][:, 0] = gA["rAw"][:, 1]
    gA["rAw"][:, -1] = gA["rAw"][:, -2]
    gA["rAw"][-1, :] = gA["rAw"][-2, :]
    print_info("rAw")

    # rAs -> (j=1, i=0), (j=1, i=nx-1), (j=ny-1,i=nx-1), (j=ny-1,i=0)
    logger.info("computing rAs")
    X1, Y1 = gA["xU"][:-2, :-1], gA["yU"][:-2, :-1]
    X2, Y2 = gA["xU"][:-2, 1:], gA["yU"][:-2, 1:]
    X3, Y3 = gA["xU"][1:-1, 1:], gA["yU"][1:-1, 1:]
    X4, Y4 = gA["xU"][1:-1, :-1], gA["yU"][1:-1, :-1]
    gA["rAs"][1:-1, :-1] = quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4)
    gA["rAs"][0, :] = gA["rAs"][1, :]
    gA["rAs"][-1, :] = gA["rAs"][-2, :]
    gA["rAs"][:, -1] = gA["rAs"][:, -2]
    print_info("rAs")

    out_file_prefix = "tile001.mitgrid"
    out_file = f"{out_file_prefix}.nc"
    logger.info(f"writing {out_file}")
    dump_grid_nc(gA, out_file)

    out_file = f"{out_file_prefix}"
    logger.info(f"writing {out_file}")
    dump_grid(gA, out_file)


def dump_grid_nc(gA, out_file):
    datavars = {}
    for varnm in GRID_VARS[:-2]:
        datavars[varnm] = (("ny1", "nx1"), gA[varnm])
    ds = xr.Dataset(data_vars=datavars)
    ds.to_netcdf(out_file)


def great_circle_a(X1, X2, Y1, Y2):
    dc = list(itertools.starmap(great_circle, zip(*map(np.ravel, [X1, Y1, X2, Y2]))))
    return np.array(dc).reshape(X1.shape)


def quad_area_a(X1, X2, X3, X4, Y1, Y2, Y3, Y4, nprocs: int = None):
    if nprocs is None:
        pool = Pool()
    else:
        pool = Pool(nprocs)

    area = pool.starmap(
        quadrilateral_area_on_earth,
        (
            ((y1, x1), (y2, x2), (y3, x3), (y4, x4))
            for x1, x2, x3, x4, y1, y2, y3, y4 in zip(
                *map(np.ravel, [X1, X2, X3, X4, Y1, Y2, Y3, Y4])
            )
        ),
    )
    pool.close()
    return np.array(area).reshape(X1.shape)


def load_grid(grid_file: Path, nx: int, ny: int) -> Dict:
    nx1, ny1 = nx + 1, ny + 1
    nxy1 = nx1 * ny1
    fdata = np.fromfile(grid_file, ">f8")
    nvars = int(fdata.shape[0] / nxy1)
    if nvars not in [len(GRID_VARS), len(GRID_VARS) - 2]:
        raise ValueError(
            f"{grid_file} does not contain enough variables needed for a mitgcm grid"
        )
    nele1 = nvars * nxy1
    nele = fdata.shape[0]
    if nvars * nxy1 != fdata.shape[0]:
        raise ValueError(
            f"nvars*(nx+1)*(ny+1) != shape of the data read: {nele1} != {nele}"
        )
    fdata = fdata.reshape([nvars, ny1, nx1])
    gridA = {}
    for i in range(nvars):
        gridA[GRID_VARS[i]] = fdata[i, :, :]
    return gridA


def dump_grid(grid: Dict, out_file: Path) -> None:
    nvars = len(GRID_VARS) - 2
    ny, nx = grid["xC"].shape
    fdata = np.zeros((nvars, ny, nx), dtype=np.float64)
    for i, var in enumerate(GRID_VARS[:-2]):
        fdata[i, :, :] = grid[var][:, :]
    fdata.astype(">f8").tofile(out_file)
