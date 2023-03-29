import os
from pathlib import Path
import matplotlib.patches as patches
from typing import List
import glob

from .utils import (
    _get_bathy_from_nml,
    _vgrid_from_parm04,
    _get_parm04_from_geo,
    _grid_from_parm04,
    _da2bin,
    _get_bathyfile_name,
    _get_end_date_wps,
    _get_start_date_wps,
    _load_yaml,
)
import numpy as np
import xarray as xr
import f90nml
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory
from matplotlib.backend_bases import MouseButton
from scipy import ndimage
import math

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
    "W": (slice(None), slice(0, 1)),
    "S": (slice(0, 1), slice(None)),
    "E": (slice(None), slice(-1, None)),
    "N": (slice(-1, None), slice(None)),
}


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


@app.command()
def gen_grid():
    """Generate MITgcm Grid file"""

    logger.info("Reading bathymetry and grid info")
    z, lat, lon = _get_bathy_from_nml(Path("data"))
    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
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
    logger.info(f"Writing grid file to Grid.nc")
    ds_out.to_netcdf(f"Grid.nc", encoding=encoding)
    z, delz = _vgrid_from_parm04(f90nml.read("data"))
    with open("vGrid.txt", "wt") as foutput:
        foutput.write(",".join(["{:.3f}".format(i) for i in z]))


@app.command()
def gen_bnd_grid():
    """Generate MITgcm boundary map nc files"""

    logger.info("Reading bathymetry and grid info")
    z, lat, lon = _get_bathy_from_nml(Path("data"))
    omask = np.array(z.shape, dtype=int)
    omask = np.where(z < 0, 1, 0)
    z, delz = _vgrid_from_parm04(f90nml.read("data"))
    bndAct = []
    for bnd in BNDDEF:
        bndMask = omask[BNDDEF[bnd]]
        isboundary = np.any(bndMask == 1)
        logger.info(f"{bnd}: {isboundary}")
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
        logger.info(f"Writing boundary grid file bndGrid{bnd}.nc")
        ds_out.to_netcdf(f"bndGrid{bnd}.nc", encoding=encoding)
    with open("vGrid.txt", "wt") as foutput:
        foutput.write(",".join(["{:.3f}".format(i) for i in z]))

    return bndAct


@app.command()
def gen_bnd(varnm: str, infiles: List[Path], addc=0.0, mulc=1.0):
    """Generate MITgcm boundary conditions"""
    grid_nml = Path("data")
    cdo = Cdo()
    z, delz = _vgrid_from_parm04(f90nml.read(grid_nml))
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    bndAct = gen_bnd_grid()

    logger.info("Getting startdate and enddate from `namelist.wps`")
    start_date = _get_start_date_wps()
    end_date = _get_end_date_wps()
    logger.info(f"Start Date {start_date}")
    logger.info(f"End Date {end_date}")

    BNDVAR = ["t", "v", "u", "s"]
    for bnd in bndAct:
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
        logger.info(f"Processing variable {varnm}; {arr.attrs}")

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
def gen_ini(varnm: str, ifile: Path, addc=0.0, mulc=1.0, wts: Path = None):
    """Generate initial conditions for MITgcm"""
    grid_nml = Path("data")
    cdo = Cdo()
    z, delz = _vgrid_from_parm04(f90nml.read(grid_nml))
    levels = ",".join(["{:.3f}".format(i) for i in z])
    # generate boundary grids
    gen_grid()
    gridFile = "Grid.nc"
    if wts is None:
        logger.info(f"Generating remaping weights")
        wgts = cdo.gencon(gridFile, input=str(ifile))
    elif wts.is_file():
        logger.info(f"Using remaping weights from file {wts}")
        wgts = wts
    else:
        logger.info(f"Generating remaping weights and saving it to {wts}")
        wgts = cdo.gencon(gridFile, input=str(ifile), output=str(wts))

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
    """Fill in missing values in a 2D NumPy array with their nearest non-missing neighbor."""

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


@app.command(name="mk_bathy")
def make_bathy(
    in_file: str = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path of input bathymetry netcdf file",
    ),
):
    """
    Create bathymetry file for MITgcm from the grid
    information taken from `data` namelist
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

    logger.info(f"Reading `data`")
    nml = f90nml.read("data")
    usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
    if not usingsphericalpolargrid:
        raise NotImplementedError(
            "bathymetry create is only implemented for spherical-polar grid"
        )

    logger.info(f"Generating grid from `data`")
    nx, ny, lon, lat = _grid_from_parm04(nml["parm04"])
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


@app.command("clip_bathy")
def clip_bathymetry(
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
    clip_bathy(z, mindepth, maxdepth, landvalue)
    logger.info(f"Saving bathymetry to file {output}")
    z.astype(">f4").tofile(output)


def clip_bathy(z, mindepth=None, maxdepth=None, landvalue=100.0):
    if mindepth:
        z[z > mindepth] = landvalue
    if maxdepth:
        z[z < maxdepth] = maxdepth


@app.command("match_wrf_lmask")
def match_wrf_lmask():
    """Edits MITgcm bathymetry to match the WRF land mask"""
    wrf_geo = "geo_em.d01.nc"
    grid_nml = Path("data")
    logger.info(f"Reading {wrf_geo}")
    ds = xr.open_dataset(wrf_geo)

    out_file = _get_bathyfile_name()

    logger.info(f"Reading bathymetry file {out_file}")
    z, zlat, zlon = _get_bathy_from_nml(grid_nml)

    luindex = ds["LU_INDEX"].squeeze()

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
        logger.info(f"No mismatch points detected!!!")
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
    logger.info(f"Saving bathymetry to file {out_file}")
    z.astype(">f4").tofile(out_file)


class Features:
    def __init__(self, name) -> None:
        self.name = name
        self.grid_nml = Path("data")
        self.out_file = _get_bathyfile_name()
        logger.info(f"Reading bathymetry from {self.grid_nml}:{self.out_file}")
        self.z, _, _ = _get_bathy_from_nml(self.grid_nml)
        self.min_depth = np.amax(self.z[self.z < 0])
        logger.info(f"Minimum ocean depth: {self.min_depth}")
        self.min_height = 100.0

        if name == "islands":
            self.mask = np.where(self.z >= 0, 1, 0)
            self.cval = self.min_depth
        elif name == "ponds":
            self.mask = np.where(self.z < 0, 1, 0)
            self.cval = self.min_height
        elif name == "creeks":
            mask = np.where(self.z < 0, 1, 0)
            self._creek_mask(mask)
            self.cval = self.min_height
        else:
            raise NotImplementedError(
                "Allowed feature names are `islands`, `ponds` and `creeks`"
            )

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

    def edit_features(self, n_points: int = 1, min_depth=None):
        """Edit features"""

        def delete_islands(event):
            logger.info(f"Deleting {self.name}")
            for i in labels:
                self.z[array == i] = self.cval
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
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(name="islands").edit_features(n_points=n_points)


@app.command("del_ponds")
def del_ponds(
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(name="ponds").edit_features(n_points=n_points)


@app.command("del_creeks")
def del_creeks(
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(name="creeks").edit_features(n_points=n_points)


class EditBathy:
    def __init__(self) -> None:
        self.grid_nml = Path("data")
        self.out_file = _get_bathyfile_name()
        logger.info(f"Reading bathymetry from {self.grid_nml}:{self.out_file}")
        self.z, _, _ = _get_bathy_from_nml(self.grid_nml)
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
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
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
        mouseevent = event.mouseevent
        if mouseevent.xdata is None or mouseevent.ydata is None:
            return

        logger.debug(
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

        if mouseevent.button is MouseButton.LEFT:
            # click left mouse button to make a land point
            self.z[_gy, _gx] = self.lnd_val
        elif mouseevent.button is MouseButton.RIGHT:
            # click right mouse button to make a ocean point
            self.z[_gy, _gx] = self.min_depth

        self.mesh.set_array(self.z)
        self.fig.canvas.draw()


@app.command("edit_bathy")
def edit_bathy():
    """Opens up a GUI to click and edit Bathymetry"""
    EditBathy()


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
