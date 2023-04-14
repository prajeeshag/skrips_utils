import f90nml
import numpy as np
import xarray as xr
import datetime
import yaml
from yaml.loader import SafeLoader
import math

from pathlib import Path
from sphericalpolygon.excess_area import polygon_area
from sphericalpolygon import Sphericalpolygon
from typing import Tuple


def _get_bathy_from_nml(data, bathy_file=None):
    """
    Get z, lat, lon from the information provided by `data` namelist
    """
    nml = f90nml.read(data)
    idir = data.parents[0]
    usingsphericalpolargrid = nml["parm04"]["usingsphericalpolargrid"]
    if not usingsphericalpolargrid:
        raise NotImplementedError(
            "The bathymetry plotting is only implemented for spherical-polar grid"
        )
    bathyfile = bathy_file
    if not bathyfile:
        bathyfile = nml["parm05"]["bathyfile"]

    nx, ny, lon, lat = _grid_from_parm04(nml["parm04"])

    z = np.fromfile(f"{idir}/{bathyfile}", ">f4").reshape(ny, nx)
    return z, lat, lon


def _grid_from_parm04(nml):
    xgorigin = nml["xgorigin"]
    ygorigin = nml["ygorigin"]
    delx = nml["delx"]
    dely = nml["dely"]
    nx = len(delx)
    ny = len(dely)

    lon_bnd = np.zeros(nx + 1)
    lat_bnd = np.zeros(ny + 1)

    lon_bnd[0] = xgorigin
    lat_bnd[0] = ygorigin

    for i, dx in enumerate(delx):
        lon_bnd[i + 1] = lon_bnd[i] + dx

    for i, dy in enumerate(dely):
        lat_bnd[i + 1] = lat_bnd[i] + dy

    lon = (lon_bnd[1:] + lon_bnd[0:-1]) * 0.5
    lat = (lat_bnd[1:] + lat_bnd[0:-1]) * 0.5

    # lon, lat = np.meshgrid(lon, lat)
    # lon_bnd, lat_bnd = np.meshgrid(lon_bnd, lat_bnd)

    # grid_out = xr.Dataset(
    #     {
    #         "lat": (["nlon", "nlat"], lat, {"units": "degrees_north"}),
    #         "lon": (["nlon", "nlat"], lon, {"units": "degrees_east"}),
    #         "lat_b": (["nlonb", "nlatb"], lat_bnd, {"units": "degrees_north"}),
    #         "lon_b": (["nlonb", "nlatb"], lon_bnd, {"units": "degrees_east"}),
    #     }
    # )

    return nx, ny, lon, lat


def _vgrid_from_parm04(nml):
    delz = np.array(nml["parm04"]["delz"])
    zi = [0.0]
    for dz in delz:
        zi.append(zi[-1] + dz)

    z = np.array(zi[1:]) - delz * 0.5
    return z, delz


def _get_parm04_from_geo():
    """Get the MITgcm PARM04 grid parameters from WRF geo_em file xarray dataset"""
    ds = xr.open_dataset("geo_em.d01.nc")
    # dx = nml_wps['geogrid']['dx']
    # dy = nml_wps['geogrid']['dy']
    lat_bnd = ds["XLAT_V"][0, :, 0].values
    lon_bnd = ds["XLONG_U"][0, 0, :].values
    nml = f90nml.Namelist()
    delx = list(lon_bnd[1:] - lon_bnd[0:-1])
    dely = list(lat_bnd[1:] - lat_bnd[0:-1])
    xg = lon_bnd[0]
    yg = lat_bnd[0]
    nml["parm04"] = {
        "usingsphericalpolargrid": True,
        "xgorigin": xg,
        "ygorigin": yg,
        "delX": delx,
        "delY": dely,
    }
    return nml


def _get_grid_from_geo():
    return _grid_from_parm04(_get_parm04_from_geo())


def _get_bathyfile_name():
    return f90nml.read("data")["parm05"]["bathyfile"]


def _wps_sdate(sdate):
    return datetime.datetime.strptime(sdate, "%Y-%m-%d_%H:%M:%S")


def _get_start_date_wps():
    sdate = f90nml.read("namelist.wps")["share"]["start_date"]
    return _wps_sdate(sdate)


def _get_end_date_wps():
    sdate = f90nml.read("namelist.wps")["share"]["end_date"]
    return _wps_sdate(sdate)


def _da2bin(da: xr.DataArray, binfile: Path, typ: str = ">f4"):
    """
    write xarray data array to with big-endian byte ordering
    as single-precision real numbers (which is NumPy float32 or
    equivalently, Fortran real*4 format)
    """
    da.values.astype(typ).tofile(binfile)


def _load_yaml(yaml_file):
    # Open the file and load the file
    with open(yaml_file) as f:
        return yaml.load(f, Loader=SafeLoader)


def great_circle(lon1, lat1, lon2, lat2, input_in_radians=False, rearth=6370.0):
    """
    Calculates the great circle distance between two points on the Earth's surface,
    given their longitude and latitude coordinates.

    Arguments:

    lon1 (float): the longitude of the first point
    lat1 (float): the latitude of the first point
    lon2 (float): the longitude of the second point
    lat2 (float): the latitude of the second point
    input_in_radians (bool): a flag indicating whether the
            input coordinates are in radians (True) or degrees (False).
            Default is False.
    rearth (float): the radius of the Earth in kilometers. Default is 6370.0 km.
    Returns:

    The great circle distance between the two points, in kilometers."""
    xlon1, xlat1, xlon2, xlat2 = lon1, lat1, lon2, lat2
    if not input_in_radians:
        xlon1, xlat1, xlon2, xlat2 = map(math.radians, [xlon1, xlat1, xlon2, xlat2])
    dlon = xlon2 - xlon1
    dlat = xlat2 - xlat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(xlat1) * math.cos(xlat2) * math.sin(dlon / 2) ** 2
    )
    return 2.0 * rearth * math.asin(math.sqrt(a))
    return rearth * (
        math.acos(
            math.sin(lat1) * math.sin(lat2)
            + math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)
        )
    )


def quadrilateral_area_on_earth(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
    R: float = 6370.0,
) -> float:
    # return polygon_area([a, b, c, d, a]) * R * R
    arr = [a, b, c, d]
    polygon = Sphericalpolygon.from_array([a, b, c, d])
    return polygon.area(R)


def load_bathy(bathy_file: Path, nx: int, ny: int):
    z = np.fromfile(bathy_file, ">f4")
    if len(z) != nx * ny:
        raise ValueError(
            f"Dimension mismatch for bathymetry field from file {bathy_file}"
        )
    return z.reshape(ny, nx)
