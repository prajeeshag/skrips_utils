import f90nml
import numpy as np


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
