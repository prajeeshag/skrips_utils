
from pathlib import Path

import f90nml
import xarray as xr 
import xesmf as xe

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors

import typer
import click

app = typer.Typer()

def plot_bathy(lat, lon, z, filepath='bathymetry.png'):
    ax = plt.axes(projection=ccrs.PlateCarree())
    levels = list(np.linspace(-3000,-200,10))[:-1] + list(np.linspace(-200,0,21))
    levels = [ -0.0000001 if item == 0.0 else item for item in levels ]

    cmap=plt.cm.jet
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

    cs=ax.contourf(lon,lat,z,levels=levels,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),extend='max')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.colorbar(cs)

    plt.savefig(filepath)

@app.command(name='create')
def make_bathy(
    input_bathy:str = typer.Argument(..., help="Path of input bathymetry netcdf file"),
    wrf_geo:str = typer.Argument(..., help="Path of WRF `geo_em` file"),
    out_file:str = typer.Option('Bathymetry', help="Path of output bathymetry file")
):
    """
    Create bathymetry file for MITGCM

    The coordinate information required for creating bathymetry can be given in following ways:
        1) Coordinates taken from the WRF geo_em file. 
           This will also create the relevant MITgcm namelist fields.
        2) Coordinates taken from MITgcm namelist. (Not Implemented)
    """

    ds_geo = xr.open_dataset(wrf_geo)
    ds_input_bathy = xr.open_dataset(input_bathy)
    input_bathy = ds_input_bathy['z']

    XLAT_M = ds_geo['XLAT_M'][0,:,0]
    XLONG_M = ds_geo['XLONG_M'][0,0,:]

    print(XLAT_M.values)
    print(XLONG_M.values)

    grid_out = xr.Dataset(
        {
            "lat": (["lat"], XLAT_M.values, {"units": "degrees_north"}),
            "lon": (["lon"], XLONG_M.values, {"units": "degrees_east"}),
        }
    )
    regridder = xe.Regridder(ds_input_bathy, grid_out, "conservative")

    dr_out = regridder(input_bathy, keep_attrs=True)

    dr_out.to_netcdf(f'{out_file}.nc')

    plot_bathy(XLAT_M, XLONG_M, dr_out, filepath=f'{out_file}.png')

