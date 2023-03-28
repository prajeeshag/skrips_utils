import typer
from pathlib import Path
import xarray as xr
import f90nml


app = typer.Typer()


@app.command()
def create_from_wrf(
    wrf_geo: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="WRF `wrf_geo` file.",
    ),
    out_file: Path = typer.Option(
        "mitgcm_grid.nml",
        writable=True,
        help="Output grid info nml file.",
    ),
):
    """
    Creates the grid parameters for MITgcm from WRf geo_em file.
    """

    nml = get_parm04_from_geo(ds)
    nml.write(out_file, force=True)


