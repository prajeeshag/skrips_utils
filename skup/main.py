#!/usr/bin/env python3

import typer

from .src import grid, bathymetry, wrf, mitgcm

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

app.add_typer(mitgcm.app, name="mitgcm")
# app.add_typer(bathymetry.app, name="bathy")
app.add_typer(wrf.app, name="wrf")

# Exposing the click object for sphinx documentation
app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
