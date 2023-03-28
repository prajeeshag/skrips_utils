#!/usr/bin/env python3

import typer

from .src import grid, bathymetry, geo_em

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

app.add_typer(bathymetry.app, name="bathy")
app.add_typer(grid.app, name="grid")
app.add_typer(geo_em.app, name="geo_em")

# Exposing the click object for sphinx documentation
app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
