#!/usr/bin/env python3

import typer

from .src import grid, bathymetry

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

app.add_typer(bathymetry.app, name="bathymetry")
app.add_typer(grid.app, name="grid")

# Exposing the click object for sphinx documentation
app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
