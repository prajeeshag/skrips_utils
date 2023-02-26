#!/usr/bin/env python3

import typer

import utils

app = typer.Typer(add_completion=False)

app.add_typer(utils.app, name="bathymetry")

# Exposing the click object for sphinx documentation
app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
