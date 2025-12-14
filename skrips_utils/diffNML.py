import os
import f90nml
import pandas as pd
from .utils import CaseInsensitiveDict
import typer
import re

app = typer.Typer(add_completion=False)


# Function to format list elements and boolean values
def format_element(element):
    if isinstance(element, list) and len(element) > 1:
        return f"{element[0]}...{element[-1]}"
    elif isinstance(element, bool):
        return str(element)
    elif isinstance(element, str):
        return os.path.basename(element)
    return element


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


@app.command()
def main(
    nml_file1: str = typer.Argument(help="Path to the first namelist file"),
    nml_file2: str = typer.Argument(help="Path to the second namelist file"),
    name1: str = typer.Option(
        None, help="Optional representative name for the first namelist file"
    ),
    name2: str = typer.Option(
        None, help="Optional representative name for the second namelist file"
    ),
):
    """
    Compare two namelist files and generate an Excel file with the differences.
    """
    parser = f90nml.Parser()
    parser.comment_tokens += "#"
    parser.comment_tokens += "$"

    nmlg1 = CaseInsensitiveDict(parser.read(nml_file1))
    nmlg2 = CaseInsensitiveDict(parser.read(nml_file2))

    nmlnms = list(set(list(nmlg1.keys()) + list(nmlg2.keys())))

    df_dict = {}

    for nm in nmlnms:
        try:
            nml1 = CaseInsensitiveDict(nmlg1[nm])
        except KeyError:
            print(f"Namelist {nm} does not exist in {nml_file1}")
            continue
        try:
            nml2 = CaseInsensitiveDict(nmlg2[nm])
        except KeyError:
            print(f"Namelist {nm} does not exist in {nml_file2}")
            continue

        vars = list(set(list(nml1.keys()) + list(nml2.keys())))

        nm_list = []
        var_list = []
        val1_list = []
        val2_list = []

        for var in vars:
            try:
                val1 = nml1[var]
            except KeyError:
                val1 = ""

            try:
                val2 = nml2[var]
            except KeyError:
                val2 = ""

            if val1 != val2:
                nm_list.append(nm)
                var_list.append(var)
                val1_list.append(val1)
                val2_list.append(val2)

        if len(var_list) == 0:
            continue
        df = pd.DataFrame(
            {
                "variable": var_list,
                name1 if name1 else nml_file1: val1_list,
                name2 if name2 else nml_file2: val2_list,
            }
        )
        df_dict[nm] = df.map(format_element)

    sanitized_file1 = sanitize_filename(nml_file1)
    sanitized_file2 = sanitize_filename(nml_file2)
    output_file = f"{sanitized_file1}_vs_{sanitized_file2}.xlsx"

    if not df_dict:
        return
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        for nm, df in df_dict.items():
            df.to_excel(writer, sheet_name=nm, index=False)

    print(f"Difference file created: {output_file}")


app_click = typer.main.get_command(app)

if __name__ == "__main__":
    app()
