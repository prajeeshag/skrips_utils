# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aps2s'
copyright = '2023, Prajeesh Ag'
author = 'Prajeesh Ag'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


sys.path.append(os.path.abspath('../'))

extensions = [
    'sphinx_copybutton',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_click'
]

# autodoc_mock_imports = ["f90nml", "xarray", "xesmf", "numpy", "matplotlib", "cartopy"]
click_mock_imports = ["f90nml", "xarray", "xesmf", "numpy", "matplotlib", "cartopy"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True



# autodoc_mock_imports = [
#     "xesmf",
#     "matplotlib",
#     "xarray",
#     "something",
#     ]
