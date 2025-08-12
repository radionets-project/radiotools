#!/usr/bin/env python3
import datetime
import tomllib
from pathlib import Path

import radiotools

pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "changes",
    "*.log",
]

source_suffix = ".rst"
master_doc = "index"


project = pyproject["project"]["name"]
author = pyproject["project"]["authors"][0]["name"]
copyright = "{}.  Last updated {}".format(
    author, datetime.datetime.now().strftime("%d %b %Y %H:%M")
)
python_requires = pyproject["project"]["requires-python"]

# make some variables available to each page
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""


version = radiotools.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_file_suffix = ".html"

html_css_files = ["radiotools.css"]


html_theme_options = {
    "github_url": "https://github.com/radionets-project/radiotools",
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo"],
    "navigation_with_keys": False,
    # "use_edit_page_button": True,
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "Radionets Project",
            "url": "https://github.com/radionets-project",
            "type": "url",
            "icon": "https://avatars.githubusercontent.com/u/77392854?s=200&v=4",  # noqa: E501
        },
    ],
    "announcement": """
        <p>radiotools is not stable yet, so expect large and rapid
        changes to structure and functionality as we explore various
        design choices before the 1.0 release.</p>
    """,
}

html_title = f"{project}"
htmlhelp_basename = project + "docs"


# Configuration for intersphinx
intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}


suppress_warnings = [
    "intersphinx.external",
]
