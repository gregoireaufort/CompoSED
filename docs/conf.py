"""Sphinx configuration for the CompoSED documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

project = "CompoSED"
author = "Gregoire Aufort"
copyright = "2026, Gregoire Aufort"

try:
    release = version("composed")
except PackageNotFoundError:
    release = "development"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

if os.environ.get("COMPOSED_DOCS_INTERSPHINX") == "1":
    extensions.append("sphinx.ext.intersphinx")
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "astropy": ("https://docs.astropy.org/en/stable", None),
    }

html_theme = "furo"
html_title = f"CompoSED {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/gregoireaufort/CompoSED/",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
