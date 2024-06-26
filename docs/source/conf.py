# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pydata_sphinx_theme


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    # Mocking of the dependencies
    sys.path.insert(0,'.')
    sys.path.pop(0)

sys.path.insert(0, os.path.abspath('./../../'))

from unittest import mock

MOCK_MODULES = ['numpy', 'tables','scipy', 'scipy.stats', 'scipy.optimize',
                'matplotlib', 'matplotlib.pyplot', 'matplotlib.colors',
                'matplotlib.patches', 'matplotlib.collections', 'matplotlib.offsetbox', 
                'matplotlib.gridspec', 'matplotlib.cm', 'seaborn',
                'pandas', 'lmfit', 'phconvert', 'fretbursts', 'H2MM_C']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
    
import burstH2MM
# -- Project information -----------------------------------------------------

project = 'burstH2MM'
copyright = '2022, Paul David Harris'
author = 'Paul David Harris'

# The full version, including alpha/beta/rc tags
from importlib.metadata import version as get_version
release:str = get_version('burstH2MM')
version:str = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.inheritance_diagram',
        'sphinx.ext.autosummary',
        'sphinx.ext.mathjax',
        'sphinx.ext.intersphinx',
        'sphinx.ext.napoleon',
        'sphinx_copybutton',
        'IPython.sphinxext.ipython_console_highlighting',
        'IPython.sphinxext.ipython_directive',
        'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['notebooks/*','**.ipynb_checkpoints' ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_favicon = 'images/logo.ico'
# html_logo = 'images/logo_light.png'
html_theme_options = {'logo':{'image_light':'images/logo_light.png', 'image_dark':'images/logo_dark.png', 'text':f'burstH2MM {version}'}}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']
