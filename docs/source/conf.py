import os
import sys

import toml

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Path Setup and Package Details
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

# IMPORTS ps I can be done not so dumbly
sys.path.insert(0, os.path.dirname(os.path.dirname(os. getcwd())))

# get package details directly from pyproject
pyproject_file = os.path.join(os.path.dirname(os.path.dirname(os. getcwd())), "pyproject.toml")
package_details = toml.load(pyproject_file).get("project")

project = package_details.get("name")
# Change me for multiple others
if isinstance(authors := package_details.get("authors"), str):
    author = authors
release = package_details.get("version")


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sphinx Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""
master_docs = 'index'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    "autoclasstoc",
    'sphinx_autodoc_typehints']

autodoc_default_options = {
    'members': True,
    'special-members': False,
    'private-members': False,
    'inherited-members': True,
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
templates_path = ['_templates']
exclude_patterns = []
language = 'en'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

source_suffix = ".rst"

html_theme = 'sphinx_rtd_theme'

pygments_style = "sphinx"

todo_include_todos = True


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sphinx Autodoc Typehints
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

typehints_defaults = 'comma'

always_document_param_types = True

typehints_fully_qualified = False

typehints_document_rtype = True

always_use_bars_union = True

simplify_optional_unions = False

typehints_use_signature = False

typehints_use_signature_return = False


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AutoClassToc Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


autoclasstoc_sections = [
    #'read-only-properties',
    #'read-write-properties',
    'public-attrs',
    'public-methods-without-dunders',
    'private-methods',
    'private-attrs',
    ]
