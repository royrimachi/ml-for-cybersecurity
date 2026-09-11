# flake8: noqa: F821
# pyright: ignore[reportUndefinedVariable]
from jupytext.ext import load_jupyter_server_extension

load_jupyter_server_extension(c)

# .jupyter/jupyter_server_config.py
c.ServerApp.contents_manager_class = "jupytext.TextFileContentsManager"

# Optional: Automatically pair EVERY new notebook with a .py file (Percent format)
c.JupytextConfiguration.default_jupytext_formats = "ipynb,py:percent"
