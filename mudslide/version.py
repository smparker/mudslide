# -*- coding: utf-8 -*-
"""Mudslide version"""

import os

__version__ = '0.12.0'


def get_install_path() -> str:
    """Return the installation path of the mudslide package."""
    return os.path.dirname(os.path.abspath(__file__))


def get_version_info() -> str:
    """Return version and installation path information for CLI help messages."""
    return f"mudslide {__version__}\nInstalled at: {get_install_path()}"
