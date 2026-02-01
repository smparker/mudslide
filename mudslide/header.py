# -*- coding: utf-8 -*-
"""Print header information about the mudslide package and environment."""

import sys
import platform
from datetime import datetime

import numpy
import scipy

from .version import __version__, get_install_path

BANNER = r"""
               |
               |
           .-'~~~`-.
         .'         `.
         |  MUDSLIDE  |
         \           /
          `.       .'
     ^^    `'~~~'`   ^^
   ^^  ^^    |||    ^^  ^^
  ^^^  ^^^   |||   ^^  ^^^
  ^^^^^^^^  // \\  ^^^^^^^^
  ^^^^^^^^ //   \\ ^^^^^^^^
"""


def print_header() -> None:
    """Print mudslide version, Python version, and key dependency info."""
    info_lines = [
        f"  Version    : {__version__}",
        f"  Python     : {sys.version.split()[0]}",
        f"  NumPy      : {numpy.__version__}",
        f"  SciPy      : {scipy.__version__}",
        f"  Platform   : {platform.platform()}",
        f"  Path       : {get_install_path()}",
        f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    width = max(len(line) for line in info_lines) + 2
    border = "=" * width

    print(BANNER)
    print(f"  Nonadiabatic Molecular Dynamics")
    print(f"  with Trajectory Surface Hopping")
    print()
    print(border)
    for line in info_lines:
        print(line)
    print(border)
    print()
