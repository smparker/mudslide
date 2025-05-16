# -*- coding: utf-8 -*-
"""Util functions"""

import os
import sys

from .constants import bohr_to_angstrom

def write_xyz(coords, atom_types, file, comment=""):
    """Write coordinates to open file handle in XYZ format"""
    file.write(f"{len(coords)}\n")
    file.write(f"{comment}\n")
    # convert coords to Angstrom
    acoords = coords * bohr_to_angstrom
    for atom, coord in zip(atom_types, acoords):
        atom = atom.capitalize()
        file.write(f"{atom} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")

def write_trajectory_xyz(model, trace, filename):
    """Write trajectory to XYZ file"""
    with open(filename, "w") as file:
        for frame in trace:
            desc = f"E={frame['energy']:g}; t={frame['time']:g}"
            coords = frame["position"].reshape(-1, 3)
            atom_types = model.atom_types if model.atom_types is not None else ["X"] * len(coords//3)
            write_xyz(coords, atom_types, file, comment=desc)
