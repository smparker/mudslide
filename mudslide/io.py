# -*- coding: utf-8 -*-
"""Util functions"""

from .constants import bohr_to_angstrom


def write_xyz(coords, atom_types, file, comment=""):
    """Write coordinates to open file handle in XYZ format"""
    file.write(f"{len(coords)}\n")
    file.write(f"{comment}\n")
    # convert coords to Angstrom
    acoords = coords * bohr_to_angstrom
    for atom, coord in zip(atom_types, acoords):
        atom = atom.capitalize()
        file.write(
            f"{atom:3s} {coord[0]:20.12f} {coord[1]:20.12f} {coord[2]:20.12f}\n"
        )


def write_trajectory_xyz(model, trace, filename, every=1):
    """Write trajectory to XYZ file"""
    natom, nd = model.dimensionality
    with open(filename, "w", encoding='utf-8') as file:
        for i, frame in enumerate(trace):
            if i % every != 0:
                continue
            desc = f"E={frame['energy']:g}; t={frame['time']:g}"
            coords = frame["position"].reshape(natom, nd)
            atom_types = model.atom_types if model.atom_types is not None else [
                "X"
            ] * natom
            write_xyz(coords, atom_types, file, comment=desc)
