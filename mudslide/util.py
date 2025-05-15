# -*- coding: utf-8 -*-
"""Util functions"""

import os
import sys

import numpy as np

def find_unique_name(name: str, location="", always_enumerate: bool = False, ending: str = "") -> str:
    """
    Given an input basename, checks whether a file with the given name already exists.
    If a file already exists, a suffix is added to make the file unique.

    :param name: initial basename

    :returns: unique basename
    """
    name_yaml = "{}{}".format(name, ending)
    if not always_enumerate and not os.path.exists(os.path.join(location, name_yaml)):
        return name
    for i in range(sys.maxsize):
        out = "{}-{:d}".format(name, i)
        out_yaml = "{}{}".format(out, ending)
        if not os.path.exists(os.path.join(location, out_yaml)):
            return out
    raise Exception("No unique name could be made from base {}.".format(name))
    return ""

def is_string(x) -> bool:
    """
    Tell whether the input is a string or some other variable

    :returns: True if x is a string
    """
    return isinstance(x, str)

def remove_center_of_mass_motion(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Remove the center of mass motion from
    a set of coordinates and masses.

    :param velocities: velocities (n_atoms, 3)
    :param masses: masses (n_atoms)

    :returns: coordinates with center of mass motion removed
    """
    com = np.sum(velocities * masses[:, np.newaxis], axis=0) / np.sum(masses)
    return velocities - com

def remove_angular_momentum(velocities: np.ndarray, masses: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Remove the angular momentum from a set of coordinates, velocities, and masses.

    :param velocities: velocities (n_atoms, 3)
    :param masses: masses (n_atoms)
    :param coordinates: coordinates (n_atoms, 3)

    :returns: velocities with angular momentum removed
    """
    natom = velocities.shape[0]
    com = np.sum(velocities * masses[:, np.newaxis], axis=0) / np.sum(masses)
    com_coord = coordinates - com

    momentum = np.einsum('ai,a->ai', velocities, masses, dtype=np.float64)

    # angular momentum
    angular_momentum = np.cross(com_coord, momentum).sum(axis=0, dtype=np.float64)

    # inertial tensor
    inertia = np.zeros((3, 3), dtype=np.float64)
    for i in range(natom):
        x, y, z = com_coord[i]
        inertia += masses[i] * (np.eye(3) * (x**2 + y**2 + z**2) -
                                np.outer(com_coord[i], com_coord[i]))

    # calculate angular velocity
    angular_velocity = np.linalg.solve(inertia, angular_momentum)

    corrected_velocities = velocities.copy()
    for i in range(natom):
        corrected_velocities[i] -= np.cross(angular_velocity, com_coord[i])

    return corrected_velocities
