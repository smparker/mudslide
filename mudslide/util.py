# -*- coding: utf-8 -*-
"""Utility functions for the mudslide package."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from .exceptions import ConfigurationError



def find_unique_name(name: str,
                     location: str = "",
                     always_enumerate: bool = False,
                     ending: str = "") -> str:
    """Generate a unique filename by adding a suffix if the file already exists.

    Parameters
    ----------
    name : str
        Initial basename for the file.
    location : str, optional
        Directory path where the file will be created, by default "".
    always_enumerate : bool, optional
        Whether to always add a suffix even if the file doesn't exist, by default False.
    ending : str, optional
        File extension to append to the name, by default "".

    Returns
    -------
    str
        Unique basename that doesn't conflict with existing files.

    Raises
    ------
    FileExistsError
        If no unique name could be generated from the base name.
    """
    name_yaml = f"{name}{ending}"
    name_yaml = f"{name}{ending}"
    if not always_enumerate and not os.path.exists(
            os.path.join(location, name_yaml)):
        return name
    for i in range(sys.maxsize):
        out = f"{name}-{i:d}"
        out_yaml = f"{out}{ending}"
        out = f"{name}-{i:d}"
        out_yaml = f"{out}{ending}"
        if not os.path.exists(os.path.join(location, out_yaml)):
            return out
    raise FileExistsError(f"No unique name could be made from base {name}.")



def is_string(x: Any) -> bool:
    """Check if the input is a string type.

    Parameters
    ----------
    x : any
        Input variable to check.

    Returns
    -------
    bool
        True if x is a string, False otherwise.
    """
    return isinstance(x, str)


def remove_center_of_mass_motion(velocities: np.ndarray,
                                 masses: np.ndarray) -> np.ndarray:
    """Remove the center of mass motion from a set of velocities.

    Parameters
    ----------
    velocities : np.ndarray
        Array of velocities with shape (n_atoms, 3).
    masses : np.ndarray
        Array of masses with shape (n_atoms,).

    Returns
    -------
    np.ndarray
        Velocities with center of mass motion removed, shape (n_atoms, 3).
    """
    com = np.sum(velocities * masses[:, np.newaxis], axis=0) / np.sum(masses)
    return velocities - com


def remove_angular_momentum(velocities: np.ndarray, masses: np.ndarray,
                            coordinates: np.ndarray) -> np.ndarray:
    """Remove the angular momentum from a set of coordinates and velocities.

    Parameters
    ----------
    velocities : np.ndarray
        Array of velocities with shape (n_atoms, 3).
    masses : np.ndarray
        Array of masses with shape (n_atoms,).
    coordinates : np.ndarray
        Array of coordinates with shape (n_atoms, 3).

    Returns
    -------
    np.ndarray
        Velocities with angular momentum removed, shape (n_atoms, 3).
    """
    natom = velocities.shape[0]
    com = np.sum(velocities * masses[:, np.newaxis], axis=0) / np.sum(masses)
    com_coord = coordinates - com

    momentum = np.einsum('ai,a->ai', velocities, masses, dtype=np.float64)

    # angular momentum
    angular_momentum = np.cross(com_coord, momentum).sum(axis=0,
                                                         dtype=np.float64)

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


def check_options(options: dict, recognized: list, strict: bool = True) -> None:
    """Check whether the options dictionary contains only recognized options.

    Parameters
    ----------
    options : dict
        Dictionary of options to check.
    recognized : list
        List of recognized option names.
    strict : bool, optional
        Whether to raise an error if an unrecognized option is found, by default True.

    Raises
    ------
    ValueError
        If strict is True and unrecognized options are found.
    """
    problems = [x for x in options if x not in recognized]

    if problems:
        if strict:  # pylint: disable=no-else-raise
            raise ConfigurationError(f"Unrecognized options found: {problems}.")
        else:
            print(f"WARNING: Ignoring unrecognized options: {problems}.")
