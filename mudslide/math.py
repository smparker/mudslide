# -*- coding: utf-8 -*-
"""Math helper functions"""

import warnings
import numpy as np

from .util import remove_center_of_mass_motion, remove_angular_momentum

from .typing import ArrayLike
from .constants import boltzmann


def poisson_prob_scale(x: ArrayLike):
    """Computes (1 - exp(-x))/x which is needed when scaling Poisson probabilities"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = np.where(np.absolute(x) < 1e-3,
                1 - x/2 + x**2/6 - x**3/24,
                -np.expm1(-x)/x)
    return out


def boltzmann_velocities(mass, temperature, remove_translation=True,
                         scale=True, seed=None):
    """Generates random velocities according to the Boltzmann distribution.

    :param mass: array of masses
    :param temperature: target temperature
    :param remove_translation: remove center of mass translation from velocities
    :param remove_rotation: remove center of mass rotation from velocities
    :param scale: scale velocities to match the target temperature
    :param seed: random seed
    :return: array of velocities
    """
    rng = np.random.default_rng(seed)
    kt = boltzmann * temperature
    sigma = np.sqrt(kt * mass)
    p = rng.normal(0.0, sigma)

    if scale:
        avg_KE = 0.5 * np.dot(p**2, np.reciprocal(mass)) / mass.size
        kbT2 = 0.5 * kt
        scal = np.sqrt(kbT2 / avg_KE)
        p *= scal

    v = p / mass

    if remove_translation:
        v3 = v.reshape((-1, 3))
        M = mass.reshape((-1, 3))[:,0]

        v = remove_center_of_mass_motion(v3, M).flatten()

    #if remove_rotation:
    #    v3 = v.reshape((-1, 3))
    #    M = mass.reshape((-1, 3))[:,0]
    #    v = remove_angular_momentum(v3, M, np.zeros_like(v3)).flatten()

    return v
