# -*- coding: utf-8 -*-
"""Math helper functions"""

from __future__ import division

import numpy as np
from .typing import ArrayLike
from .constants import boltzmann
import warnings


def poisson_prob_scale(x: ArrayLike):
    """Computes (1 - exp(-x))/x which is needed when scaling Poisson probabilities"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = np.where(np.absolute(x) < 1e-3,
                1 - x/2 + x**2/6 - x**3/24,
                -np.expm1(-x)/x)
    return out


def boltzmann_velocities(mass, temperature, scale=True, seed=None):
    rng = np.random.default_rng(seed)
    kt = boltzmann * temperature
    sigma = np.sqrt(kt * mass)
    p = rng.normal(0.0, sigma)
    if scale:
        avg_KE = 0.5 * np.dot(p**2, np.reciprocal(mass)) / mass.size
        kbT2 = 0.5 * kt
        scal = np.sqrt(kbT2 / avg_KE)
        p *= scal
    p = p / mass
    return p
