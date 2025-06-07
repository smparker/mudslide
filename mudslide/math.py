# -*- coding: utf-8 -*-
"""Math helper functions for molecular dynamics simulations."""

from collections import deque
import warnings
import numpy as np
from numpy.typing import ArrayLike

from .util import remove_center_of_mass_motion

from .constants import boltzmann


def poisson_prob_scale(x: ArrayLike):
    """Compute (1 - exp(-x))/x for scaling Poisson probabilities.

    Parameters
    ----------
    x : ArrayLike
        Input array of values.

    Returns
    -------
    ArrayLike
        The computed value (1 - exp(-x))/x. For small values of x (< 1e-3),
        uses a Taylor series expansion for numerical stability.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = np.where(np.absolute(x) < 1e-3,
                1 - x/2 + x**2/6 - x**3/24,
                -np.expm1(-x)/x)
    return out


def boltzmann_velocities(mass, temperature, remove_translation=True,
                         scale=True, seed=None):
    """Generate random velocities according to the Boltzmann distribution.

    Parameters
    ----------
    mass : ArrayLike
        Array of particle masses.
    temperature : float
        Target temperature for the velocity distribution.
    remove_translation : bool, optional
        Whether to remove center of mass translation from velocities.
        Default is True.
    scale : bool, optional
        Whether to scale velocities to match the target temperature.
        Default is True.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    ArrayLike
        Array of velocities following the Boltzmann distribution.
    """
    rng = np.random.default_rng(seed)
    kt = boltzmann * temperature
    sigma = np.sqrt(kt * mass)
    p = rng.normal(0.0, sigma)

    if remove_translation:
        v = p / mass
        v3 = v.reshape((-1, 3))
        M = mass.reshape((-1, 3))[:,0] # pylint: disable=invalid-name

        v = remove_center_of_mass_motion(v3, M).flatten()
        p = v * mass

    #if remove_rotation:
    #    v3 = v.reshape((-1, 3))
    #    M = mass.reshape((-1, 3))[:,0]
    #    v = remove_angular_momentum(v3, M, np.zeros_like(v3)).flatten()

    if scale:
        avg_KE = 0.5 * np.dot(p**2, np.reciprocal(mass)) / mass.size # pylint: disable=invalid-name
        kbT2 = 0.5 * kt # pylint: disable=invalid-name
        scal = np.sqrt(kbT2 / avg_KE)
        p *= scal

    v = p / mass

    return v


class RollingAverage:
    """Compute a rolling average of a series of values.

    This class maintains a fixed-size window of values and computes their
    average. When the window is full, adding a new value automatically
    removes the oldest value.

    Parameters
    ----------
    window_size : int, optional
        The number of values to include in the rolling average.
        Default is 50.

    Attributes
    ----------
    window_size : int
        The maximum number of values in the window.
    values : deque
        The collection of values in the current window.
    sum : float
        The current sum of all values in the window.
    """

    def __init__(self, window_size=50):
        """Initialize the RollingAverage calculator.

        Parameters
        ----------
        window_size : int, optional
            The number of values to include in the rolling average.
            Default is 50.
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0

    def insert(self, value):
        """Add a new value to the rolling average.

        If the window is full, the oldest value is automatically removed.

        Parameters
        ----------
        value : float
            The new value to add to the rolling average.
        """
        # If window is full, subtract the oldest value from sum
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]

        # Add new value
        self.values.append(value)
        self.sum += value

    def get_average(self):
        """Calculate and return the current rolling average.

        Returns
        -------
        float
            The current rolling average, or 0.0 if no values have been added.
        """
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    def __len__(self):
        """Return the current number of values in the window.

        Returns
        -------
        int
            The number of values currently in the window.
        """
        return len(self.values)
