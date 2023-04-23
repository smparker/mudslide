# -*- coding: utf-8 -*-
"""Harmonic model"""

import numpy as np

import mudslide

from typing import Any
from .typing import ArrayLike, DtypeLike

class HarmonicModel(mudslide.models.ElectronicModel_):
    r"""Adiabatic model for ground state dynamics

    """
    ndim_: int = 1
    nstates_: int = 1

    def __init__(self,
                 x0: ArrayLike,
                 E0: float,
                 H0: ArrayLike,
                 mass: ArrayLike):
        """Constructor

        Args:
            x0: Equilibrium position
            E0: Ground state energy
            H0: Hessian at equilibrium
            mass: Mass of the coordinates
        """
        self.x0 = np.array(x0)
        self.ndim_ = len(self.x0)

        self.E0 = E0
        self.H0 = np.array(H0)

        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim_)

        if self.H0.shape != (self.ndim_, self.ndim_):
            raise ValueError("Incorrect shape of Hessian")

        if self.mass.shape != (self.ndim_,):
            raise ValueError("Incorrect shape of mass")

    def compute(self, X: ArrayLike, gradients: Any = None, reference: Any = None) -> None:
        """Compute and store the energies and gradients

        Args:
            X: coordinates
            gradients: Gradients
            reference: Reference

        Returns:
            None
        """
        dx = X - self.x0
        grad = self.H0 @ dx
        energy = self.E0 + 0.5 * np.dot(dx.T, grad)

        self.energies = np.array([energy])
        self.force = -grad

