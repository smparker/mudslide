# -*- coding: utf-8 -*-
"""Harmonic model"""

import json
import numpy as np
import yaml

from mudslide.models import ElectronicModel_

from typing import Any
from mudslide.typing import ArrayLike


class HarmonicModel(ElectronicModel_):
    r"""Adiabatic model for ground state dynamics

    """
    ndim_: int = 1
    nstates_: int = 1
    reference: Any = None

    def __init__(self, x0: ArrayLike, E0: float, H0: ArrayLike, mass: ArrayLike):
        """Constructor

        Args:
            x0: Equilibrium position
            E0: Ground state energy
            H0: Hessian at equilibrium
            mass: Mass of the coordinates
        """
        super().__init__(representation="adiabatic")
        self.x0 = np.array(x0)
        self.ndim_ = len(self.x0)

        self.E0 = E0
        self.H0 = np.array(H0)

        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim_)

        if self.H0.shape != (self.ndim_, self.ndim_):
            raise ValueError("Incorrect shape of Hessian")

        if self.mass.shape != (self.ndim_,):
            raise ValueError("Incorrect shape of mass")

    def compute(self, X: ArrayLike, gradients: Any = None, couplings: Any = None, reference: Any = None) -> None:
        """Compute and store the energies and gradients

        Args:
            X: coordinates
            gradients: Gradients
            reference: Reference

        Returns:
            None
        """
        self._position = np.array(X)
        dx = self._position - self.x0
        grad = self.H0 @ dx
        energy = self.E0 + 0.5 * np.dot(dx.T, grad)

        self.energies = np.array([energy])
        self._hamiltonian = np.array([energy])
        self._force = -grad.reshape([1, self.ndim_])
        self._forces_available = [True]

    @classmethod
    def from_dict(cls, model_dict: dict) -> "HarmonicModel":
        """Create a harmonic model from a dictionary

        Args:
            model_dict: Dictionary with model data
        """

        x0 = np.array(model_dict["x0"])
        E0 = float(model_dict["E0"])
        H0 = np.array(model_dict["H0"])
        mass = np.array(model_dict["mass"])

        return cls(x0, E0, H0, mass)

    @classmethod
    def from_file(cls, filename: str) -> "HarmonicModel":
        """Create a harmonic model from a file

        Args:
            filename: Name of the file
        """
        data = {}
        with open(filename, "r") as f:
            # check filename for json or yaml
            if filename.endswith(".json"):
                data = json.load(f)
            elif filename.endswith(".yaml"):
                data = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError("Unknown file format")

        return cls.from_dict(data)

    def to_file(self, filename: str) -> None:
        """Write model to a yaml or json

        Use the ending on the filename to determine the format.
        """
        out = {"x0": self.x0.tolist(), "E0": self.E0, "H0": self.H0.tolist(), "mass": self.mass.tolist()}

        with open(filename, "w") as f:
            if filename.endswith(".json"):
                json.dump(out, f)
            elif filename.endswith(".yaml"):
                yaml.dump(out, f)
            else:
                raise ValueError("Unknown file format")
