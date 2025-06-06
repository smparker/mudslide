# -*- coding: utf-8 -*-
"""Harmonic model"""

import json
import numpy as np
import yaml

from mudslide.models import ElectronicModel_

from typing import Any, List
from mudslide.typing import ArrayLike, DtypeLike


class HarmonicModel(ElectronicModel_):
    r"""Adiabatic model for ground state dynamics

    """
    _ndof: int = 1
    nstates_: int = 1
    reference: Any = None

    def __init__(self, x0: ArrayLike, E0: float, H0: ArrayLike, mass: ArrayLike,
                 atom_types: List[str] = None, ndims: int = 1, nparticles: int = 1):
        """Constructor

        Args:
            x0: Equilibrium position (ndims * nparticles)
            E0: Ground state energy
            H0: Hessian at equilibrium (ndims * nparticles x ndims * nparticles)
            mass: Mass of the coordinates (ndims * nparticles)
            natom_types: Atom types (list of strings)
            ndims: Number of dimensions (e.g. 3 for 3D)
            nparticles: Number of particles (e.g. 1 for a single particle)
        """
        super().__init__(ndims=ndims, nparticles=nparticles,
                         atom_types=atom_types, representation="adiabatic")
        self.x0 = np.array(x0)

        self.E0 = E0
        self.H0 = np.array(H0)

        self.mass = np.array(mass, dtype=np.float64).reshape(self._ndof)

        if self.H0.shape != (self._ndof, self._ndof):
            raise ValueError("Incorrect shape of Hessian")

        if self.mass.shape != (self._ndof,):
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
        self._force = -grad.reshape([1, self._ndof])
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
        atom_types = model_dict.get("atom_types", None)
        nparticles = len(atom_types) if atom_types is not None else 1
        ndims = len(x0) // nparticles

        return cls(x0, E0, H0, mass, atom_types=atom_types,
                   ndims=ndims, nparticles=nparticles)

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
        if self.atom_types is not None:
            out["atom_types"] = self.atom_types

        with open(filename, "w") as f:
            if filename.endswith(".json"):
                json.dump(out, f)
            elif filename.endswith(".yaml"):
                yaml.dump(out, f)
            else:
                raise ValueError("Unknown file format")
