# -*- coding: utf-8 -*-
"""Handle storage and computation of electronic degrees of freedom"""

from __future__ import annotations

import copy as cp
from typing import Tuple, Any, List

import numpy as np
from numpy.typing import ArrayLike


class ElectronicModel_:
    """Base class for handling electronic structure part of dynamics.

    The electronic model is a base class for handling the electronic
    structure part of the dynamics.

    Attributes
    ----------
    _representation : str
        The representation of the electronic model. Can be "adiabatic" or "diabatic".
    _reference : Any
        The reference electronic state. Can be None.
    _nstates : int
        The number of electronic states.
    _ndof : int
        The number of classical degrees of freedom.
    _ndims : int
        The number of dimensions for each particle in the system.
    _nparticles : int
        The number of particles in the system.
    atom_types : List[str]
        The types of atoms in the system.
    _position : np.ndarray
        The position of the system.
    _hamiltonian : np.ndarray
        The electronic Hamiltonian.
    _force : np.ndarray
        The force on the system.
    _forces_available : np.ndarray
        A boolean array indicating which forces are available.
    _derivative_coupling : np.ndarray
        The derivative coupling matrix.
    """

    def __init__(self,
                 representation: str = "adiabatic",
                 reference: Any = None,
                 nstates: int = 0,
                 ndims: int = 1,
                 nparticles: int = 1,
                 ndof: int | None = None,
                 atom_types: List[str] | None = None):
        """Initialize the electronic model.

        Parameters
        ----------
        representation : str, optional
            The representation to use ("adiabatic" or "diabatic"), by default "adiabatic"
        reference : Any, optional
            The reference electronic state, by default None
        nstates : int, optional
            Number of electronic states, by default 0
        ndims : int, optional
            Number of dimensions per particle, by default 1
        nparticles : int, optional
            Number of particles, by default 1
        ndof : int, optional
            Total number of classical degrees of freedom, by default None
        atom_types : List[str], optional
            List of atom types, by default None
        """
        self._ndims = ndims
        self._nparticles = nparticles
        if ndof is None:
            self._ndof = ndims * nparticles
        else:
            self._ndof = ndof
            self._ndims = ndof
            self._nparticles = 1
        self._nstates = nstates

        self._representation = representation
        self._position: np.ndarray
        self._reference = reference

        self._hamiltonian: np.ndarray
        self.energies: np.ndarray
        self._force: np.ndarray
        self._forces_available: np.ndarray = np.zeros(self.nstates, dtype=bool)
        self._derivative_coupling: np.ndarray
        self._derivative_couplings_available: np.ndarray = np.zeros(
            (self.nstates, self.nstates), dtype=bool)
        self._force_matrix: np.ndarray

        self.mass: np.ndarray
        self.atom_types: List[str] | None = atom_types

    @property
    def ndof(self) -> int:
        """Get the number of classical degrees of freedom.

        Returns
        -------
        int
            Number of classical degrees of freedom
        """
        return self._ndof

    @property
    def nparticles(self) -> int:
        """Get the number of particles.

        Returns
        -------
        int
            Number of particles
        """
        return self._nparticles

    @property
    def ndims(self) -> int:
        """Get the number of dimensions per particle.

        Returns
        -------
        int
            Number of dimensions per particle
        """
        return self._ndims

    @property
    def dimensionality(self) -> Tuple[int, int]:
        """Get the number of particles and dimensions.

        Returns
        -------
        Tuple[int, int]
            Tuple of (number of particles, number of dimensions)
        """
        return self._nparticles, self._ndims

    @property
    def nstates(self) -> int:
        """Get the number of electronic states.

        Returns
        -------
        int
            Number of electronic states
        """
        return self._nstates

    @property
    def hamiltonian(self) -> np.ndarray:
        """Get the electronic Hamiltonian.

        Returns
        -------
        np.ndarray
            The electronic Hamiltonian matrix
        """
        return self._hamiltonian

    def force(self, state: int = 0) -> np.ndarray:
        """Return the force on a given state"""
        if not self._forces_available[state]:
            raise ValueError(f"Force on state {state} not available")
        return self._force[state, :]

    def derivative_coupling(self, state1: int, state2: int) -> np.ndarray:
        """Return the derivative coupling between two states"""
        if not self._derivative_couplings_available[state1, state2]:
            raise ValueError(
                f"Derivative coupling between states {state1} and {state2} not available"
            )
        return self._derivative_coupling[state1, state2, :]

    @property
    def derivative_coupling_tensor(self) -> np.ndarray:
        """Return the derivative coupling tensor"""
        if not np.all(self._derivative_couplings_available):
            false_indices = np.argwhere(~self._derivative_couplings_available)
            print(
                f"Derivative couplings not available for state pairs: {false_indices.tolist()}"
            )
            print(
                f"Full availability matrix:\n{self._derivative_couplings_available}"
            )
            raise ValueError("All derivative couplings not available")
        return self._derivative_coupling

    def NAC_matrix(self, velocity: np.ndarray) -> np.ndarray:
        """Return the non-adiabatic coupling matrix
        for a given velocity vector
        """
        if not np.all(self._derivative_couplings_available):
            raise ValueError("NAC_matrix needs all derivative couplings")
        return np.einsum("ijk,k->ij", self._derivative_coupling, velocity)

    @property
    def force_matrix(self) -> np.ndarray:
        """Return the force matrix"""
        if not np.all(self._forces_available):
            raise ValueError("Force matrix needs all forces")
        return self._force_matrix

    def _needed_gradients(self, gradients: Any) -> List[int]:
        """Filter requested gradients to only those not yet available.

        Parameters
        ----------
        gradients : list of int or None
            Requested gradient state indices. None means all states.

        Returns
        -------
        List[int]
            State indices whose forces still need to be computed.
        """
        if gradients is None:
            candidates = range(self.nstates)
        else:
            candidates = gradients
        return [s for s in candidates if not self._forces_available[s]]

    def _needed_couplings(self, couplings: Any) -> List[Tuple[int, int]]:
        """Filter requested couplings to only those not yet available.

        Parameters
        ----------
        couplings : list of tuple(int, int) or None
            Requested coupling pairs. None means all pairs.

        Returns
        -------
        List[Tuple[int, int]]
            Coupling pairs that still need to be computed.
        """
        if couplings is None:
            candidates = [
                (i, j) for i in range(self.nstates) for j in range(self.nstates)
            ]
        else:
            candidates = couplings
        return [(i, j)
                for (i, j) in candidates
                if not self._derivative_couplings_available[i, j]]

    def compute_additional(self,
                           couplings: Any = None,
                           gradients: Any = None) -> None:
        """Compute additional gradients/couplings at the current geometry.

        Checks what's already available and only computes what's missing.
        If everything requested is already available, returns immediately (no-op).
        Subclasses that support selective computation should override this.

        Parameters
        ----------
        couplings : list of tuple(int, int) or None, optional
            Coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            State indices whose forces are needed. None means all.
        """
        needed_g = self._needed_gradients(gradients)
        needed_c = self._needed_couplings(couplings)
        if not needed_g and not needed_c:
            return
        raise NotImplementedError(
            f"compute_additional not implemented; missing gradients={needed_g}, couplings={needed_c}"
        )

    def compute(self,
                X: np.ndarray,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None) -> None:
        """
        Central function for model objects. After the compute function exits, the following
        data must be provided:
          - self._hamiltonian -> nstates x nstates array containing electronic hamiltonian
          - self._force -> nstates x ndof array containing the force on each state
          - self._derivative_coupling -> nstates x nstates x ndof array containing derivative couplings
          - self._forces_available -> boolean array of length nstates
          - self._derivative_couplings_available -> nstates x nstates boolean array

        Parameters
        ----------
        X : ArrayLike
            Position at which to compute properties
        couplings : list of tuple(int, int) or None, optional
            Which coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            Which state forces to compute. None means all.
        reference : Any, optional
            Reference electronic state for phase fixing.

        Nothing is returned, but the model object should contain all the
        necessary data.
        """
        raise NotImplementedError("ElectronicModel_ need a compute function")

    def update(self,
               X: np.ndarray,
               electronics: Any = None,
               couplings: Any = None,
               gradients: Any = None) -> ElectronicModel_:
        """
        Convenience function that copies the present object, updates the position,
        calls compute, and then returns the new object
        """
        out = cp.copy(self)

        if electronics and electronics._reference is not None:
            reference = electronics._reference
        else:
            reference = self._reference

        out.compute(X,
                    couplings=couplings,
                    gradients=gradients,
                    reference=reference)
        return out

    def clone(self) -> ElectronicModel_:
        """Create a copy of the electronics object that
        owns its own resources, including disk

        Major difference from deepcopy is that clone
        should prepare any resources that are needed
        for the object to be used, for example, disk
        access, network access, and so on. deepcopy
        should only copy the object in memory, not
        acquire resources like disk space.

        """
        return cp.deepcopy(self)

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the model"""
        out = {
            "nstates": self.nstates,
            "ndof": self.ndof,
            "position": self._position.tolist(),
            "hamiltonian": self._hamiltonian.tolist(),
            "force": self._force.tolist(),
            "forces_available": self._forces_available.tolist()
        }

        for key in ["_derivative_coupling", "_force_matrix"]:
            if hasattr(self, key):
                out[key.lstrip('_')] = getattr(self, key).tolist()

        return out


class DiabaticModel_(ElectronicModel_):
    """Base class to handle model problems given in simple diabatic forms.

    To derive from DiabaticModel_, the following functions must be implemented:

    - def V(self, X: np.ndarray) -> ArrayLike
      V(x) should return an ndarray of shape (nstates, nstates)
    - def dV(self, X: np.ndarray) -> ArrayLike
      dV(x) should return an ndarray of shape (nstates, nstates, ndof)
    """

    #: Minimum energy gap threshold used when computing derivative couplings.
    #: When the energy difference between two states is smaller than this value,
    #: it is clamped to avoid division by near-zero. Override in subclasses if needed.
    coupling_energy_threshold: float = 1.0e-14

    def __init__(self,
                 representation: str = "adiabatic",
                 reference: Any = None,
                 nstates: int = 0,
                 ndof: int = 0):
        """Initialize the diabatic model.

        Parameters
        ----------
        representation : str, optional
            The representation to use, by default "adiabatic"
        reference : Any, optional
            The reference electronic state, by default None
        nstates : int, optional
            Number of electronic states, by default 0
        ndof : int, optional
            Number of classical degrees of freedom, by default 0
        """
        ElectronicModel_.__init__(self,
                                  representation=representation,
                                  reference=reference,
                                  nstates=nstates,
                                  ndof=ndof)

    def compute(self,
                X: np.ndarray,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None) -> None:
        """Compute electronic properties at position X.

        Parameters
        ----------
        X : ArrayLike
            Position at which to compute properties
        couplings : list of tuple(int, int) or None, optional
            Which coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            Which state forces to compute. None means all.
        reference : Any, optional
            Reference state information, by default None
        """
        self._position = X

        self._reference, self._hamiltonian = self._compute_basis_states(
            self.V(X), reference=reference)
        dV = self.dV(X)

        self._derivative_coupling = self._compute_derivative_coupling(
            self._reference, dV, np.diag(self._hamiltonian))

        # Create new availability arrays to avoid sharing with shallow copies
        self._derivative_couplings_available = np.zeros(
            (self.nstates, self.nstates), dtype=bool)
        if couplings is None:
            self._derivative_couplings_available[:, :] = True
        else:
            for (i, j) in couplings:
                self._derivative_couplings_available[i, j] = True

        self._force = self._compute_force(dV, self._reference)

        # Create new availability array to avoid sharing with shallow copies
        self._forces_available = np.zeros(self.nstates, dtype=bool)
        if gradients is None:
            self._forces_available[:] = True
        else:
            for s in gradients:
                self._forces_available[s] = True

        self._force_matrix = self._compute_force_matrix(dV, self._reference)

    def compute_additional(self,
                           couplings: Any = None,
                           gradients: Any = None) -> None:
        """Compute additional gradients/couplings at the current geometry.

        Since diabatic models compute everything analytically, this just
        marks the newly requested quantities as available.

        Parameters
        ----------
        couplings : list of tuple(int, int) or None, optional
            Coupling pairs to make available. None means all.
        gradients : list of int or None, optional
            State indices whose forces should be made available. None means all.
        """
        needed_g = self._needed_gradients(gradients)
        needed_c = self._needed_couplings(couplings)
        for s in needed_g:
            self._forces_available[s] = True
        for (i, j) in needed_c:
            self._derivative_couplings_available[i, j] = True

    def _compute_basis_states(
            self,
            V: np.ndarray,
            reference: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self._representation == "adiabatic":
            energies, coeff = np.linalg.eigh(V)
            if reference is not None:
                try:
                    for mo in range(self.nstates):
                        if np.dot(coeff[:, mo], reference[:, mo]) < 0.0:
                            coeff[:, mo] *= -1.0
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to regularize new ElectronicStates from a reference object {reference}"
                    ) from exc
            return (coeff, np.diag(energies))
        elif self._representation == "diabatic":
            return (np.eye(self.nstates, dtype=np.float64), V)
        else:
            raise ValueError("Unrecognized run mode")

    def _compute_force(self, dV: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        r""":math:`-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle`"""
        nst = self.nstates
        ndof = self.ndof

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndof], dtype=np.float64)
        for ist in range(self.nstates):
            out[ist, :] += -np.einsum("i,ix->x", coeff[:, ist], half[:, ist, :])
        return out

    def _compute_force_matrix(self, dV: np.ndarray,
                              coeff: np.ndarray) -> np.ndarray:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: np.ndarray, dV: np.ndarray,
                                     energies: np.ndarray) -> np.ndarray:
        r"""Compute derivative couplings :math:`d^\alpha_{ij} = \langle \phi_i | \nabla_\alpha \phi_j \rangle`.

        Uses the Hellmann-Feynman relation to compute derivative couplings from
        the energy gap. When the energy gap between two states is smaller than
        :attr:`coupling_energy_threshold`, the gap is clamped to avoid numerical
        instability.
        """
        if self._representation == "diabatic":
            return np.zeros([self.nstates, self.nstates, self.ndof],
                            dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        thresh = self.coupling_energy_threshold
        for j in range(self.nstates):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < thresh:
                    dE = np.copysign(thresh, dE)

                out[i, j, :] /= dE
                out[j, i, :] /= -dE
            out[j, j, :] = 0.0

        return out

    def V(self, X: np.ndarray) -> np.ndarray:
        """Return the diabatic potential matrix V(X)."""
        raise NotImplementedError(
            "Diabatic models must implement the function V")

    def dV(self, X: np.ndarray) -> np.ndarray:
        """Return the gradient of the diabatic potential matrix dV/dX."""
        raise NotImplementedError(
            "Diabatic models must implement the function dV")


class AdiabaticModel_(ElectronicModel_):
    """Base class to handle model problems that have an auxiliary electronic problem.

    This class handles model problems that have an auxiliary electronic problem admitting
    many electronic states that are truncated to just a few. Sort of a truncated DiabaticModel_.
    """

    #: Minimum energy gap threshold used when computing derivative couplings.
    #: When the energy difference between two states is smaller than this value,
    #: it is clamped to avoid division by near-zero. Override in subclasses if needed.
    coupling_energy_threshold: float = 1.0e-14

    def __init__(self,
                 representation: str = "adiabatic",
                 reference: Any = None,
                 nstates: int = 0,
                 ndof: int = 0):
        """Initialize the adiabatic model.

        Parameters
        ----------
        representation : str, optional
            The representation to use, by default "adiabatic"
        reference : Any, optional
            The reference electronic state, by default None
        nstates : int, optional
            Number of electronic states, by default 0
        ndof : int, optional
            Number of classical degrees of freedom, by default 0

        Raises
        ------
        Exception
            If representation is set to "diabatic"
        """
        if representation == "diabatic":
            raise ValueError(
                'Adiabatic models can only be run in adiabatic mode')
        ElectronicModel_.__init__(self,
                                  representation=representation,
                                  reference=reference,
                                  nstates=nstates,
                                  ndof=ndof)

    def compute(self,
                X: np.ndarray,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None) -> None:
        """Compute electronic properties at position X.

        Parameters
        ----------
        X : ArrayLike
            Position at which to compute properties
        couplings : list of tuple(int, int) or None, optional
            Which coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            Which state forces to compute. None means all.
        reference : Any, optional
            Reference state information, by default None
        """
        self._position = X

        self._reference, self._hamiltonian = self._compute_basis_states(
            self.V(X), reference=reference)
        dV = self.dV(X)

        self._derivative_coupling = self._compute_derivative_coupling(
            self._reference, dV, np.diag(self._hamiltonian))

        # Create new availability arrays to avoid sharing with shallow copies
        self._derivative_couplings_available = np.zeros(
            (self.nstates, self.nstates), dtype=bool)
        if couplings is None:
            self._derivative_couplings_available[:, :] = True
        else:
            for (i, j) in couplings:
                self._derivative_couplings_available[i, j] = True

        self._force = self._compute_force(dV, self._reference)

        # Create new availability array to avoid sharing with shallow copies
        self._forces_available = np.zeros(self.nstates, dtype=bool)
        if gradients is None:
            self._forces_available[:] = True
        else:
            for s in gradients:
                self._forces_available[s] = True

        self._force_matrix = self._compute_force_matrix(dV, self._reference)

    def compute_additional(self,
                           couplings: Any = None,
                           gradients: Any = None) -> None:
        """Compute additional gradients/couplings at the current geometry.

        Since adiabatic models compute everything analytically, this just
        marks the newly requested quantities as available.

        Parameters
        ----------
        couplings : list of tuple(int, int) or None, optional
            Coupling pairs to make available. None means all.
        gradients : list of int or None, optional
            State indices whose forces should be made available. None means all.
        """
        needed_g = self._needed_gradients(gradients)
        needed_c = self._needed_couplings(couplings)
        for s in needed_g:
            self._forces_available[s] = True
        for (i, j) in needed_c:
            self._derivative_couplings_available[i, j] = True

    def update(self,
               X: np.ndarray,
               electronics: Any = None,
               couplings: Any = None,
               gradients: Any = None) -> AdiabaticModel_:
        """Update the model with new position and electronic information.

        Parameters
        ----------
        X : ArrayLike
            New position
        electronics : Any, optional
            Electronic state information, by default None
        couplings : Any, optional
            Coupling information, by default None
        gradients : Any, optional
            Gradient information, by default None

        Returns
        -------
        AdiabaticModel_
            Updated model instance
        """
        out = cp.copy(self)
        if electronics:
            self._reference = electronics._reference
        out._position = X
        out.compute(X,
                    couplings=couplings,
                    gradients=gradients,
                    reference=self._reference)
        return out

    def _compute_basis_states(
            self,
            V: np.ndarray,
            reference: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self._representation == "adiabatic":
            en, co = np.linalg.eigh(V)
            nst = self.nstates
            coeff = co[:, :nst]
            energies = en[:nst]

            if reference is not None:
                try:
                    for mo in range(self.nstates):
                        if np.dot(coeff[:, mo], reference[:, mo]) < 0.0:
                            coeff[:, mo] *= -1.0
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to regularize new ElectronicStates from a reference object {reference}"
                    ) from exc
            return coeff, np.diag(energies)
        elif self._representation == "diabatic":
            raise ValueError(
                "Adiabatic models can only be run in adiabatic mode")
        else:
            raise ValueError("Unrecognized representation")

    def _compute_force(self, dV: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        r""":math:`-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle`"""
        nst = self.nstates
        ndof = self.ndof

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndof], dtype=np.float64)
        for ist in range(self.nstates):
            out[ist, :] += -np.einsum("i,ix->x", coeff[:, ist], half[:, ist, :])
        return out

    def _compute_force_matrix(self, dV: np.ndarray,
                              coeff: np.ndarray) -> np.ndarray:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: np.ndarray, dV: np.ndarray,
                                     energies: np.ndarray) -> np.ndarray:
        r"""Compute derivative couplings :math:`d^\alpha_{ij} = \langle \phi_i | \nabla_\alpha \phi_j \rangle`.

        Uses the Hellmann-Feynman relation to compute derivative couplings from
        the energy gap. When the energy gap between two states is smaller than
        :attr:`coupling_energy_threshold`, the gap is clamped to avoid numerical
        instability.
        """
        if self._representation == "diabatic":
            return np.zeros([self.nstates, self.nstates, self.ndof],
                            dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        thresh = self.coupling_energy_threshold
        for j in range(self.nstates):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < thresh:
                    dE = np.copysign(thresh, dE)

                out[i, j, :] /= dE
                out[j, i, :] /= -dE

        return out

    def V(self, X: np.ndarray) -> np.ndarray:
        """Return the full electronic Hamiltonian matrix V(X)."""
        raise NotImplementedError(
            "Adiabatic models must implement the function V")

    def dV(self, X: np.ndarray) -> np.ndarray:
        """Return the gradient of the electronic Hamiltonian dV/dX."""
        raise NotImplementedError(
            "Adiabatic models must implement the function dV")
