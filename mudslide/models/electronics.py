# -*- coding: utf-8 -*-
"""Handle storage and computation of electronic degrees of freedom"""

import copy as cp
from typing import Tuple, Any

import numpy as np

from mudslide.typing import ArrayLike

class ElectronicModel_(object):
    """
    Base class for handling electronic structure part of dynamics
    """
    def __init__(self, representation: str = "adiabatic", reference: Any = None,
                 nstates: int = 0, ndim: int = 0):
        self.ndim_ = ndim
        self.nstates_ = nstates

        self._representation = representation
        self._position: ArrayLike
        self._reference = reference

        self._hamiltonian: ArrayLike
        self._force: ArrayLike
        self._forces_available: ArrayLike = np.zeros(self.nstates(), dtype=bool)
        self._derivative_coupling: ArrayLike
        self._derivative_couplings_available: ArrayLike = np.zeros((self.nstates(), self.nstates()), dtype=bool)

    def ndim(self) -> int:
        """Number of classical degrees of freedom"""
        return self.ndim_

    def nstates(self) -> int:
        """Number of electronic states"""
        return self.nstates_

    def hamiltonian(self) -> ArrayLike:
        """Return the electronic hamiltonian"""
        return self._hamiltonian

    def force(self, state: int=0) -> ArrayLike:
        """Return the force on a given state"""
        if not self._forces_available[state]:
            raise Exception("Force on state %d not available" % state)
        return self._force[state,:]

    def derivative_coupling(self, state1: int, state2: int) -> ArrayLike:
        """Return the derivative coupling between two states"""
        if not self._derivative_couplings_available[state1, state2]:
            raise Exception("Derivative coupling between states %d and %d not available" % (state1, state2))
        return self._derivative_coupling[state1, state2, :]

    def derivative_coupling_tensor(self) -> ArrayLike:
        """Returns the entire derivative coupling tensor

        The derivative coupling tensor is a rank 3 tensor
        with shape (nstates, nstates, ndim) where the
        derivative_coupling_tensor[i,j,:] is the derivative
        coupling between states i and j

        If the derivative coupling tensor between two states
        is not available, the corresponding element will be
        zero.
        """
        return self._derivative_coupling

    def NAC_matrix(self, velocity: ArrayLike) -> ArrayLike:
        """Return the non-adiabatic coupling matrix
        for a given velocity vector
        """
        if not np.all(self._derivative_couplings_available):
            raise Exception("NAC_matrix needs all derivative couplings")
        return np.einsum("ijk,k->ij", self._derivative_coupling, velocity)

    def force_matrix(self) -> ArrayLike:
        """Return the force matrix"""
        if not np.all(self._forces_available):
            raise Exception("Force matrix needs all forces")
        return self._force_matrix

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        """
        Central function for model objects. After the compute function exists, the following
        data must be provided:
          - self._hamiltonian -> n x n array containing electronic hamiltonian
          - self.force -> n x ndim array containing the force on each diagonal
          - self._derivative_coupling -> n x n x ndim array containing derivative couplings

        The couplings and gradients options are currently unimplemented, but are
        intended to allow specification of which couplings and gradients are needed
        so that computational cost can be reduced.

        Nothing is returned, but the model object should contain all the
        necessary date.
        """
        raise NotImplementedError("ElectronicModels need a compute function")

    def update(self, X: ArrayLike, electronics: Any = None, couplings: Any = None, gradients: Any = None) -> 'ElectronicModel_':
        """
        Convenience function that copies the present object, updates the position,
        calls compute, and then returns the new object
        """
        out = cp.copy(self)

        if electronics and electronics._reference is not None:
            reference = electronics._reference
        else:
            reference = self._reference

        out.compute(X, couplings=couplings, gradients=gradients, reference=reference)
        return out

    def clone(self):
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

    def as_dict(self):
        """Return a dictionary representation of the model"""
        out = {
            "nstates": self.nstates(),
            "ndim": self.ndim(),
            "position": self._position.tolist(),
            "hamiltonian": self._hamiltonian.tolist(),
            "force": self._force.tolist()
        }

        for key in [ "_derivative_coupling", "_force_matrix" ]:
            if hasattr(self, key):
                out[key.lstrip('_')] = getattr(self, key).tolist()

        return out


class DiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems given in
    simple diabatic forms.

    To derive from DiabaticModel_, the following functions
    must be implemented:
        - def V(self, X: ArrayLike) -> ArrayLike
          V(x) should return an ndarray of shape (nstates, nstates)
        - def dV(self, X: ArrayLike) -> ArrayLike
          dV(x) shoudl return an ndarry of shape (nstates, nstates, ndim)
    '''

    def __init__(self, representation: str = "adiabatic", reference: Any = None,
                 nstates:int = 0, ndim: int = 0):
        ElectronicModel_.__init__(self, representation=representation, reference=reference,
                                  nstates=nstates, ndim=ndim)

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        self._position = X

        self._reference, self._hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self._derivative_coupling = self._compute_derivative_coupling(self._reference, dV, np.diag(self._hamiltonian))
        self._derivative_couplings_available[:,:] = True

        self._force = self._compute_force(dV, self._reference)
        self._forces_available[:] = True

        self._force_matrix = self._compute_force_matrix(dV, self._reference)

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None) -> Tuple[ArrayLike, ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self._representation == "adiabatic":
            energies, coeff = np.linalg.eigh(V)
            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:, mo], reference[:, mo]) < 0.0):
                            coeff[:, mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" %
                                    (reference))
            return (coeff, np.diag(energies))
        elif self._representation == "diabatic":
            return (np.eye(self.nstates(), dtype=np.float64), V)
        else:
            raise Exception("Unrecognized run mode")

    def _compute_force(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r""":math:`-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle`"""
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndim], dtype=np.float64)
        for ist in range(self.nstates()):
            out[ist, :] += -np.einsum("i,ix->x", coeff[:, ist], half[:, ist, :])
        return out

    def _compute_force_matrix(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: ArrayLike, dV: ArrayLike, energies: ArrayLike) -> ArrayLike:
        r"""returns :math:`\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}`"""
        if self._representation == "diabatic":
            return np.zeros([self.nstates(), self.nstates(), self.ndim()], dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < 1.0e-10:
                    dE = np.copysign(1.0e-10, dE)

                out[i, j, :] /= dE
                out[j, i, :] /= -dE
            out[j, j, :] = 0.0

        return out

    def V(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function dV")


class AdiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems that have an auxiliary electronic problem admitting
    many electronic states that are truncated to just a few. Sort of a truncated DiabaticModel_.
    '''

    def __init__(self, representation: str = "adiabatic", reference: Any = None,
                 nstates:int = 0, ndim: int = 0):
        if representation == "diabatic":
            raise Exception('Adiabatic models can only be run in adiabatic mode')
        ElectronicModel_.__init__(self, representation=representation, reference=reference,
                                  nstates=nstates, ndim=ndim)

    def nstates(self) -> int:
        return self.nstates_

    def ndim(self) -> int:
        return self.ndim_

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        self._position = X

        self._reference, self._hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self._derivative_coupling = self._compute_derivative_coupling(self._reference, dV, np.diag(self._hamiltonian))
        self._derivative_couplings_available[:,:] = True

        self._force = self._compute_force(dV, self._reference)
        self._forces_available[:] = True

        self._force_matrix = self._compute_force_matrix(dV, self._reference)

    def update(self, X: ArrayLike, electronics: Any = None, couplings: Any = None, gradients: Any = None) -> 'AdiabaticModel_':
        out = cp.copy(self)
        if electronics:
            self._reference = electronics._reference
        out._position = X
        out.compute(X, couplings=couplings, gradients=gradients, reference=self._reference)
        return out

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None) -> Tuple[ArrayLike, ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self._representation == "adiabatic":
            en, co = np.linalg.eigh(V)
            nst = self.nstates()
            coeff = co[:, :nst]
            energies = en[:nst]

            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:, mo], reference[:, mo]) < 0.0):
                            coeff[:, mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" %
                                    (reference))
            return coeff, np.diag(energies)
        elif self._representation == "diabatic":
            raise Exception("Adiabatic models can only be run in adiabatic mode")
            return None
        else:
            raise Exception("Unrecognized representation")

    def _compute_force(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r""":math:`-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle`"""
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndim], dtype=np.float64)
        for ist in range(self.nstates()):
            out[ist, :] += -np.einsum("i,ix->x", coeff[:, ist], half[:, ist, :])
        return out

    def _compute_force_matrix(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: ArrayLike, dV: ArrayLike, energies: ArrayLike) -> ArrayLike:
        r"""returns :math:`\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}`"""
        if self._representation == "diabatic":
            return np.zeros([self.nstates(), self.nstates(), self.ndim()], dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i, j, :] /= dE
                out[j, i, :] /= -dE

        return out

    def V(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function dV")
