# -*- coding: utf-8 -*-
"""Handle storage and computation of electronic degrees of freedom"""

from __future__ import division

import copy as cp

import numpy as np

from typing import Tuple, Any
from .typing import ArrayLike

class ElectronicModel_(object):
    '''
    Base class for handling electronic structure part of dynamics
    '''
    def __init__(self, representation: str = "adiabatic", reference: Any = None):
        self.representation = representation
        self.position: ArrayLike
        self.reference = reference

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        raise NotImplementedError("ElectronicModels need a compute function")

    def update(self, X: ArrayLike, couplings: Any = None, gradients: Any = None) -> 'ElectronicModel_':
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

class DiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems given in
    simple diabatic forms.
    '''
    nstates_: int
    ndim_ : int

    def __init__(self, representation: str = "adiabatic", reference: Any = None):
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

    def nstates(self) -> int:
        return self.nstates_

    def ndim(self) -> int:
        return self.ndim_

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        self.position = X

        self.reference, self.hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self.derivative_coupling = self._compute_derivative_coupling(self.reference,
                dV, np.diag(self.hamiltonian))

        self.force = self._compute_force(dV, self.reference)

    def update(self, X: ArrayLike, couplings: Any = None, gradients: Any = None) -> 'DiabaticModel_':
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None) -> Tuple[ArrayLike,ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self.representation == "adiabatic":
            energies, coeff = np.linalg.eigh(V)
            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:,mo], reference[:,mo]) < 0.0):
                            coeff[:,mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
            return (coeff, np.diag(energies))
        elif self.representation == "diabatic":
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
            out[ist,:] += -np.einsum("i,ix->x", coeff[:,ist], half[:,ist,:])
        return out

    def _compute_force_matrix(self, coeff: ArrayLike, dV: ArrayLike) -> ArrayLike:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: ArrayLike, dV: ArrayLike, energies: ArrayLike) -> ArrayLike:
        r"""returns :math:`\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}`"""
        if self.representation == "diabatic":
            return np.zeros([self.nstates(), self.nstates(), self.ndim()], dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i,j,:] /= dE
                out[j,i,:] /= -dE
            out[j,j,:] = 0.0

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
    nstates_: int
    ndim_: int

    def __init__(self, representation: str = "adiabatic", reference: Any = None):
        if representation=="diabatic":
            raise Exception('Adiabatic models can only be run in adiabatic mode')
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

    def nstates(self) -> int:
        return self.nstates_

    def ndim(self) -> int:
        return self.ndim_

    def compute(self, X: ArrayLike, couplings: Any = None, gradients: Any = None, reference: Any = None) -> None:
        self.position = X

        self.reference, self.hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self.derivative_coupling = self._compute_derivative_coupling(self.reference,
                dV, np.diag(self.hamiltonian))

        self.force = self._compute_force(dV, self.reference)

    def update(self, X: ArrayLike, couplings: Any = None, gradients: Any = None) -> 'AdiabaticModel_':
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None) -> Tuple[ArrayLike,ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self.representation == "adiabatic":
            en, co = np.linalg.eigh(V)
            nst = self.nstates()
            coeff = co[:,:nst]
            energies = en[:nst]

            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:,mo], reference[:,mo]) < 0.0):
                            coeff[:,mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
            return coeff, np.diag(energies)
        elif self.representation == "diabatic":
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
            out[ist,:] += -np.einsum("i,ix->x", coeff[:,ist], half[:,ist,:])
        return out

    def _compute_force_matrix(self, coeff: ArrayLike, dV: ArrayLike) -> ArrayLike:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(self, coeff: ArrayLike, dV: ArrayLike, energies: ArrayLike) -> ArrayLike:
        r"""returns :math:`\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}`"""
        if self.representation == "diabatic":
            return np.zeros([self.nstates(), self.nstates(), self.ndim()], dtype=np.float64)

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i,j,:] /= dE
                out[j,i,:] /= -dE

        return out

    def V(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function dV")

