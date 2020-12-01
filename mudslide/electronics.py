# -*- coding: utf-8 -*-
"""Handle storage and computation of electronic degrees of freedom"""

from __future__ import division

import copy as cp

import numpy as np

import math

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
        e1 = 8.5037
        e2 = 9.4523
        om = np.array([0.1117, 0.2021, 0.2723, 0.4102])
        k1 = np.array([-0.0456, 0.0399, -0.2139, -0.0864])
        k2 = np.array([-0.0393, 0.0463, 0.2877, -0.1352])
        lamb = 0.3289
        w0 = 0
        r0 = 100 #distance to hydrogen atom - set arbitrarily currently
        mh = 1.01 #mass of hydrogen
        w5 = 1 #freq of neutral ground-state normal vibration - set arbitrarily currently
        A_subn = np.array([1.4823, -0.2191, 0.0525, -0.0118])
        theta = X[5]
        k11 = np.zeros(4)
        k22 = np.zeros(4)
        q5 = np.zeros(4)


        for i in range(4):
            w0 = w0 + ((om[i]/2)*(X[i]**2))
            k11[i] = k1[i]*X[i]
            k22[i] = k2[i]*X[i]
            w12 = w12 + lamb*X[i] + lamb*r0*(math.sqrt(w5*mh))*math.sin(theta)

        for i in A_subn:
            q5[i] = A_subn[i]*((math.sin((i+1)*theta))**2)

        w11 = e1 + w0 + np.sum(k11) + np.sum(q5)
        w22 = e2 + w0 + np.sum(k22) + np.sum(q5)
        w21 = w12

        out = np.array([ [w11, w12],
                        [w21, w22]], dtype = np.float64)
        return out
        

    def dV(self, X: ArrayLike) -> ArrayLike:
        om = np.array([0.1117, 0.2021, 0.2723, 0.4102])
        k1 = np.array([-0.0456, 0.0399, -0.2139, -0.0864])
        k2 = np.array([-0.0393, 0.0463, 0.2877, -0.1352])
        lamb = 0.3289*4
        w0 = 0
        r0 = 100 #distance to hydrogen atom - set arbitrarily currently
        mh = 1.01 #mass of hydrogen
        w5 = 1 #freq of neutral ground-state normal vibration - set arbitrarily currently
        theta = X[5]
        A_subn = np.array([1.4823, -0.2191, 0.0525, -0.0118])
        q5 = np.zeros(4)

        for i in range(4):
            w0 = w0 + om[i]*X[i]

        for i in A_subn:
            q5[i] = A_subn[i]*(i+1)*(math.sin((i+1)*theta)*(math.cos((i+1)*theta)))
            
        w11 = w0 + np.sum(k1) + np.sum(q5)
        w22 = w0 + np.sum(k2) + np.sum(q5)
        w12 = lamb + 0.3289*r0*(math.sqrt(w5*mh))*math.cos(theta)
        w21 = w12

        out = np.array([ [w11, w12],
                        [w21, w22]], dtype = np.float64)
        return out.reshape([1, 2, 2])


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

