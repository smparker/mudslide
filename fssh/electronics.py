#!/usr/bin/env python
## @package electronics
#  Package responsible for handling the storage and computation of
#  electronic degrees of freedom

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018, Shane Parker <smparker@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function, division

import copy as cp

import numpy as np

class ElectronicModel_(object):
    '''
    Base class for handling electronic structure part of dynamics
    '''
    def __init__(self, representation="adiabatic", reference=None):
        self.representation = representation
        self.position = None
        self.reference = reference

    def compute(self, X, couplings=None, gradients=None, reference=None):
        raise NotImplementedError("ElectronicModels need a compute function")

    def update(self, X, couplings=None, gradients=None):
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

class DiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems given in
    simple diabatic forms.
    '''
    def __init__(self, representation="adiabatic", reference=None):
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

    def nstates(self):
        return self.nstates_

    def ndim(self):
        return self.ndim_

    def compute(self, X, couplings=None, gradients=None, reference=None):
        self.position = X

        self.reference, self.hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self.derivative_coupling = self._compute_derivative_coupling(self.reference,
                dV, np.diag(self.hamiltonian))

        self.force = self._compute_force(dV, self.reference)

    def update(self, X, couplings=None, gradients=None):
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

    ## Computes coefficient matrix for basis states
    # if a diabatic representation is chosen, no transformation takes place
    # @param reference optional ElectronicStates from previous step used only to fix phase
    def _compute_basis_states(self, V, reference=None):
        if self.representation == "adiabatic":
            energies, coeff = np.linalg.eigh(V)
            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:,mo], reference[:,mo]) < 0.0):
                            coeff[:,mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
            return coeff, np.diag(energies)
        elif self.representation == "diabatic":
            return np.eye(self.nstates(), dtype=np.float64), V

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    def _compute_force(self, dV, coeff):
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndim], dtype=np.float64)
        for ist in range(self.nstates()):
            out[ist,:] += -np.einsum("i,ix->x", coeff[:,ist], half[:,ist,:])
        return out

    ## returns \f$F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle\f$
    def _compute_force_matrix(self, coeff, dV):
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def _compute_derivative_coupling(self, coeff, dV, energies):
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

    def V(self, X):
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X):
        raise NotImplementedError("Diabatic models must implement the function dV")


class AdiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems that have an auxiliary electronic problem admitting
    many electronic states that are truncated to just a few. Sort of a truncated DiabaticModel_.
    '''
    def __init__(self, representation="adiabatic", reference=None):
        if representation=="diabatic":
            raise Exception('Adiabatic models can only be run in adiabatic mode')
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

    def nstates(self):
        return self.nstates_

    def ndim(self):
        return self.ndim_

    def compute(self, X, couplings=None, gradients=None, reference=None):
        self.position = X

        self.reference, self.hamiltonian = self._compute_basis_states(self.V(X), reference=reference)
        dV = self.dV(X)

        self.derivative_coupling = self._compute_derivative_coupling(self.reference,
                dV, np.diag(self.hamiltonian))

        self.force = self._compute_force(dV, self.reference)

    def update(self, X, couplings=None, gradients=None):
        self.position = X
        out = cp.copy(self)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

    ## Computes coefficient matrix for basis states
    # if a diabatic representation is chosen, no transformation takes place
    # @param reference optional ElectronicStates from previous step used only to fix phase
    def _compute_basis_states(self, V, reference=None):
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

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    def _compute_force(self, dV, coeff):
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndim], dtype=np.float64)
        for ist in range(self.nstates()):
            out[ist,:] += -np.einsum("i,ix->x", coeff[:,ist], half[:,ist,:])
        return out

    ## returns \f$F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle\f$
    def _compute_force_matrix(self, coeff, dV):
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def _compute_derivative_coupling(self, coeff, dV, energies):
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

    def V(self, X):
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X):
        raise NotImplementedError("Diabatic models must implement the function dV")

