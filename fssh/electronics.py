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

import numpy as np

## Wrapper around all the information computed for a set of electronics
#  states at a given position: V, dV, eigenvectors, eigenvalues
#  Parameters with names starting with '_' indicate things that by
#  design should not be called outside of ElectronicStates.
class ElectronicStates(object):
    ## Constructor
    # @param V Hamiltonian/potential
    # @param dV Gradient of Hamiltonian/potential
    # @param reference [optional] set of coefficients (for example, from a previous step) used to keep the sign of eigenvectors consistent
    def __init__(self, V, dV, reference = None):
        # raw internal quantities
        self._V = V
        self._dV = dV
        self._coeff = self._compute_coeffs(reference)

        # processed quantities used in simulation
        self.hamiltonian = self._compute_hamiltonian()
        self.force = self._compute_force()
        self.derivative_coupling = self._compute_derivative_coupling()

    ## returns dimension of (electronic) Hamiltonian
    def nstates(self):
        return self._V.shape[1]

    ## returns dimensionality of model (nuclear coordinates)
    def ndim(self):
        return self._dV.shape[0]

    ## returns coefficient matrix for basis states
    # @param reference optional ElectronicStates from previous step used only to fix phase
    def _compute_coeffs(self, reference=None):
        energies, coeff = np.linalg.eigh(self._V)
        if reference is not None:
            try:
                ref_coeff = reference._coeff
                for mo in range(self.nstates()):
                    if (np.dot(coeff[:,mo], ref_coeff[:,mo]) < 0.0):
                        coeff[:,mo] *= -1.0
            except:
                raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
        self._energies = energies
        return coeff

    ## returns \f$-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle\f$ of Hamiltonian
    def _compute_force(self):
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", self._dV, self._coeff)

        out = np.zeros([nst, ndim])
        for ist in range(self.nstates()):
            out[ist,:] += -np.einsum("i,ix->x", self._coeff[:,ist], half[:,ist,:])
        return out

    ## returns \f$\phi_{\mbox{state}} | H | \phi_{\mbox{state}} = \varepsilon_{\mbox{state}}\f$
    def _compute_hamiltonian(self):
        return np.dot(self._coeff.T, np.dot(self._V, self._coeff))

    ## returns \f$\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}\f$
    def _compute_derivative_coupling(self):
        out = np.einsum("ip,xij,jq->pqx", self._coeff, self._dV, self._coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = self._energies[j] - self._energies[i]
                if abs(dE) < 1.0e-14:
                    dE = m.copysign(1.0e-14, dE)

                out[i,j,:] /= dE
                out[j,i,:] /= -dE

        return out

    ## returns \f$ \sum_\alpha v^\alpha D^\alpha \f$ where \f$ D^\alpha_{ij} = d^\alpha_{ij} \f$
    # @param velocity [ndim] numpy array of velocities
    def NAC_matrix(self, velocity):
        out = np.einsum("pqx,x->pq", self.derivative_coupling, velocity)
        return out

    ## returns \f$F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle\f$
    def force_matrix(self):
        out = -np.einsum("ip,xij,jq->pqx", self._coeff, self._dV, self._coeff)
        return out

