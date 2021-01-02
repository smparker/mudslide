# -*- coding: utf-8 -*-
"""Propagating Augmented-FSSH (A-FSSH) trajectories"""

import numpy as np

from .propagation import rk4
from .trajectory_sh import TrajectorySH

from typing import Any, Union
from .typing import ArrayLike, DtypeLike, ElectronicT

class AugmentedFSSH(TrajectorySH):
    """Augmented-FSSH (A-FSSH) dynamics, by Subotnik and coworkers

    Initial implementation based on original paper:
      Subotnik, Shenvi JCP 134, 024105 (2011); doi: 10.1063/1.3506779
    """
    def __init__(self, *args: Any, **kwargs: Any):
        TrajectorySH.__init__(self, *args, **kwargs)

        self.delR = np.zeros([self.model.ndim(), self.model.nstates(), self.model.nstates()],
                dtype=np.complex128)
        self.delP = np.zeros([self.model.ndim(), self.model.nstates(), self.model.nstates()],
                dtype=np.complex128)

    def advance_position(self, last_electronics: Union[ElectronicT,None],
            this_electronics: ElectronicT) -> None:
        """Move classical position and delR"""
        # Use base class propagation
        TrajectorySH.advance_position(self, last_electronics, this_electronics)

        self.advance_delR(last_electronics, this_electronics)

    def advance_velocity(self, last_electronics: Union[ElectronicT,None],
            this_electronics: ElectronicT) -> None:
        """Move classical velocity and delP"""
        # Use base class propagation
        TrajectorySH.advance_velocity(self, last_electronics, this_electronics)

        self.advance_delP(last_electronics, this_electronics)

    def compute_delF(self, this_electronics):
        delF = np.copy(this_electronics.force_matrix)
        F0 = self.force(this_electronics)
        for i in range(self.model.nstates()):
            delF[i,i,:] -= F0
        return delF

    def advance_delR(self, last_electronics, this_electronics):
        """Propagate delR using Eq. (29) from Subotnik 2011 JCP"""

        dt = self.dt
        H = self.hamiltonian_propagator(last_electronics, this_electronics)
        delV = np.zeros_like(self.delP)
        for x in range(self.delP.shape[0]):
            delV[x,:,:] = self.delP[x,:,:] / self.mass[x]

        def ydot(RR: ArrayLike, t: DtypeLike) -> ArrayLike:
            assert t >= 0.0 and t <= dt
            HR = np.einsum("pr,xrq->xpq", H, RR)
            RH = np.einsum("xpr,rq->xpq", RR, H)

            return -1j * (HR - RH) + delV

        nsteps = 128
        Rt = rk4(self.delR, ydot, 0.0, dt, nsteps)
        self.delR = Rt

    def advance_delP(self, last_electronics, this_electronics):
        """Propagate delP using Eq. (31) from Subotnik JCP 2011"""

        dt = self.dt
        H = self.hamiltonian_propagator(last_electronics, this_electronics)
        delF = self.compute_delF(this_electronics)
        dFrho_comm = np.einsum("prx,rq->xpq", delF, self.rho) + np.einsum("pr,rqx->xpq", self.rho, delF)
        dFrho_comm *= 0.5

        def ydot(PP: ArrayLike, t: DtypeLike) -> ArrayLike:
            assert t >= 0.0 and t <= dt
            HP = np.einsum("pr,xrq->xpq", H, PP)
            PH = np.einsum("xpr,rq->xpq", PP, H)

            return -1j * (HP - PH) + dFrho_comm

        nsteps = 128
        Pt = rk4(self.delP, ydot, 0.0, dt, nsteps)
        self.delP = Pt
        return

    def direction_of_rescale(self, source: int, target: int, electronics: ElectronicT=None) -> np.ndarray:
        """
        Return direction in which to rescale momentum. In Subotnik JCP 2011,
        they suggest to use the difference between the momenta on delP

        :param source: active state before hop
        :param target: active state after hop
        :param electronics: electronic model information (ignored)

        :return: unit vector pointing in direction of rescale
        """
        out = self.delP[:, source, source] - self.delP[:, target, target]
        assert np.linalg.norm(np.imag(out)) < 1e-8
        return np.real(out)

    def gamma_collapse(self, electronics: ElectronicT=None) -> np.ndarray:
        """
        Computes the probability of collapse to each electronic state using Eq. (55)
        in Subotnik JCP 2011.

        This formula has some major problems and is tweaked or abandoned in future papers

        :param electronics: electronic model information (forces)
        """
        nst = self.model.nstates()
        ndim = self.model.ndim()
        out = np.zeros(nst, dtype=np.float64)

        def shifted_diagonal(X, k: int) -> np.ndarray:
            out = np.zeros([nst, ndim])
            for i in range(nst):
                out[i,:] = np.real(X[:,k,k] - X[:,i,i])
            return out

        ddR = shifted_diagonal(self.delR, self.state)
        ddP = shifted_diagonal(self.delP, self.state)
        ddP = np.where(np.abs(ddP) == 0.0, 1e-10, ddP)
        ddF = shifted_diagonal(np.einsum("pqx->xpq", electronics.force_matrix), self.state)
        ddR = ddR * np.sign(ddR/ddP)

        for i in range(nst):
            out[i] = np.dot(ddF[i,:], ddR[i,:])

        out[self.state] = 0.0 # zero out self collapse for safety

        return 0.5 * out * self.dt

    def surface_hopping(self, last_electronics: ElectronicT, this_electronics: ElectronicT) -> None:
        """Specialized version of surface_hopping that handles collapsing

        :param last_electronics: ElectronicStates from previous step
        :param this_electronics: ElectronicStates from current step
        """
        TrajectorySH.surface_hopping(self, last_electronics, this_electronics)

        gamma = self.gamma_collapse(this_electronics)

        eta = np.zeros_like(gamma)
        for i in range(self.model.nstates()):
            if i == self.state:
                continue
            e = self.random()
            eta[i] = e

            if e < gamma[i]:
                assert self.model.nstates() == 2

                # reset the density matrix
                self.rho[:,:] = 0.0
                self.rho[self.state, self.state] = 1.0

                # reset delR and delP
                self.delR[:,:,:] = 0.0
                self.delP[:,:,:] = 0.0

                self.tracer.record_event("collapse", {
                    "time" : self.time,
                    "removed" : i,
                    "gamma" : gamma[i],
                    "eta" : eta
                    })

    def hop_update(self, hop_from, hop_to):
        """Shift delR and delP after hops"""
        dRb = self.delR[:, hop_to, hop_to]
        dPb = self.delP[:, hop_to, hop_to]

        for i in range(self.model.nstates()):
            self.delR[:,i,i] -= dRb
            self.delP[:,i,i] -= dPb
