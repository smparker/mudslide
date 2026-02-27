# -*- coding: utf-8 -*-
"""Propagating Augmented-FSSH (A-FSSH) trajectories."""

from __future__ import annotations

from typing import Any, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ComputeError, ConfigurationError
from .util import is_string
from .math import poisson_prob_scale
from .propagation import rk4
from .surface_hopping_md import SurfaceHoppingMD
from .propagator import Propagator_

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_
    from .surface_hopping_propagator import SHPropagator


IMAGINARY_NORM_TOLERANCE: float = 1e-10
ZERO_DIVISION_FLOOR: float = 1e-10


class AFSSHVVPropagator(Propagator_):
    """Surface Hopping Velocity Verlet propagator."""

    def __init__(self, **options: Any) -> None:
        """Initialize the propagator.

        Parameters
        ----------
        **options : Any
            Option dictionary for configuration.
        """
        super().__init__()

    def __call__(self, traj: 'AugmentedFSSH', nsteps: int) -> None:  # type: ignore[override]
        """Propagate trajectory using Surface Hopping Velocity Verlet algorithm.

        Parameters
        ----------
        traj : AugmentedFSSH
            Trajectory object to propagate.
        nsteps : int
            Number of steps to propagate.
        """
        dt = traj.dt
        # first update nuclear coordinates
        for _ in range(nsteps):
            # Advance position using Velocity Verlet
            acceleration = traj.force(traj.electronics) / traj.mass
            traj.last_position = traj.position
            traj.position += traj.velocity * dt + 0.5 * acceleration * dt * dt

            traj.advance_delR(traj.last_electronics, traj.electronics)  # type: ignore[arg-type]

            # calculate electronics at new position
            traj.last_electronics, traj.electronics = traj.electronics, traj.model.update(
                traj.position,
                electronics=traj.electronics,
                gradients=traj.needed_gradients(),
                couplings=traj.needed_couplings())

            # Update velocity using Velocity Verlet
            last_acceleration = traj.force(traj.last_electronics) / traj.mass
            this_acceleration = traj.force(traj.electronics) / traj.mass
            traj.last_velocity = traj.velocity
            traj.velocity += 0.5 * (last_acceleration + this_acceleration) * dt

            assert traj.last_electronics is not None
            assert traj.electronics is not None
            traj.advance_delP(traj.last_electronics, traj.electronics)

            # now propagate the electronic wavefunction to the new time
            traj.propagate_electronics(traj.last_electronics, traj.electronics,
                                       dt)
            traj.surface_hopping(traj.last_electronics, traj.electronics)

            traj.time += dt
            traj.nsteps += 1


class AFSSHPropagator(Propagator_):  # pylint: disable=abstract-method
    """Surface Hopping propagator factory.

    This class serves as a factory for creating different types of propagators
    used in adiabatic FSSH molecular dynamics simulations.
    """

    def __new__(cls, model: ElectronicModel_, prop_options: Any = "vv") -> Propagator_:  # type: ignore[misc]
        """Create a new surface hopping propagator instance.

        Parameters
        ----------
        model : Any
            Model object defining the problem.
        prop_options : Any, optional
            Propagator options, can be a string or dictionary, by default "vv".

        Returns
        -------
        SHPropagator
            A new propagator instance.

        Raises
        ------
        ValueError
            If the propagator type is unknown.
        Exception
            If prop_options is not a string or dictionary.
        """
        if is_string(prop_options):
            prop_options = {"type": prop_options}
        elif not isinstance(prop_options, dict):
            raise ConfigurationError("prop_options must be a string or a dictionary")

        proptype = prop_options.get("type", "vv")
        if proptype.lower() == "vv":
            return AFSSHVVPropagator(**prop_options)
        raise ConfigurationError(
            f"Unrecognized surface hopping propagator type: {proptype}.")


class AugmentedFSSH(SurfaceHoppingMD):
    """Augmented-FSSH (A-FSSH) dynamics, by Subotnik and coworkers.

    Initial implementation based on original paper:
      Subotnik, Shenvi JCP 134, 024105 (2011); doi: 10.1063/1.3506779
    """

    recognized_options = SurfaceHoppingMD.recognized_options + [
        "augmented_integration"
    ]

    def __init__(self, *args: Any, **options: Any):
        options[
            'hopping_method'] = 'instantaneous'  # force instantaneous hopping
        SurfaceHoppingMD.__init__(self, *args, **options)

        self.augmented_integration = options.get(
            "augmented_integration", self.electronic_integration).lower()

        self.delR = np.zeros(
            [self.model.ndof, self.model.nstates, self.model.nstates],
            dtype=np.complex128)
        self.delP = np.zeros(
            [self.model.ndof, self.model.nstates, self.model.nstates],
            dtype=np.complex128)

        self.propagator: Propagator_ = AFSSHPropagator(self.model, "vv")  # type: ignore[assignment]

    def needed_gradients(self) -> list[int] | None:
        """A-FSSH needs all forces for force_matrix computation.

        Returns
        -------
        None
            None means all state gradients are needed.
        """
        return None

    def compute_delF(self, this_electronics: ElectronicModel_) -> np.ndarray:
        """Compute the difference in forces between states.

        Parameters
        ----------
        this_electronics : ElectronicModel
            Current electronic state information.

        Returns
        -------
        numpy.ndarray
            Matrix of force differences between states.
        """
        delF = np.copy(this_electronics.force_matrix)
        F0 = self.force(this_electronics)
        for i in range(self.model.nstates):
            delF[i, i, :] -= F0
        return delF

    def advance_delR(self, last_electronics: ElectronicModel_, this_electronics: ElectronicModel_) -> None:
        """Propagate delR using Eq. (29) from Subotnik 2011 JCP.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic state information from previous step.
        this_electronics : ElectronicModel
            Electronic state information from current step.
        """
        dt = self.dt
        H = self.hamiltonian_propagator(last_electronics, this_electronics)
        delV = np.zeros_like(self.delP)
        for x in range(self.delP.shape[0]):
            delV[x, :, :] = self.delP[x, :, :] / self.mass[x]

        if self.augmented_integration == "exp":
            eps, co = np.linalg.eigh(H)
            expiht = np.exp(-1j * dt * np.subtract.outer(eps, eps))

            delV = np.einsum("pi,xpq,qj->xij", co.conj(), delV, co)
            RR = np.einsum("pi,xpq,qj->xij", co.conj(), self.delR, co)
            Rt = (RR + delV * dt) * expiht
            self.delR = np.einsum("pi,xij,qj->xpq", co, Rt, co.conj())
        elif self.augmented_integration == "rk4":

            def ydot(RR: np.ndarray, t: float) -> np.ndarray:
                assert t >= 0.0 and t <= dt
                HR = np.einsum("pr,xrq->xpq", H, RR)
                RH = np.einsum("xpr,rq->xpq", RR, H)

                return -1j * (HR - RH) + delV

            nsteps = 4
            Rt = rk4(self.delR, ydot, 0.0, dt, nsteps)
            self.delR = Rt
        else:
            raise ConfigurationError(
                f"Unrecognized augmented integration method: {self.augmented_integration}"
            )

    def advance_delP(self, last_electronics: ElectronicModel_, this_electronics: ElectronicModel_) -> None:
        """Propagate delP using Eq. (31) from Subotnik JCP 2011.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic state information from previous step.
        this_electronics : ElectronicModel
            Electronic state information from current step.
        """
        dt = self.dt
        H = self.hamiltonian_propagator(last_electronics, this_electronics)
        delF = self.compute_delF(this_electronics)

        if self.augmented_integration == "exp":
            eps, co = np.linalg.eigh(H)

            expiht = np.exp(-1j * dt * np.subtract.outer(eps, eps))
            eee = np.subtract.outer(2 * eps, np.add.outer(eps, eps))
            poiss = -poisson_prob_scale(1j * eee * dt) * dt
            poiss_star = -poisson_prob_scale(-1j * eee * dt) * dt

            delF = np.einsum("pi,pqx,qj->xij", co.conj(), delF, co)
            PP = np.einsum("pi,xpq,qj->xij", co.conj(), self.delP, co)
            rho = np.einsum("pi,pq,qj->ij", co.conj(), self.rho, co)

            FF = np.einsum("xik,kj,jik->xij", delF, rho, poiss) + np.einsum(
                "ik,xkj,ijk->xij", rho, delF, poiss_star)
            FF *= -0.5

            Pt = (PP + FF) * expiht
            self.delP = np.einsum("pi,xij,qj->xpq", co, Pt, co.conj())
        elif self.augmented_integration == "rk4":
            dFrho_comm = np.einsum("prx,rq->xpq", delF, self.rho) + np.einsum(
                "pr,rqx->xpq", self.rho, delF)
            dFrho_comm *= 0.5

            def ydot(PP: np.ndarray, t: float) -> np.ndarray:
                assert t >= 0.0 and t <= dt
                HP = np.einsum("pr,xrq->xpq", H, PP)
                PH = np.einsum("xpr,rq->xpq", PP, H)

                return -1j * (HP - PH) + dFrho_comm

            nsteps = 4
            Pt = rk4(self.delP, ydot, 0.0, dt, nsteps)
            self.delP = Pt
        else:
            raise ConfigurationError(
                f"Unrecognized augmented integration method: {self.augmented_integration}"
            )
        return

    def direction_of_rescale(
            self,
            source: int,
            target: int,
            electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Return direction in which to rescale momentum.

        In Subotnik JCP 2011, they suggest to use the difference between the momenta on delP.

        Parameters
        ----------
        source : int
            Active state before hop.
        target : int
            Active state after hop.
        electronics : ElectronicModel, optional
            Electronic model information (ignored), by default None.

        Returns
        -------
        numpy.ndarray
            Unit vector pointing in direction of rescale.
        """
        out = self.delP[:, source, source] - self.delP[:, target, target]
        if np.linalg.norm(np.imag(out)) >= IMAGINARY_NORM_TOLERANCE:
            raise ComputeError(
                "Rescale direction has unexpectedly large imaginary component: "
                f"{np.linalg.norm(np.imag(out)):.2e}")
        return np.real(out)

    def gamma_collapse(self,
                       electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Compute probability of collapse to each electronic state.

        Uses Eq. (55) in Subotnik JCP 2011. This formula has some major problems
        and is tweaked or abandoned in future papers.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            Electronic model information (forces), by default None.

        Returns
        -------
        numpy.ndarray
            Array of collapse probabilities for each state.
        """
        nst = self.model.nstates
        ndof = self.model.ndof
        out = np.zeros(nst, dtype=np.float64)

        def shifted_diagonal(X: np.ndarray, k: int) -> np.ndarray:
            out = np.zeros([nst, ndof])
            for i in range(nst):
                out[i, :] = np.real(X[:, k, k] - X[:, i, i])
            return out

        ddR = shifted_diagonal(self.delR, self.state)
        ddP = shifted_diagonal(self.delP, self.state)
        ddP = np.where(np.abs(ddP) == 0.0, ZERO_DIVISION_FLOOR, ddP)
        assert electronics is not None
        ddF = shifted_diagonal(np.einsum("pqx->xpq", electronics.force_matrix),
                               self.state)
        ddR = ddR * np.sign(ddR / ddP)

        for i in range(nst):
            out[i] = np.dot(ddF[i, :], ddR[i, :])

        out[self.state] = 0.0  # zero out self collapse for safety

        return 0.5 * out * self.dt

    def surface_hopping(self, last_electronics: ElectronicModel_,
                        this_electronics: ElectronicModel_) -> None:
        """Specialized version of surface_hopping that handles collapsing.

        Parameters
        ----------
        last_electronics : ElectronicModel
            ElectronicStates from previous step.
        this_electronics : ElectronicModel
            ElectronicStates from current step.
        """
        SurfaceHoppingMD.surface_hopping(self, last_electronics,
                                         this_electronics)

        gamma = self.gamma_collapse(this_electronics)

        eta = np.zeros_like(gamma)
        for i in range(self.model.nstates):
            if i == self.state:
                continue
            e = self.random()
            eta[i] = e

            if e < gamma[i]:
                if self.model.nstates != 2:
                    raise NotImplementedError(
                        "A-FSSH collapse is only implemented for 2-state systems"
                    )

                # reset the density matrix
                self.rho[:, :] = 0.0
                self.rho[self.state, self.state] = 1.0

                # reset delR and delP
                self.delR[:, :, :] = 0.0
                self.delP[:, :, :] = 0.0

                self.tracer.record_event(
                    {
                        "time": self.time,
                        "removed": i,
                        "gamma": gamma[i],
                        "eta": eta
                    },
                    event_type="collapse")

    def hop_update(self, hop_from: int, hop_to: int) -> None:
        """Shift delR and delP after hops.

        Parameters
        ----------
        hop_from : int
            State index before hop.
        hop_to : int
            State index after hop.
        """
        dRb = self.delR[:, hop_to, hop_to]
        dPb = self.delP[:, hop_to, hop_to]

        for i in range(self.model.nstates):
            self.delR[:, i, i] -= dRb
            self.delP[:, i, i] -= dPb
