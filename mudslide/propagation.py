# -*- coding: utf-8 -*-
"""Propagators for ODEs from quantum dynamics"""

from __future__ import annotations

from typing import Callable
import numpy as np
from numpy.typing import ArrayLike


def propagate_exponential(rho0: np.ndarray, H: np.ndarray,
                          dt: float) -> None:
    """Propagate density matrix in place using exponential of Hamiltonian.

    Propagates rho0 in place by exponentiating exp(-i H dt).
    H is assumed to be Hermitian, but can be complex.

    Parameters
    ----------
    rho0 : np.ndarray
        Initial density matrix
    H : np.ndarray
        Effective Hamiltonian for propagation
    dt : float
        Time step for propagation

    Returns
    -------
    None
        The density matrix is modified in place
    """
    diags, coeff = np.linalg.eigh(H)

    U = np.linalg.multi_dot(
        [coeff, np.diag(np.exp(-1j * diags * dt)),
         coeff.T.conj()])
    np.dot(U, np.dot(rho0, U.T.conj()), out=rho0)



def propagate_interpolated_rk4(rho0: np.ndarray, h0: np.ndarray,
                               tau0: np.ndarray, vel0: np.ndarray,
                               h1: np.ndarray, tau1: np.ndarray,
                               vel1: np.ndarray, dt: float,
                               nsteps: int) -> None:
    """Propagate density matrix using linearly interpolated quantities and RK4.

    Propagate density matrix forward by linearly interpolating all quantities
    and using RK4. Note: this is split off so a faster version could hypothetically be
    implemented.

    Parameters
    ----------
    rho0 : np.ndarray
        Input/output density matrix
    h0 : np.ndarray
        Hamiltonian from prior step
    tau0 : np.ndarray
        Derivative coupling from prior step
    vel0 : np.ndarray
        Velocity from prior step
    h1 : np.ndarray
        Hamiltonian from current step
    tau1 : np.ndarray
        Derivative coupling from current step
    vel1 : np.ndarray
        Velocity from current step
    dt : float
        Time step
    nsteps : int
        Number of inner time steps

    Returns
    -------
    None
        The density matrix is modified in place
    """
    TV00 = np.einsum("ijx,x->ij", tau0, vel0)
    TV11 = np.einsum("ijx,x->ij", tau1, vel1)
    TV01 = np.einsum("ijx,x->ij", tau0, vel1) + np.einsum(
        "ijx,x->ij", tau1, vel0)

    eigs, vecs = np.linalg.eigh(h0)

    H0 = np.linalg.multi_dot([vecs.T, h0, vecs])
    H1 = np.linalg.multi_dot([vecs.T, h1, vecs])
    W00 = np.linalg.multi_dot([vecs.T, TV00, vecs])
    W11 = np.linalg.multi_dot([vecs.T, TV11, vecs])
    W01 = np.linalg.multi_dot([vecs.T, TV01, vecs])

    def ydot(rho: np.ndarray, t: float) -> np.ndarray:
        """Calculate time derivative of density matrix.

        Parameters
        ----------
        rho : np.ndarray
            Current density matrix
        t : float
            Current time point

        Returns
        -------
        ArrayLike
            Time derivative of density matrix
        """
        assert t >= 0.0 and t <= dt
        w0 = 1.0 - t / dt
        w1 = t / dt

        ergs = np.exp(1j * eigs * t).reshape([1, -1])
        phases = np.dot(ergs.T, ergs.conj())

        H = H0 * (w0 - 1.0) + H1 * w1
        Hbar = H - 1j * (w0 * w0 * W00 + w1 * w1 * W11 + w0 * w1 * W01)
        HI = Hbar * phases

        out = -1j * (np.dot(HI, rho) - np.dot(rho, HI))
        return out

    tmprho = np.linalg.multi_dot([vecs.T, rho0, vecs])
    tmprho = rk4(tmprho, ydot, 0.0, dt, nsteps)
    ergs = np.exp(1j * eigs * dt).reshape([1, -1])
    phases = np.dot(ergs.T.conj(), ergs)

    rho0[:, :] = np.linalg.multi_dot([vecs, tmprho * phases, vecs.T])



def rk4(y0: np.ndarray, ydot: Callable, t0: float, tf: float,
        nsteps: int) -> np.ndarray:
    """Propagate using 4th-order Runge-Kutta (RK4) method.

    Parameters
    ----------
    y0 : np.ndarray
        Initial state vector
    ydot : Callable
        Function that computes the time derivative of y
    t0 : float
        Initial time
    tf : float
        Final time
    nsteps : int
        Number of integration steps

    Returns
    -------
    ArrayLike
        Final state vector after propagation
    """
    dt = (tf - t0) / nsteps

    y = np.copy(y0)

    for i in range(nsteps):
        t = t0 + i * dt

        k1 = ydot(y, t)
        k2 = ydot(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = ydot(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = ydot(y + dt * k3, t + dt)

        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y
