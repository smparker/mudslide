# -*- coding: utf-8 -*-
"""Propagators for ODEs from quantum dynamics"""

from __future__ import division

import numpy as np

from typing import Callable
from .typing import ArrayLike, DtypeLike

def propagate_exponential(rho0: ArrayLike, H: ArrayLike, dt: DtypeLike) -> None:
    """
    Propagates rho0 in place by exponentiating exp(-i H dt).
    H is assumed to be Hermitian, but can be complex.

    :param param0: initial density matrix
    :param H: effective Hamiltonian for propagation
    """
    diags, coeff = np.linalg.eigh(H)

    U = np.linalg.multi_dot([ coeff, np.diag(np.exp(-1j * diags * dt)), coeff.T.conj() ])
    np.dot(U, np.dot(rho0, U.T.conj()), out=rho0)

def propagate_interpolated_rk4(rho0: ArrayLike, h0: ArrayLike, tau0: ArrayLike, vel0: ArrayLike,
        h1: ArrayLike, tau1: ArrayLike, vel1: ArrayLike, dt: DtypeLike, nsteps: int) -> None:
    """
    Propagate density matrix forward by linearly interpolating all quantities
    and using RK4. Note: this is split off so a faster version could hypothetically be
    implemented

    :param rho0: input/output density matrix
    :param h0: Hamiltonian from prior step
    :param tau0: derivative coupling from prior step
    :param vel0: velocity from prior step
    :param h1: Hamiltonian from current step
    :param tau1: derivative coupling from current step
    :param vel1: velocity from current step
    :param dt: time step
    :param nsteps: number of inner time steps
    """
    TV00 = np.einsum("ijx,x->ij", tau0, vel0)
    TV11 = np.einsum("ijx,x->ij", tau1, vel1)
    TV01 = np.einsum("ijx,x->ij", tau0, vel1) + np.einsum("ijx,x->ij", tau1, vel0)

    eigs, vecs = np.linalg.eigh(h0)

    H0  = np.linalg.multi_dot([vecs.T, h0, vecs])
    H1  = np.linalg.multi_dot([vecs.T, h1, vecs])
    W00 = np.linalg.multi_dot([vecs.T, TV00, vecs])
    W11 = np.linalg.multi_dot([vecs.T, TV11, vecs])
    W01 = np.linalg.multi_dot([vecs.T, TV01, vecs])

    def ydot(rho: ArrayLike, t: DtypeLike) -> ArrayLike:
        assert t >= 0.0 and t <= dt
        w0 = 1.0 - t/dt
        w1 = t/dt

        ergs = np.exp(1j * eigs * t).reshape([1, -1])
        phases = np.dot(ergs.T, ergs.conj())

        H = H0 * (w0 - 1.0) + H1 * w1
        Hbar = H - 1j * (w0*w0*W00 + w1*w1*W11 + w0*w1*W01)
        HI = Hbar * phases

        out = -1j * ( np.dot(HI, rho) - np.dot(rho, HI) )
        return out

    tmprho = np.linalg.multi_dot([vecs.T, rho0, vecs])
    tmprho = rk4(tmprho, ydot, 0.0, dt, nsteps)
    ergs = np.exp(1j * eigs * dt).reshape([1, -1])
    phases = np.dot(ergs.T.conj(), ergs)

    rho0[:,:] = np.linalg.multi_dot([vecs, tmprho * phases, vecs.T])

def rk4(y0: ArrayLike, ydot: Callable, t0: DtypeLike, tf: DtypeLike, nsteps: int) -> ArrayLike:
    """
    Propagates using 4th-order Runge-Kutta (RK4).
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
