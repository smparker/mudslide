# -*- coding: utf-8 -*-
"""Implementations of the one-dimensional two-state models Tully demonstrated FSSH on in Tully, J.C. <I>J. Chem. Phys.</I> 1990 <B>93</B> 1061."""

import numpy as np

from typing import Any

from mudslide.models.electronics import DiabaticModel_

from mudslide.typing import ArrayLike

class SubotnikW(DiabaticModel_):
    def __init__(
            self,
            representation: str = "adiabatic",
            reference: Any = None,
            mass: np.float64 = 2000.0,
            nstates: int = 8,
            eps: np.float64 = 0.1
    ):
        DiabaticModel_.__init__(self, representation=representation, reference=reference,
                                nstates=nstates, ndim=1)
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())
        self.eps = eps

    def V(self, X: ArrayLike) -> ArrayLike:
        N = self.nstates()
        m = np.arange(0, N)

        v = 0.1 / np.sqrt(N)
        diag = np.tan(0.5 * np.pi - (2 * m - 1) * np.pi / (2 * N)) * X[0] + \
                (m - 1) * self.eps

        out = np.zeros([N, N], dtype=np.float64)
        out[:, :] = v
        np.fill_diagonal(out, diag)
        return out

    def dV(self, X: ArrayLike) -> ArrayLike:
        N = self.nstates()
        m = np.arange(0, N)

        v = 0.1 / np.sqrt(N)
        diag = np.tan(0.5 * np.pi - (2 * m - 1) * np.pi / (2 * N)) + \
                (m - 1) * self.eps

        out = np.zeros([1, N, N], dtype=np.float64)
        out[0, :, :] = v
        np.fill_diagonal(out[0], diag)
        return out

class SubotnikZ(DiabaticModel_):
    def __init__(
            self,
            representation: str = "adiabatic",
            reference: Any = None,
            mass: np.float64 = 2000.0,
            nstates: int = 8,
            eps: np.float64 = 0.1
    ):
        DiabaticModel_.__init__(self, representation=representation, reference=reference,
                                nstates=nstates, ndim=1)
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())
        self.eps = eps

    def V(self, X: ArrayLike) -> ArrayLike:
        N = self.nstates()

        v = 0.1 / np.sqrt(N)
        m1 = np.arange(0, N//2)
        m2 = np.arange(N//2, N)

        diag = np.zeros(N, dtype=np.float64)
        d1 = X[0] + (m1 - 1) * self.eps
        d2 = -X[0] + (N - m2) * self.eps
        diag[:len(m1)] = d1
        diag[len(m1):] = d2

        out = np.zeros([N, N], dtype=np.float64)
        out[:, :] = v
        np.fill_diagonal(out, diag)
        return out

    def dV(self, X: ArrayLike) -> ArrayLike:
        N = self.nstates()

        v = 0.1 / np.sqrt(N)
        m1 = np.arange(0, N//2)
        m2 = np.arange(N//2, N)

        diag = np.zeros(N, dtype=np.float64)
        d1 = X[0] + (m1 - 1) * self.eps
        d2 = -X[0] + (N - m2) * self.eps
        diag[:len(m1)] = d1
        diag[len(m1):] = d2

        out = np.zeros([1, N, N], dtype=np.float64)
        out[0, :, :] = v
        np.fill_diagonal(out[0], diag)
        return out

