# -*- coding: utf-8 -*-
"""Propagate Adiabatic MD trajectory"""

from typing import Any

import numpy as np

from .constants import fs_to_au, boltzmann
from .propagator import Propagator_
from .util import remove_center_of_mass_motion, remove_angular_momentum

class VVPropagator(Propagator_):
    """Velocity Verlet propagator"""

    def __init__(self, **options: Any) -> None:
        """Constructor

        :param options: option dictionary
        """
        super().__init__()

    def __call__(self, traj: 'AdiabaticMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
        """
        dt = traj.dt
        # first update nuclear coordinates
        for _ in range(nsteps):
            acceleration = traj.force(traj.electronics) / traj.mass
            traj.last_position = traj.position
            traj.position += traj.velocity * dt + 0.5 * acceleration * dt * dt

            # calculate electronics at new position
            traj.last_electronics = traj.electronics
            traj.electronics = traj.model.update(traj.position, electronics=traj.electronics)

            # update velocity
            last_acceleration = acceleration
            acceleration = traj.force(traj.electronics) / traj.mass

            traj.last_velocity = traj.velocity
            traj.velocity += 0.5 * (last_acceleration + acceleration) * dt

            # optionally remove COM motion and total angular momentum
            if traj.remove_com_every > 0 and (traj.nsteps % traj.remove_com_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                vnew = remove_center_of_mass_motion(v, m)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            if traj.remove_angular_momentum_every > 0 and \
                   (traj.nsteps % traj.remove_angular_momentum_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                x = traj.position.reshape(d)
                vnew = remove_angular_momentum(v, m, x)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            traj.time += dt
            traj.nsteps += 1

class NoseHooverChainPropagator(Propagator_):
    """
    G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and Michael L. Klein,
    "Explicit reversible integrators for extended systems dynamics"
    Molecular Physics, 87, 1117-1157 (1996)
    """

    def __init__(self, temperature: np.float64, timescale: np.float64 = 1e2 * fs_to_au,
                 ndof: int = 3, nchains: int = 3, nys: int = 3, nc: int = 1):
        """Constructor

        :param temperature: thermostat temperature
        :param timescale: thermostat timescale
        :param nchains: number of thermostat chains
        """
        self.temperature = temperature
        self.timescale = timescale
        self.ndof = ndof
        self.nchains = nchains
        self.nys = nys
        self.nc = nc

        assert self.temperature > 0.0
        assert self.timescale > 0.0
        assert self.nchains >= 1
        assert self.nys in (3,5)
        assert self.nc >= 1

        self.nh_position = np.zeros(nchains, dtype=np.float64)
        self.nh_velocity = np.zeros(nchains, dtype=np.float64)
        Q = timescale**2 * boltzmann * temperature
        self.nh_mass = np.ones(nchains, dtype=np.float64) * Q
        self.nh_mass[0] *= ndof

        if nys == 3:
            tmp = 1 / (2 - 2**(1./3))
            self.w = np.array([tmp, 1 - 2*tmp, tmp]) / nc
        elif nys == 5:
            tmp = 1 / (4 - 4**(1./3))
            self.w = np.array([tmp, tmp, 1 - 4*tmp, tmp, tmp]) / nc
        else:
            raise ValueError("nys must be either 3 or 5")

        self.G = np.zeros(nchains)

        print("Nose-Hoover chain thermostat initialized with:")
        print("  Temperature: {:.2f} K".format(temperature))
        print("  Number of chains: {}".format(nchains))
        print("  Timescale: {:.2f} fs".format(timescale / fs_to_au))
        print("  Thermostat mass: {}".format(self.nh_mass))

    def nhc_step(self, velocity, mass, dt: float):
        """
        Move forward one step in the extended system variables
        """
        # convenience definitions
        G = self.G
        V = self.nh_velocity
        X = self.nh_position
        M = self.nchains

        scale = 1.0
        K2 = np.sum(velocity * velocity * mass)
        kt = self.temperature * boltzmann
        nkt = self.temperature * boltzmann * self.ndof

        G[0] = (K2 - nkt) / self.nh_mass[0]

        for _ in range(self.nc):
            for iys in range(self.nys):
                wdt = self.w[iys] * dt
                V[M-1] += G[M-1] * wdt / 4.0

                for kk in range(0, M-1):
                    AA = np.exp(-wdt * V[M - 1 - kk] / 8.0)
                    V[M - 2 - kk] *= AA * AA
                    V[M - 2 - kk] += 0.25 * wdt * G[M - 2 - kk] * AA

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * V[0])
                scale *= AA

                # update the forces
                G[0] = (scale * scale * K2 - nkt) / self.nh_mass[0]

                # update the thermostat positions
                X += 0.5 * V * wdt

                # update the thermostat velocities
                for kk in range(M - 1):
                    AA = np.exp(-0.125 * wdt * V[kk + 1])
                    V[kk] = V[kk] * AA * AA \
                              + 0.25 * wdt * G[kk] * AA
                    G[kk+1] = (self.nh_mass[kk] * V[kk]**2 - kt) / self.nh_mass[kk + 1]
                V[M - 1] += 0.25 * G[M - 1] * wdt

        return scale

    def __call__(self, traj: 'AdiabaticMD', nsteps: int) -> None:
        """Propagate trajectory using Velocity Verlet algorithm with Nose-Hoover thermostat

        :param traj: trajectory object to propagate
        :param nsteps: number of steps to propagate
        """
        dt = traj.dt

        # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
        for _ in range(nsteps):
            vscale = self.nhc_step(traj.velocity, traj.mass, dt)
            acceleration = traj.force(traj.electronics) / traj.mass

            x0 = traj.position
            v1 = traj.velocity * vscale
            v1 += 0.5 * dt * acceleration

            traj.position = x0 + v1 * dt
            traj.last_position = x0

            # calculate electronics at new position
            traj.last_electronics = traj.electronics
            traj.electronics = traj.model.update(traj.position, electronics=traj.electronics)

            acceleration = traj.force(traj.electronics) / traj.mass

            v2p  = v1 + 0.5 * dt * acceleration

            vscale = self.nhc_step(v2p, traj.mass, dt)
            traj.last_velocity = traj.velocity
            traj.velocity = v2p * vscale

            # optionally remove COM motion and total angular momentum
            if traj.remove_com_every > 0 and (traj.nsteps % traj.remove_com_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                vnew = remove_center_of_mass_motion(v, m)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            if traj.remove_angular_momentum_every > 0 and \
                    (traj.nsteps % traj.remove_angular_momentum_every) == 0:
                d = traj.model.dimensionality
                v = traj.velocity.reshape(d)
                m = traj.mass.reshape(d)[:,0]
                x = traj.position.reshape(d)
                vnew = remove_angular_momentum(v, m, x)
                traj.velocity = vnew.reshape(traj.velocity.shape)

            traj.time += dt
            traj.nsteps += 1

class AdiabaticPropagator:
    """Factory class for creating propagator objects"""
    def __new__(cls, model: Any, prop_options: Any = "vv") -> Propagator_:
        """Factory function for creating propagator objects

        :param prop_type: string or dict propagator type
        :param model: model object (may be necessary to set some defaults)

        :return: propagator object
        """
        if isinstance(prop_options, str):
            prop_options = { "type": prop_options.lower() }

        prop_options["ndof"] = model.ndim()

        if isinstance(prop_options, dict):
            prop_type = prop_options.pop("type", "vv")
            if prop_type in ["vv", "velocity verlet"]:
                return VVPropagator(**prop_options)
            if prop_type in ["nh", "nhc", "nose-hoover"]:
                return NoseHooverChainPropagator(**prop_options)
            else:
                raise ValueError(f"Unknown propagator type: {prop_type}")

        raise ValueError(f"Unknown propagator options: {prop_options}")
