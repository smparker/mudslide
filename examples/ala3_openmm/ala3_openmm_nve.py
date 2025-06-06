#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for OpenMM functionalities"""

import numpy as np
import os
import shutil

import mudslide
import yaml

import openmm
import openmm.app

if __name__ == "__main__":
    pdb = openmm.app.PDBFile("ala3.pdb")
    ff = openmm.app.ForceField('amber14-all.xml')
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=openmm.app.NoCutoff,
                             constraints=None,
                             rigidWater=False)
    mm = mudslide.models.OpenMM(pdb, ff, system)

    mm.compute(mm._position)

    masses = mm.mass
    velocities = mudslide.math.boltzmann_velocities(masses, 300.0, seed=1234)
    KE = 0.5 * np.sum(velocities**2 * masses)

    traj = mudslide.AdiabaticMD(mm, mm._position, velocities, dt=40, max_steps=1000)
    results = traj.simulate()

    mudslide.io.write_trajectory_xyz(mm, results, 'ala3.xyz')

