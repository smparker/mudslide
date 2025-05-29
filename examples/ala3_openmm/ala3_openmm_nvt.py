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
    p = velocities * masses
    KE = 0.5 * np.sum(p**2 / masses)
    print("Initial kinetic energy:", KE)
    print("Initial temperature: ", KE / (0.5 * mm.ndim() * mudslide.constants.boltzmann))

    fs = mudslide.constants.fs_to_au
    traj = mudslide.AdiabaticMD(mm, mm._position, p, propagator={ "type": "nhc", "temperature": 300, "timescale": 10*fs }, dt=1.0*fs, max_steps=10000, remove_com_every=0)
    results = traj.simulate()

    mudslide.io.write_trajectory_xyz(mm, results, 'ala3.xyz', every=10)

    with open('ala3_nvt.txt', 'w') as f:
        results.print_egylog(file=f, T_window=1000)
