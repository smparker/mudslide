#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for OpenMM functionalities"""

import numpy as np
import os
import shutil
import unittest
import pytest

import mudslide
import yaml

import openmm
import openmm.app

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

def h2o5_mm():
    pdb = openmm.app.PDBFile("h2o5.pdb")
    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=openmm.app.NoCutoff,
                             constraints=None,
                             rigidWater=False)
    mm = mudslide.models.OpenMM(pdb, ff, system)
    return mm

@unittest.skipUnless(mudslide.openmm_model.openmm_is_installed(),
                     "OpenMM must be installed")
class TestOpenMM(unittest.TestCase):
    """Test Suite for TMModel class"""

    testname = "openmm_h2o5"

    def setUp(self):
        self.refdir = os.path.join(_refdir, self.testname)
        self.rundir = os.path.join(_checkdir, self.testname)

        clean_directory(self.rundir)
        os.makedirs(self.rundir, exist_ok=True)

        self.origin = os.getcwd()

        os.chdir(self.rundir)
        with os.scandir(self.refdir) as it:
            for fil in it:
                if fil.name.endswith(".input") and fil.is_file():
                    filename = fil.name[:-6]
                    shutil.copy(os.path.join(self.refdir, fil.name), filename)

    def test_energies_forces(self):
        """Test energies and forces for OpenMM model"""
        mm = h2o5_mm()
        mm.compute(mm._position)

        energy_ref = -0.03390827401818114
        forces_ref = np.loadtxt("f0.txt")

        assert np.allclose(mm.energies[0], energy_ref)
        assert np.allclose(mm.force(), forces_ref)

    def test_dynamics(self):
        mm = h2o5_mm()

        masses = mm.mass
        velocities = mudslide.math.boltzmann_velocities(masses, 300.0, seed=1234)
        p = velocities * masses
        KE = 0.5 * np.sum(p**2 / masses)

        assert np.isclose(KE, 0.021375978053325008)

        traj = mudslide.AdiabaticMD(mm, mm._position, p, dt=20, max_steps=10)
        results = traj.simulate()

        xref = np.loadtxt("x.txt")
        pref = np.loadtxt("p.txt")

        assert np.allclose(results[-1]["position"], xref)
        assert np.allclose(results[-1]["momentum"], pref)


    def tearDown(self):
        os.chdir(self.origin)

