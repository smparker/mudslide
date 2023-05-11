#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for OpenMM functionalities"""

import numpy as np
import os
import shutil
import unittest

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
        pdb = openmm.app.PDBFile("h2o5.pdb")
        forcefield = openmm.app.ForceField('amber14-all.xml',
                                           'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=openmm.app.NoCutoff,
                                         constraints=None,
                                         rigidWater=False)
        mm = mudslide.models.OpenMM(pdb, forcefield, system)

        mm.compute(mm._position)

        energy_ref = -0.03390827401818114
        forces_ref = np.loadtxt("f0.txt")

        assert np.allclose(mm._energy, energy_ref)
        assert np.allclose(mm.force(), forces_ref)


    def tearDown(self):
        os.chdir(self.origin)

