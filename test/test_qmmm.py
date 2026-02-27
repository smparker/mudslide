#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for OpenMM functionalities"""

import numpy as np
import os
import shutil
import pytest

import mudslide
import yaml

openmm = pytest.importorskip("openmm")
openmm_app = pytest.importorskip("openmm.app")

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

@pytest.fixture
def h2o5_mm():
    pdb = openmm.app.PDBFile("h2o5.pdb")
    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=openmm.app.NoCutoff,
                             constraints=None,
                             rigidWater=False)
    mm = mudslide.models.OpenMM(pdb, ff, system)
    return mm

@pytest.mark.skipif(not (mudslide.models.openmm_model.openmm_is_installed()
                         and mudslide.models.turbomole_model.turbomole_is_installed_or_prefixed()), reason="Turbomole and OpenMM must be installed")
class TestQMMM:
    """Test Suite for QMMM class"""

    testname = "qmmm_h2o6"

    def setup_class(self):
        self.refdir = os.path.join(_refdir, self.testname)
        self.rundir = os.path.join(_checkdir, self.testname)

        if os.path.isdir(self.rundir):
            shutil.rmtree(self.rundir)
        os.makedirs(self.rundir, exist_ok=True)

        self.origin = os.getcwd()

        os.chdir(self.rundir)
        with os.scandir(self.refdir) as it:
            for fil in it:
                if fil.name.endswith(".input") and fil.is_file():
                    filename = fil.name[:-6]
                    shutil.copy(os.path.join(self.refdir, fil.name), filename)

    def teardown_class(self):
        os.chdir(self.origin)

    def test_raise_on_polarizable(self):
        qm = mudslide.models.TMModel([0])
        pdb_qm = openmm.app.PDBFile('water_qm.pdb')

        modeller = openmm.app.Modeller(pdb_qm.topology, pdb_qm.positions)
        pdb_mm = openmm.app.PDBFile('water_five.pdb')
        modeller.add(pdb_mm.topology, pdb_mm.positions)

        ff = openmm.app.ForceField('amoeba2018.xml')
        system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff,
                constraints=None, rigidWater=False, removeCMMotion=False)
        with pytest.raises(ValueError):
            mm = mudslide.models.OpenMM(modeller, ff, system)

    def test_raise_on_cmmotion(self):
        qm = mudslide.models.TMModel([0])
        pdb_qm = openmm.app.PDBFile('water_qm.pdb')

        modeller = openmm.app.Modeller(pdb_qm.topology, pdb_qm.positions)
        pdb_mm = openmm.app.PDBFile('water_five.pdb')
        modeller.add(pdb_mm.topology, pdb_mm.positions)

        ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff,
                constraints=None, rigidWater=False, removeCMMotion=True)
        mm = mudslide.models.OpenMM(modeller, ff, system)

        with pytest.raises(ValueError) as excinfo:
            qmmm = mudslide.models.QMMM(qm, mm)

        assert "removeCMMotion=False" in str(excinfo.value)

    def test_raise_on_periodic_boundaries(self):
        qm = mudslide.models.TMModel([0])
        pdb_qm = openmm.app.PDBFile('water_qm.pdb')

        modeller = openmm.app.Modeller(pdb_qm.topology, pdb_qm.positions)
        pdb_mm = openmm.app.PDBFile('water_five.pdb')
        modeller.add(pdb_mm.topology, pdb_mm.positions)

        ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        with pytest.raises(ValueError) as excinfo:
            system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
                        constraints=None, rigidWater=False, removeCMMotion=False)
        #mm = mudslide.models.OpenMM(modeller, ff, system)

        #with pytest.raises(ValueError) as excinfo:
        #    qmmm = mudslide.models.QMMM(qm, mm)

    def test_single_point(self):
        qm = mudslide.models.TMModel([0])
        pdb_qm = openmm.app.PDBFile('water_qm.pdb')

        modeller = openmm.app.Modeller(pdb_qm.topology, pdb_qm.positions)
        pdb_mm = openmm.app.PDBFile('water_five.pdb')
        modeller.add(pdb_mm.topology, pdb_mm.positions)

        ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff,
                constraints=None, rigidWater=False, removeCMMotion=False)
        mm = mudslide.models.OpenMM(modeller, ff, system)

        qmmm = mudslide.models.QMMM(qm, mm)

        qmmm.compute(qmmm._position)

        Eref = -76.29937888
        Fref = np.loadtxt('force.txt')

        assert np.isclose(qmmm.hamiltonian[0,0], Eref)
        assert np.allclose(qmmm.force(), Fref, atol=1e-4)
