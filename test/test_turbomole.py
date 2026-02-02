#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import unittest
import pytest
from pathlib import Path
import mudslide
import yaml

from mudslide.models import TMModel, turbomole_is_installed
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def _turbomole_available():
    return turbomole_is_installed() or "MUDSLIDE_TURBOMOLE_PREFIX" in os.environ

pytestmark = pytest.mark.skipif(not _turbomole_available(),
                                     reason="Turbomole must be installed")

def test_raise_on_missing_control():
    """Test if an exception is raised if no control file is found"""
    with pytest.raises(RuntimeError):
        model = TMModel(states=[0],
                        command_prefix=os.environ.get("MUDSLIDE_TURBOMOLE_PREFIX"))

class _TestTM(unittest.TestCase):
    """Base class for TMModel class"""
    testname = None

    def setUp(self):
        self.command_prefix = os.environ.get("MUDSLIDE_TURBOMOLE_PREFIX")

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

    def tearDown(self):
        os.chdir(self.origin)

class TestTMGround(_TestTM):
    """Test ground state calculation"""
    testname = "tm-c2h4-ground"

    def test_ridft_rdgrad(self):
        model = TMModel(states=[0], command_prefix=self.command_prefix)
        xyz = model._position

        model.compute(xyz)

        Eref = -78.40037210973
        Fref = np.loadtxt("force.ref.txt")

        assert np.isclose(model.hamiltonian[0,0], Eref)
        assert np.allclose(model.force(0), Fref)

class TestTMGroundPC(_TestTM):
    """Test ground state calculation with point charges"""
    testname = "tm-c2h4-ground-pc"

    def test_ridft_rdgrad_w_pc(self):
        model = TMModel(states=[0], command_prefix=self.command_prefix)
        xyzpc = np.array([[3.0, 3.0, 3.0],[-3.0, -3.0, -3.0]])
        pcharges = np.array([2, -2])
        model.control.add_point_charges(xyzpc, pcharges)

        xyz = model._position
        model.compute(xyz)

        Eref = -78.63405047062
        Fref = np.loadtxt("force.ref.txt")

        assert np.isclose(model.hamiltonian[0,0], Eref)
        assert np.allclose(model.force(0), Fref)

        xyzpc1, q1, dpc = model.control.read_point_charge_gradients()

        forcepcref = np.loadtxt("forcepc.ref.txt")

        assert np.allclose(xyzpc, xyzpc1)
        assert np.allclose(pcharges, q1)
        assert np.allclose(dpc, forcepcref)

class TestTMExDynamics(_TestTM):
    """Test short excited state dynamics"""
    testname = "tm-c2h4"

    def test_get_gs_ex_properties(self):
        """test for gs_ex_properties function"""
        model = TMModel(states=[0, 1, 2, 3], expert=True,
                        command_prefix=self.command_prefix)
        positions = model._position

        # yapf: disable
        mom = [ 5.583286976987380000, -2.713959745507320000,  0.392059702162967000,
               -0.832994241764031000, -0.600752326053757000, -0.384006560250834000,
               -1.656414687719690000,  1.062437820195600000, -1.786171104341720000,
               -2.969087779972610000,  1.161804203506510000, -0.785009852486148000,
                2.145175145340160000,  0.594918215579156000,  1.075977514428970000,
               -2.269965412856570000,  0.495551832268249000,  1.487150300486560000]
        # yapf: enable

        velocities = np.array(mom) / model.mass

        log = mudslide.YAMLTrace(base_name="TMtrace", location=self.rundir, log_pitch=8)
        traj = mudslide.SurfaceHoppingMD(model, positions, velocities, 3, tracer=log, dt=20, max_time=81, t0=1,
                                     seed_sequence=57892,
                                     hopping_method="instantaneous")
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "TMtrace-0.yaml"

        refs = mudslide.load_log(os.path.join(self.refdir, "traj-0.yaml"))

        ref_times = [0, 1, 2, 3]
        states = [0, 1, 2, 3]

        for t in ref_times:
            for s in states:
                np.testing.assert_almost_equal(refs[t]["electronics"]["hamiltonian"][s][s],
                                       results[t]["electronics"]["hamiltonian"][s][s],
                                       decimal=8)
                np.testing.assert_almost_equal(refs[t]["electronics"]["force"][s],
                                               results[t]["electronics"]["force"][s],
                                               decimal=8)

        for t in ref_times:
            np.testing.assert_almost_equal(refs[t]["density_matrix"], results[t]["density_matrix"], decimal=8)
            np.testing.assert_almost_equal(refs[t]["position"], results[t]["position"], decimal=8)
            np.testing.assert_almost_equal(refs[t]["velocity"], results[t]["velocity"], decimal=8)

        for t in ref_times:
            for s1 in states:
                for s2 in range(s1, 3):
                    np.testing.assert_almost_equal(refs[t]["electronics"]["derivative_coupling"][s1][s2],
                                                   results[t]["electronics"]["derivative_coupling"][s1][s2],
                                                   decimal=6)

