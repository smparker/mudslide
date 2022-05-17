#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import unittest
import sys
from pathlib import Path
import mudslide
import yaml

from mudslide.models import TMModel
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""

    def setUp(self):
        self.refdir = os.path.join(_refdir, "tm-c2h4")

        self.rundir = os.path.join(_checkdir, "tm-c2h4")

        clean_directory(self.rundir)
        os.makedirs(self.rundir, exist_ok=True)

        self.origin = os.getcwd()

        os.chdir(self.rundir)
        with os.scandir(self.refdir) as it:
            for fil in it:
                if fil.name.endswith(".input") and fil.is_file():
                    filename = fil.name[:-6]
                    shutil.copy(os.path.join(self.refdir, fil.name), filename)

    def test_get_gs_ex_properties(self):
        """test for gs_ex_properties function"""
        model = TMModel(states = [0,1,2,3], turbomole_dir = ".", expert=True)
        positions = model.X

        mom = [5.583286976987380000, -2.713959745507320000, 0.392059702162967000,
                -0.832994241764031000, -0.600752326053757000, -0.384006560250834000,
                -1.656414687719690000, 1.062437820195600000, -1.786171104341720000,
                -2.969087779972610000, 1.161804203506510000, -0.785009852486148000,
                2.145175145340160000, 0.594918215579156000, 1.075977514428970000,
                -2.269965412856570000,  0.495551832268249000,   1.487150300486560000]

        log = mudslide.YAMLTrace(base_name="TMtrace", location=self.rundir, log_pitch=8)
        traj = mudslide.TrajectorySH(model, positions, mom, 3,tracer=log,  dt = 20, max_time =81, t0 = 1)
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "TMtrace-0.yaml"

        refs = mudslide.load_log(os.path.join(self.refdir, "traj-0.yaml"))

        ref_times = [ 0, 1, 2, 3]
        states = [ 0, 1, 2, 3 ]

        for t in ref_times:
            for s in states:
                self.assertAlmostEqual(refs[t]["electronics"]["hamiltonian"][s][s],results[t]["electronics"]["hamiltonian"][s][s], places = 8)
                np.testing.assert_almost_equal(refs[t]["electronics"]["force"][s],results[t]["electronics"]["force"][s], decimal = 8)

        for t in ref_times:
                np.testing.assert_almost_equal(refs[t]["density_matrix"],results[t]["density_matrix"], decimal = 8)
                np.testing.assert_almost_equal(refs[t]["position"],results[t]["position"], decimal = 8)
                np.testing.assert_almost_equal(refs[t]["momentum"],results[t]["momentum"], decimal = 8)

        for t in ref_times:
            for s1 in states:
                for s2 in range(s1,3):
                    np.testing.assert_almost_equal(refs[t]["electronics"]["derivative_coupling"][s1][s2], results[t]["electronics"]["derivative_coupling"][s1][s2], decimal = 6)

    def tearDown(self):
        os.chdir(self.origin)

if __name__ == '__main__':
    unittest.main()
