#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os, shutil

import numpy as np
import mudslide

testdir = os.path.dirname(__file__)
refdir = os.path.join(testdir, "ref", "tracer")
checkdir = os.path.join(testdir, "check", "tracer")

class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        # clean out check dir
        if os.path.isdir(checkdir):
            shutil.rmtree(checkdir)

    def test_log_yaml(self):
        model = mudslide.models.TullySimpleAvoidedCrossing()
        log = mudslide.YAMLTrace(base_name="test-traj", location=checkdir, log_pitch=8)
        traj = mudslide.TrajectorySH(model, [-3.0], [10.0], 0, dt=4, tracer=log, max_time=150, zeta_list=[0.2, 0.2, 0.9])
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "test-traj-0.yaml"

        snap_t16 = results[16]

        refs = mudslide.load_log(os.path.join(refdir,"test-traj-0.yaml"))

        ref_t16 = refs[16]

        for prop in ["position", "momentum", "density_matrix"]:
            np.testing.assert_almost_equal(snap_t16[prop], ref_t16[prop], decimal=8)

        for prop in ["potential", "kinetic", "hopping"]:
            self.assertAlmostEqual(snap_t16[prop], ref_t16[prop], places=8)

if __name__ == '__main__':
    unittest.main()
