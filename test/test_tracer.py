#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os, shutil

import numpy as np
import mudslide

testdir = os.path.dirname(__file__)

_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")


def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)


class TrajectoryTest(unittest.TestCase):

    def test_log_yaml(self):
        refdir = os.path.join(_refdir, "tracer")
        rundir = os.path.join(_checkdir, "tracer")
        clean_directory(rundir)

        model = mudslide.models.TullySimpleAvoidedCrossing()
        log = mudslide.YAMLTrace(base_name="test-traj", location=rundir, log_pitch=8)
        traj = mudslide.TrajectorySH(model, [-3.0], [10.0],
                                     0,
                                     dt=4,
                                     tracer=log,
                                     max_time=80,
                                     zeta_list=[0.2, 0.2, 0.9])
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "test-traj-0.yaml"

        snap_t16 = results[16]

        refs = mudslide.load_log(os.path.join(refdir, "test-traj-0.yaml"))

        ref_t16 = refs[16]

        for prop in ["position", "momentum", "density_matrix"]:
            np.testing.assert_almost_equal(snap_t16[prop], ref_t16[prop], decimal=8)

        for prop in ["potential", "kinetic", "hopping"]:
            self.assertAlmostEqual(snap_t16[prop], ref_t16[prop], places=8)

    def test_restart_from_trace(self):
        refdir = os.path.join(_refdir, "tracer")
        rundir = os.path.join(_checkdir, "trace_restart")
        clean_directory(rundir)

        model = mudslide.models.TullySimpleAvoidedCrossing()
        log = mudslide.YAMLTrace(base_name="test-traj", location=rundir, log_pitch=8)
        traj = mudslide.TrajectorySH(model, [-3.0], [10.0],
                                     0,
                                     dt=4,
                                     tracer=log,
                                     max_time=40,
                                     zeta_list=[0.2, 0.2, 0.9])
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "test-traj-0.yaml"

        yaml_trace = mudslide.load_log(os.path.join(results.location, main_log))
        traj2 = mudslide.TrajectorySH.restart(model, yaml_trace, max_time=80)
        results2 = traj2.simulate()
        snap_t16 = results2[16]

        refs = mudslide.load_log(os.path.join(refdir, "test-traj-0.yaml"))
        ref_t16 = refs[16]

        for prop in ["position", "momentum", "density_matrix"]:
            with self.subTest(property=prop):
                np.testing.assert_almost_equal(snap_t16[prop], ref_t16[prop], decimal=8)

        for prop in ["potential", "kinetic", "hopping"]:
            with self.subTest(property=prop):
                self.assertAlmostEqual(snap_t16[prop], ref_t16[prop], places=8)


if __name__ == '__main__':
    unittest.main()
