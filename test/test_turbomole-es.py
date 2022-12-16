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
import queue
from mudslide.even_sampling import SpawnStack
from mudslide.even_sampling import EvenSamplingTrajectory
from mudslide.tracer import InMemoryTrace, YAMLTrace, TraceManager
from mudslide.batch import TrajGenConst, TrajGenNormal, BatchedTraj

from mudslide.models import TMModel
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(), "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""

    def setUp(self):
        self.refdir = os.path.join(_refdir, "tm-es-c2h4")

        self.rundir = os.path.join(_checkdir, "tm-es-c2h4")

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
        tm_model = TMModel(states=[0, 1, 2, 3], expert=True) 

        # yapf: disable
        mom = [ 5.583286976987380000, -2.713959745507320000,  0.392059702162967000,
               -0.832994241764031000, -0.600752326053757000, -0.384006560250834000,
               -1.656414687719690000,  1.062437820195600000, -1.786171104341720000,
               -2.969087779972610000,  1.161804203506510000, -0.785009852486148000,
                2.145175145340160000,  0.594918215579156000,  1.075977514428970000,
               -2.269965412856570000,  0.495551832268249000,  1.487150300486560000]
        # yapf: enable
        positions = tm_model.X
        mass = tm_model.mass
        q = queue.Queue()
        dt = 20
        max_time=81
        t0 = 1
        sample_stack = SpawnStack.from_quadrature(nsamples=[2, 2, 2])
        sample_stack.sample_stack[0]["zeta"]=0.003
        samples = 1
        nprocs = 1
        trace_type = YAMLTrace
        trace_options = {}
        electronic_integration = 'exp'
        trace_options["location"] = ""
        trace_options["base_name"] = "TMtrace"
        every = 1
        model=tm_model
        traj_gen = TrajGenConst(positions, mom, 3, dt)

        fssh = mudslide.BatchedTraj(model,
                       traj_gen,
                       trajectory_type=EvenSamplingTrajectory,
                       mom=mom,
                       positions=positions,
                       samples=samples,
                       max_time = max_time,
                       nprocs=nprocs,
                       dt=dt,
                       t0=t0,
                       tracemanager=TraceManager(TraceType=trace_type, trace_kwargs=trace_options),
                       trace_every=every,
                       spawn_stack=sample_stack,
                       electronic_integration=electronic_integration)
        results = fssh.compute()
        outcomes = results.outcomes

        refs = mudslide.load_log(os.path.join(self.refdir, "traj-1.yaml"))

        ref_times = [0, 1, 2]
        states = [0, 1, 2, 3]

        for t in ref_times:
            for s in states:
                self.assertAlmostEqual(refs[t]["electronics"]["hamiltonian"][s][s],
                                       results[1][t]["electronics"]["hamiltonian"][s][s],
                                       places=8)

                np.testing.assert_almost_equal(refs[t]["electronics"]["force"][s],
                                               results[1][t]["electronics"]["force"][s],
                                               decimal=8)

        for t in ref_times:
            np.testing.assert_almost_equal(refs[t]["density_matrix"], results[1][t]["density_matrix"], decimal=8)
            np.testing.assert_almost_equal(refs[t]["position"], results[1][t]["position"], decimal=8)
            np.testing.assert_almost_equal(refs[t]["momentum"], results[1][t]["momentum"], decimal=8)

        for t in ref_times:
            for s1 in states:
                for s2 in range(s1, 3):
                    np.testing.assert_almost_equal(refs[t]["electronics"]["derivative_coupling"][s1][s2],
                                                   results[1][t]["electronics"]["derivative_coupling"][s1][s2],
                                                   decimal=6)

    def tearDown(self):
        os.chdir(self.origin)


if __name__ == '__main__':
    unittest.main()
