#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for Turbomole_class with Even Sampling"""

import numpy as np
import os
import shutil
import pytest
from pathlib import Path
import mudslide
import yaml
import queue
from mudslide.even_sampling import SpawnStack
from mudslide.even_sampling import EvenSamplingTrajectory
from mudslide.tracer import TraceManager
from mudslide.batch import TrajGenConst, TrajGenNormal, BatchedTraj

from mudslide.models import TMModel, turbomole_is_installed
from mudslide.config import get_config

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

def _turbomole_available():
    return (turbomole_is_installed()
            or "MUDSLIDE_TURBOMOLE_PREFIX" in os.environ
            or get_config("turbomole.command_prefix") is not None)

pytestmark = pytest.mark.skipif(not _turbomole_available(),
                                reason="Turbomole must be installed")


@pytest.fixture
def tm_es_setup():
    """Set up turbomole even-sampling test directory."""
    refdir = os.path.join(_refdir, "tm-es-c2h4")
    rundir = os.path.join(_checkdir, "tm-es-c2h4")

    clean_directory(rundir)
    os.makedirs(rundir, exist_ok=True)

    origin = os.getcwd()
    os.chdir(rundir)
    with os.scandir(refdir) as it:
        for fil in it:
            if fil.name.endswith(".input") and fil.is_file():
                filename = fil.name[:-6]
                shutil.copy(os.path.join(refdir, fil.name), filename)

    yield {"refdir": refdir, "rundir": rundir}

    os.chdir(origin)


def test_get_gs_ex_properties(tm_es_setup):
    """test for gs_ex_properties function"""
    refdir = tm_es_setup["refdir"]

    tm_model = TMModel(states=[0, 1, 2, 3], expert=True)

    # yapf: disable
    mom = [ 5.583286976987380000, -2.713959745507320000,  0.392059702162967000,
           -0.832994241764031000, -0.600752326053757000, -0.384006560250834000,
           -1.656414687719690000,  1.062437820195600000, -1.786171104341720000,
           -2.969087779972610000,  1.161804203506510000, -0.785009852486148000,
            2.145175145340160000,  0.594918215579156000,  1.075977514428970000,
           -2.269965412856570000,  0.495551832268249000,  1.487150300486560000]
    # yapf: enable
    velocities = mom / tm_model.mass
    positions = tm_model._position
    dt = 20
    max_time = 41
    t0 = 1
    sample_stack = SpawnStack.from_quadrature(nsamples=[2, 2, 2])
    sample_stack.sample_stack[0]["zeta"] = 0.003
    samples = 1
    nprocs = 1
    trace_type = "yaml"
    electronic_integration = 'exp'
    every = 1
    model = tm_model
    traj_gen = TrajGenConst(positions, velocities, 3, dt)

    fssh = mudslide.BatchedTraj(model,
                   traj_gen,
                   trajectory_type=EvenSamplingTrajectory,
                   samples=samples,
                   max_time=max_time,
                   nprocs=nprocs,
                   dt=dt,
                   t0=t0,
                   tracemanager=TraceManager(trace_type=trace_type),
                   trace_every=every,
                   spawn_stack=sample_stack,
                   electronic_integration=electronic_integration)
    results = fssh.compute()
    outcomes = results.outcomes

    refs = mudslide.load_log(os.path.join(refdir, "traj-1.yaml"))

    ref_times = [0, 1, 2]
    states = [0, 1, 2, 3]

    for t in ref_times:
        available = results[1][t]["electronics"]["forces_available"]
        for s in states:
            assert results[1][t]["electronics"]["hamiltonian"][s][s] == pytest.approx(
                refs[t]["electronics"]["hamiltonian"][s][s], abs=1e-8)

            if available[s]:
                np.testing.assert_almost_equal(results[1][t]["electronics"]["force"][s],
                                               refs[t]["electronics"]["force"][s],
                                               decimal=8)

    for t in ref_times:
        np.testing.assert_almost_equal(results[1][t]["density_matrix"], refs[t]["density_matrix"],  decimal=8)
        np.testing.assert_almost_equal(results[1][t]["position"], refs[t]["position"], decimal=8)
        np.testing.assert_almost_equal(results[1][t]["velocity"], refs[t]["velocity"], decimal=8)

    for t in ref_times:
        for s1 in states:
            for s2 in range(s1, 3):
                np.testing.assert_almost_equal(results[1][t]["electronics"]["derivative_coupling"][s1][s2],
                                               refs[t]["electronics"]["derivative_coupling"][s1][s2],
                                               decimal=6)


def test_ridft_convergence_failure(tm_es_setup):
    """Test that non-convergence of ridft raises RuntimeError"""
    tm_model = TMModel(states=[0, 1, 2, 3], expert=True)
    tm_model.control.adg("scfiterlimit", 2)
    # perturb geometry so the converged MOs are no longer a good guess
    positions = tm_model._position.copy()
    positions += 0.5 * np.random.default_rng(42).standard_normal(positions.shape)
    with pytest.raises(RuntimeError):
        tm_model.compute(positions)


def test_egrad_convergence_failure(tm_es_setup):
    """Test that non-convergence of egrad raises RuntimeError"""
    tm_model = TMModel(states=[0, 1, 2, 3], expert=True)
    tm_model.control.adg("escfiterlimit", 2)
    positions = tm_model._position.copy()
    with pytest.raises(RuntimeError):
        tm_model.compute(positions)
