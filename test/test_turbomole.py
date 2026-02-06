#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import pytest
from pathlib import Path
import mudslide
import yaml

from mudslide.models import TMModel, turbomole_is_installed
from mudslide.config import get_config
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")


def _turbomole_available():
    return (turbomole_is_installed() or
            "MUDSLIDE_TURBOMOLE_PREFIX" in os.environ or
            get_config("turbomole.command_prefix") is not None)


pytestmark = pytest.mark.skipif(not _turbomole_available(),
                                reason="Turbomole must be installed")


def test_raise_on_missing_control():
    """Test if an exception is raised if no control file is found"""
    with pytest.raises(RuntimeError):
        model = TMModel(states=[0])


@pytest.fixture
def tm_setup(request, tmp_path):
    """Set up a Turbomole test directory from reference inputs."""
    testname = request.param
    refdir = os.path.join(_refdir, testname)
    rundir = os.path.join(_checkdir, testname)

    if os.path.isdir(rundir):
        shutil.rmtree(rundir)
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


@pytest.mark.parametrize("tm_setup", ["tm-c2h4-ground"], indirect=True)
def test_ridft_rdgrad(tm_setup):
    """Test ground state calculation"""
    model = TMModel(states=[0])
    xyz = model._position

    model.compute(xyz)

    Eref = -78.40037210973
    Fref = np.loadtxt("force.ref.txt")

    assert np.isclose(model.hamiltonian[0, 0], Eref)
    assert np.allclose(model.force(0), Fref)


@pytest.mark.parametrize("tm_setup", ["tm-c2h4-ground-pc"], indirect=True)
def test_ridft_rdgrad_w_pc(tm_setup):
    """Test ground state calculation with point charges"""
    model = TMModel(states=[0])
    xyzpc = np.array([[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]])
    pcharges = np.array([2, -2])
    model.control.add_point_charges(xyzpc, pcharges)

    xyz = model._position
    model.compute(xyz)

    Eref = -78.63405047062
    Fref = np.loadtxt("force.ref.txt")

    assert np.isclose(model.hamiltonian[0, 0], Eref)
    assert np.allclose(model.force(0), Fref)

    xyzpc1, q1, dpc = model.control.read_point_charge_gradients()

    forcepcref = np.loadtxt("forcepc.ref.txt")

    assert np.allclose(xyzpc, xyzpc1)
    assert np.allclose(pcharges, q1)
    assert np.allclose(dpc, forcepcref)


@pytest.mark.parametrize("tm_setup", ["tm-c2h4"], indirect=True)
def test_get_gs_ex_properties(tm_setup):
    """test for gs_ex_properties function"""
    refdir = tm_setup["refdir"]
    rundir = tm_setup["rundir"]

    model = TMModel(states=[0, 1, 2, 3], expert=True)
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

    log = mudslide.YAMLTrace(base_name="TMtrace",
                             location=rundir,
                             log_pitch=8)
    traj = mudslide.SurfaceHoppingMD(model,
                                     positions,
                                     velocities,
                                     3,
                                     tracer=log,
                                     dt=20,
                                     max_time=81,
                                     t0=1,
                                     seed_sequence=57892,
                                     hopping_method="instantaneous")
    results = traj.simulate()

    main_log = results.main_log

    assert main_log == "TMtrace-0.yaml"

    refs = mudslide.load_log(os.path.join(refdir, "traj-0.yaml"))

    ref_times = [0, 1, 2, 3]
    states = [0, 1, 2, 3]

    for t in ref_times:
        available = results[t]["electronics"]["forces_available"]
        act_ham = results[t]["electronics"]["hamiltonian"]
        ref_ham = refs[t]["electronics"]["hamiltonian"]
        act_force = results[t]["electronics"]["force"]
        ref_force = refs[t]["electronics"]["force"]
        for s in states:
            np.testing.assert_almost_equal(act_ham[s][s],
                                           ref_ham[s][s],
                                           decimal=8)
            if available[s]:
                np.testing.assert_almost_equal(act_force[s],
                                               ref_force[s],
                                               decimal=8)

    for t in ref_times:
        act = results[t]
        ref = refs[t]
        np.testing.assert_almost_equal(act["density_matrix"],
                                       ref["density_matrix"],
                                       decimal=8)
        np.testing.assert_almost_equal(act["position"],
                                       ref["position"],
                                       decimal=8)
        np.testing.assert_almost_equal(act["velocity"],
                                       ref["velocity"],
                                       decimal=8)

    for t in ref_times:
        act_tau = results[t]["electronics"]["derivative_coupling"]
        ref_tau = refs[t]["electronics"]["derivative_coupling"]
        for s1 in states:
            for s2 in range(s1, 3):
                np.testing.assert_almost_equal(act_tau[s1][s2],
                                               ref_tau[s1][s2],
                                               decimal=6)
