#!/usr/bin/env python

import numpy as np
import pytest

import mudslide

def test_tully_avoided_crossing_cumulative():
    # Set up the model and initial conditions
    simple_model = mudslide.models.TullySimpleAvoidedCrossing()
    x0 = np.array([-5.0])
    p0 = np.array([15.0])
    v0 = p0 / simple_model.mass

    # Initialize and run the trajectory
    traj = mudslide.SurfaceHoppingMD(simple_model, x0, v0, 0, dt=1, max_steps=1000,
                                    hopping_method='cumulative', trace_every=1,
                                    seed_sequence=7943)
    log = traj.simulate()

    # Convert log to list for easier indexing
    log_list = list(log)

    # Test properties at step 600
    step_600 = log_list[600]
    assert step_600["time"] == pytest.approx(600.0)
    assert step_600["hopping"] == pytest.approx(0.0011025058087533923)  # No hopping at this step
    assert step_600["prob_cum"] == pytest.approx(0.04597172379011495)  # Cumulative probability
    assert step_600["zeta"] == pytest.approx(0.0957267487289587)  # Random number for hopping

    # Test properties at step 800
    step_800 = log_list[800]
    assert step_800["time"] == pytest.approx(800.0)
    assert step_800["hopping"] == pytest.approx(0.0031239680338338203)  # No hopping at this step
    assert step_800["prob_cum"] == pytest.approx(0.07770987897120947)  # Cumulative probability
    assert step_800["zeta"] == pytest.approx(0.26871620122784523)  # Random number for hopping

def test_forced_hop():
    # Use smaller coupling (c=0.003) so the minimum adiabatic gap (2c = 0.006)
    # falls below the forced_hop_threshold of 0.008
    simple_model = mudslide.models.TullySimpleAvoidedCrossing(c=0.003)
    x0 = np.array([-5.0])
    p0 = np.array([15.0])
    v0 = p0 / simple_model.mass

    # Start on excited state (state 1) and run until forced hop triggers
    traj = mudslide.SurfaceHoppingMD(simple_model, x0, v0, 1, dt=1, max_steps=700,
                                    forced_hop_threshold=0.008, trace_every=1,
                                    seed_sequence=7943)
    log = traj.simulate()

    # A forced hop to the lowest state should have occurred
    assert len(log.hops) >= 1
    forced = log.hops[0]
    assert forced["hop_from"] == 1
    assert forced["hop_to"] == 0
    assert forced["prob"] == 1.0  # forced hops are recorded with prob=1.0

def test_tully_avoided_crossing_cumulative_integrated():
    # Set up the model and initial conditions
    simple_model = mudslide.models.TullySimpleAvoidedCrossing()
    x0 = np.array([-5.0])
    p0 = np.array([15.0])
    v0 = p0 / simple_model.mass

    # Initialize and run the trajectory
    traj = mudslide.SurfaceHoppingMD(simple_model, x0, v0, 0, dt=1, max_steps=1000,
                                    hopping_method='cumulative_integrated', trace_every=1,
                                    seed_sequence=7943)
    log = traj.simulate()

    # Convert log to list for easier indexing
    log_list = list(log)

    # Test properties at step 600
    step_600 = log_list[600]
    assert step_600["time"] == pytest.approx(600.0)
    assert step_600["hopping"] == pytest.approx(0.0011025058087533923)  # No hopping at this step
    assert step_600["prob_cum"] == pytest.approx(0.047061968340092575)  # Cumulative probability
    assert step_600["zeta"] == pytest.approx(0.10062369515892865)  # Random number for hopping

    # Test properties at step 800
    step_800 = log_list[800]
    assert step_800["time"] == pytest.approx(800.0)
    assert step_800["hopping"] == pytest.approx(0.0031239680338338203)  # No hopping at this step
    assert step_800["prob_cum"] == pytest.approx(0.08089544003192849)  # Cumulative probability
    assert step_800["zeta"] == pytest.approx(0.31295366096108956)  # Random number for hopping
