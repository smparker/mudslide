# -*- coding: utf-8 -*-
"""Test simple short trajectories for adiabatic MD"""

import json

import numpy as np
import mudslide

water_json = """
{"x0": [0.0, 0.0, -0.12178983933899, 1.41713420892173, 0.0, 0.96657854674257, -1.41713420892173, 0.0, 0.96657854674257], "E0": 0.0, "H0": [[0.783125763, 0.0, 5e-10, -0.3915628813, 0.0, -0.3007228667, -0.3915628817, 0.0, 0.3007228662], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5e-10, 0.0, 0.482147457, -0.245482284, 0.0, -0.2410737287, 0.2454822835, 0.0, -0.2410737283], [-0.3915628813, 0.0, -0.245482284, 0.4189939059, 0.0, 0.2731025751, -0.0274310246, 0.0, -0.0276202911], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.3007228667, 0.0, -0.2410737287, 0.2731025751, 0.0, 0.2360154334, 0.0276202915, 0.0, 0.0050582953], [-0.3915628817, 0.0, 0.2454822835, -0.0274310246, 0.0, 0.0276202915, 0.4189939063, 0.0, -0.273102575], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3007228662, 0.0, -0.2410737283, -0.0276202911, 0.0, 0.0050582953, -0.273102575, 0.0, 0.236015433]], "mass": [29156.945697766205, 29156.945697766205, 29156.945697766205, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108], "atom_types": ["O", "H", "H"]}"""

water_model = mudslide.models.HarmonicModel.from_dict(json.loads(water_json))


def test_zero_temperature_dynamics():
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=10)
    results = traj.simulate()

    assert np.allclose(x, results[0]["position"])
    assert np.allclose(v, results[0]["velocity"])
    assert np.isclose(water_model.E0, results[0]["potential"])
    assert np.isclose(0.0, results[0]["kinetic"])

    assert np.allclose(x, results[9]["position"])
    assert np.allclose(v, results[9]["velocity"])
    assert np.isclose(water_model.E0, results[9]["potential"])
    assert np.isclose(0.0, results[9]["kinetic"])


def test_1000K_from_equilibrium():
    x = np.array(water_model.x0)

    masses = water_model.mass
    velocities = mudslide.math.boltzmann_velocities(masses,
                                                    temperature=1000.0,
                                                    remove_translation=False,
                                                    seed=1234)
    KE = 0.5 * np.sum(velocities**2 * masses)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                velocities,
                                dt=20,
                                max_steps=10,
                                remove_com_every=0,
                                remove_angular_momentum_every=0)
    results = traj.simulate()

    assert np.allclose(x, results[0]["position"])
    assert np.allclose(velocities, results[0]["velocity"])
    assert np.isclose(water_model.E0, results[0]["potential"])
    assert np.isclose(KE, results[0]["kinetic"])

    x9 = np.array([
        -0.04686147, 0.0026712, -0.08034362, 1.15101179, 0.14339456, 1.24457676,
        -1.68818976, 0.15696283, 0.72781869
    ])
    v9 = np.array([
        -5.06144019, 0.43268942, 8.0867855, -5.16979919, 1.46354279, 1.17949342,
        -2.84217795, 1.60202598, -2.15221203
    ]) / masses
    V9 = 0.0003035421477246298
    KE9 = 0.013957193155584462

    assert np.allclose(x9, results[9]["position"])
    assert np.allclose(v9, results[9]["velocity"])
    assert np.isclose(V9, results[9]["potential"])
    assert np.isclose(KE9, results[9]["kinetic"])


def test_simulate_early_return():
    """simulate returns immediately when max_steps=0"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=0)
    results = traj.simulate()
    assert len(results) == 0


def test_update_weight():
    """update_weight sets weight on trajectory and tracer"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=5)
    traj.update_weight(np.float64(0.5))
    assert traj.weight == 0.5
    assert traj.tracer.weight == 0.5
    assert not traj.force_quit


def test_update_weight_zero_triggers_force_quit():
    """update_weight with 0 sets force_quit"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=5)
    traj.update_weight(np.float64(0.0))
    assert traj.force_quit
    assert not traj.continue_simulating()


def test_random():
    """random returns value in [0, 1)"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=5,
                                seed_sequence=42)
    r = traj.random()
    assert 0.0 <= r < 1.0


def test_duration_with_bounds():
    """duration_initialize sets box_bounds from bounds option"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    lo = [-5.0] * 9
    hi = [5.0] * 9
    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=5,
                                bounds=[lo, hi])
    assert traj.duration["box_bounds"] is not None
    assert traj.duration["box_bounds"][0].shape == (9,)


def test_currently_interacting_with_bounds():
    """currently_interacting returns True when inside bounds"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    lo = [-5.0] * 9
    hi = [5.0] * 9
    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=5,
                                bounds=[lo, hi])
    assert traj.currently_interacting()


def test_currently_interacting_outside_bounds():
    """currently_interacting returns False when outside bounds"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    lo = [100.0] * 9
    hi = [200.0] * 9
    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=5,
                                bounds=[lo, hi])
    assert not traj.currently_interacting()


def test_currently_interacting_no_bounds():
    """currently_interacting returns False without bounds"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=5)
    assert not traj.currently_interacting()


def test_continue_simulating_max_time():
    """continue_simulating stops at max_time"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=100000,
                                max_time=50)
    results = traj.simulate()
    assert results[-1]["time"] <= 60  # at most one step past max_time


def test_continue_simulating_with_box_bounds():
    """simulate with bounds: enters box, then exits and stops"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=1000.0,
                                           remove_translation=False,
                                           seed=42)
    # tight bounds so trajectory exits quickly
    lo = [-2.0] * 9
    hi = [2.0] * 9
    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=20,
                                max_steps=100,
                                bounds=[lo, hi])
    results = traj.simulate()
    # should terminate before max_steps (exited box)
    assert len(results) > 0


def test_force_method():
    """force returns the ground state force"""
    x = np.array(water_model.x0)
    v = np.zeros_like(x)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=5)
    traj.electronics = water_model.update(x)

    f = traj.force()
    assert f.shape == (9,)
    # at equilibrium, forces should be near zero
    assert np.allclose(f, 0.0, atol=1e-8)


def test_mode_kinetic_energy():
    """mode_kinetic_energy computes KE along a direction"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model, x, v, dt=20, max_steps=5)

    direction = np.zeros(9)
    direction[0] = 1.0
    ke_mode = traj.mode_kinetic_energy(direction)
    assert ke_mode >= 0.0

    # total KE >= any single-mode KE
    total_ke = traj.kinetic_energy()
    assert total_ke >= ke_mode
