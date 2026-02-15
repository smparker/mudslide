# -*- coding: utf-8 -*-
"""Tests for adiabatic_propagator.py"""

import json

import numpy as np
import pytest

import mudslide
from mudslide.exceptions import ConfigurationError
from mudslide.adiabatic_propagator import (VVPropagator,
                                           NoseHooverChainPropagator,
                                           AdiabaticPropagator)

water_json = """
{"x0": [0.0, 0.0, -0.12178983933899, 1.41713420892173, 0.0, 0.96657854674257, -1.41713420892173, 0.0, 0.96657854674257], "E0": 0.0, "H0": [[0.783125763, 0.0, 5e-10, -0.3915628813, 0.0, -0.3007228667, -0.3915628817, 0.0, 0.3007228662], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5e-10, 0.0, 0.482147457, -0.245482284, 0.0, -0.2410737287, 0.2454822835, 0.0, -0.2410737283], [-0.3915628813, 0.0, -0.245482284, 0.4189939059, 0.0, 0.2731025751, -0.0274310246, 0.0, -0.0276202911], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.3007228667, 0.0, -0.2410737287, 0.2731025751, 0.0, 0.2360154334, 0.0276202915, 0.0, 0.0050582953], [-0.3915628817, 0.0, 0.2454822835, -0.0274310246, 0.0, 0.0276202915, 0.4189939063, 0.0, -0.273102575], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3007228662, 0.0, -0.2410737283, -0.0276202911, 0.0, 0.0050582953, -0.273102575, 0.0, 0.236015433]], "mass": [29156.945697766205, 29156.945697766205, 29156.945697766205, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108], "atom_types": ["O", "H", "H"]}"""

water_model = mudslide.models.HarmonicModel.from_dict(json.loads(water_json))


def test_vv_propagator_with_com_removal():
    """VV propagation with center-of-mass removal exercised"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=10,
                                max_steps=3,
                                remove_com_every=1)
    results = traj.simulate()

    assert len(results) > 0
    assert np.isfinite(results[-1]["energy"])


def test_vv_propagator_with_angular_momentum_removal():
    """VV propagation with angular momentum removal exercised"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=10,
                                max_steps=3,
                                remove_com_every=1,
                                remove_angular_momentum_every=1)
    results = traj.simulate()
    assert len(results) > 0
    assert np.isfinite(results[-1]["energy"])


def test_nhc_constructor_nys3():
    """NHC constructor with nys=3"""
    nhc = NoseHooverChainPropagator(temperature=300.0, ndof=9)

    assert nhc.temperature == 300.0
    assert nhc.ndof == 9
    assert nhc.nchains == 3
    assert nhc.nys == 3
    assert nhc.nc == 1
    assert len(nhc.w) == 3
    assert nhc.nh_position.shape == (3,)
    assert nhc.nh_velocity.shape == (3,)
    assert nhc.nh_mass.shape == (3,)
    assert nhc.G.shape == (3,)


def test_nhc_constructor_nys5():
    """NHC constructor with nys=5"""
    nhc = NoseHooverChainPropagator(temperature=300.0, ndof=9, nys=5)
    assert len(nhc.w) == 5


def test_nhc_constructor_invalid_nys():
    """NHC raises AssertionError on invalid nys"""
    with pytest.raises(AssertionError):
        NoseHooverChainPropagator(temperature=300.0, ndof=9, nys=4)


def test_nhc_step():
    """NHC step returns a positive scale factor"""
    nhc = NoseHooverChainPropagator(temperature=300.0, ndof=9)

    rng = np.random.default_rng(42)
    velocity = rng.normal(0, 0.001, size=9)
    mass = np.array(water_model.mass)

    scale = nhc.nhc_step(velocity, mass, dt=10.0)
    assert isinstance(scale, float)
    assert scale > 0.0


def test_nhc_propagator_trajectory():
    """NHC propagator runs a full trajectory"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=10,
                                max_steps=3,
                                propagator={
                                    "type": "nhc",
                                    "temperature": 300.0
                                })
    results = traj.simulate()

    assert len(results) > 0
    assert np.isfinite(results[-1]["energy"])


def test_nhc_propagator_with_com_and_angular_momentum_removal():
    """NHC propagator with COM and angular momentum removal"""
    x = np.array(water_model.x0)
    v = mudslide.math.boltzmann_velocities(water_model.mass,
                                           temperature=300.0,
                                           remove_translation=False,
                                           seed=42)

    traj = mudslide.AdiabaticMD(water_model,
                                x,
                                v,
                                dt=10,
                                max_steps=3,
                                propagator={
                                    "type": "nhc",
                                    "temperature": 300.0
                                },
                                remove_com_every=1,
                                remove_angular_momentum_every=1)
    results = traj.simulate()

    assert len(results) > 0
    assert np.isfinite(results[-1]["energy"])


def test_factory_vv_from_string():
    """Factory creates VVPropagator from string"""
    prop = AdiabaticPropagator(water_model, "vv")
    assert isinstance(prop, VVPropagator)


def test_factory_vv_from_dict():
    """Factory creates VVPropagator from dict"""
    prop = AdiabaticPropagator(water_model, {"type": "velocity verlet"})
    assert isinstance(prop, VVPropagator)


def test_factory_nhc_from_dict():
    """Factory creates NHC propagator from dict"""
    prop = AdiabaticPropagator(water_model, {
        "type": "nhc",
        "temperature": 300.0
    })
    assert isinstance(prop, NoseHooverChainPropagator)


def test_factory_unknown_type_raises():
    """Factory raises ValueError for unknown propagator type"""
    with pytest.raises(ConfigurationError, match="Unknown propagator type"):
        AdiabaticPropagator(water_model, {"type": "unknown"})
