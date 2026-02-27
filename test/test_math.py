#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for mudslide"""

import pytest
import sys

from mudslide import poisson_prob_scale
from mudslide.exceptions import ConfigurationError
from mudslide.math import boltzmann_velocities
from mudslide.constants import boltzmann
import numpy as np


class TestMath:
    """Test Suite for math functions"""

    def test_poisson_scale(self):
        """Poisson scaling function"""
        args = np.array([0.0, 0.5, 1.0])
        fx = poisson_prob_scale(args)
        refs = np.array([1.0, 0.7869386805747332, 0.6321205588285577])
        for i, x in enumerate(args):
            assert fx[i] == pytest.approx(refs[i], abs=1e-10)
            assert poisson_prob_scale(args[i]) == pytest.approx(refs[i], abs=1e-10)


class TestBoltzmannVelocities:
    """Test Suite for boltzmann_velocities function"""

    def _setup(self):
        natom = 4
        temperature = 300.0
        seed = 42
        # masses for 4 atoms, repeated for each DOF
        atom_masses = np.array([16.0, 1.0, 1.0, 12.0])
        mass_flat = np.repeat(atom_masses, 3)  # shape (12,)
        mass_2d = np.column_stack([atom_masses] * 3)  # shape (4, 3)
        # coordinates for angular momentum removal
        coords_flat = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ])
        coords_2d = coords_flat.reshape((4, 3))
        return (natom, temperature, seed, atom_masses,
                mass_flat, mass_2d, coords_flat, coords_2d)

    def test_flat_input_returns_flat_output(self):
        """Flat mass input (ndof,) should return flat velocities (ndof,)"""
        _, temperature, seed, _, mass_flat, _, _, _ = self._setup()
        v = boltzmann_velocities(mass_flat, temperature, seed=seed)
        assert v.shape == mass_flat.shape

    def test_2d_input_returns_2d_output(self):
        """2D mass input (natom, 3) should return 2D velocities (natom, 3)"""
        _, temperature, seed, _, _, mass_2d, _, _ = self._setup()
        v = boltzmann_velocities(mass_2d, temperature, seed=seed)
        assert v.shape == mass_2d.shape

    def test_flat_and_2d_give_same_velocities(self):
        """Flat and 2D inputs with same seed should produce equivalent velocities"""
        _, temperature, seed, _, mass_flat, mass_2d, _, _ = self._setup()
        v_flat = boltzmann_velocities(mass_flat, temperature, seed=seed)
        v_2d = boltzmann_velocities(mass_2d, temperature, seed=seed)
        np.testing.assert_allclose(v_flat, v_2d.reshape(-1), atol=1e-12)

    def test_com_removed(self):
        """Center of mass momentum should be zero after removal"""
        _, temperature, seed, _, mass_flat, _, _, _ = self._setup()
        v = boltzmann_velocities(mass_flat, temperature,
                                 remove_translation=True, seed=seed)
        v3 = v.reshape((-1, 3))
        atom_masses = mass_flat.reshape((-1, 3))[:, 0]
        com_momentum = np.sum(v3 * atom_masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(com_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_removed(self):
        """Angular momentum should be approximately zero after removal"""
        _, temperature, seed, _, mass_flat, _, coords_flat, coords_2d = self._setup()
        v = boltzmann_velocities(mass_flat, temperature,
                                 remove_translation=True,
                                 coords=coords_flat,
                                 remove_rotation=True,
                                 seed=seed)
        v3 = v.reshape((-1, 3))
        atom_masses = mass_flat.reshape((-1, 3))[:, 0]
        momentum = v3 * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_removed_2d(self):
        """Angular momentum removal should work with 2D mass and coords"""
        _, temperature, seed, _, _, mass_2d, _, coords_2d = self._setup()
        v = boltzmann_velocities(mass_2d, temperature,
                                 remove_translation=True,
                                 coords=coords_2d,
                                 remove_rotation=True,
                                 seed=seed)
        assert v.shape == mass_2d.shape
        atom_masses = mass_2d[:, 0]
        momentum = v * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_auto_enabled_with_coords(self):
        """Rotation removal should be auto-enabled when coords are provided"""
        _, temperature, seed, _, mass_flat, _, coords_flat, coords_2d = self._setup()
        v = boltzmann_velocities(mass_flat, temperature,
                                 coords=coords_flat, seed=seed)
        v3 = v.reshape((-1, 3))
        atom_masses = mass_flat.reshape((-1, 3))[:, 0]
        momentum = v3 * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_remove_rotation_without_coords_raises(self):
        """Requesting rotation removal without coords should raise ValueError"""
        _, temperature, seed, _, mass_flat, _, _, _ = self._setup()
        with pytest.raises(ConfigurationError):
            boltzmann_velocities(mass_flat, temperature,
                                 remove_rotation=True, seed=seed)
