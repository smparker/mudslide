#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for mudslide"""

import unittest
import sys

from mudslide import poisson_prob_scale
from mudslide.math import boltzmann_velocities
from mudslide.constants import boltzmann
import numpy as np


class TestMath(unittest.TestCase):
    """Test Suite for math functions"""

    def test_poisson_scale(self):
        """Poisson scaling function"""
        args = np.array([0.0, 0.5, 1.0])
        fx = poisson_prob_scale(args)
        refs = np.array([1.0, 0.7869386805747332, 0.6321205588285577])
        for i, x in enumerate(args):
            self.assertAlmostEqual(fx[i], refs[i], places=10)
            self.assertAlmostEqual(poisson_prob_scale(args[i]), refs[i], places=10)


class TestBoltzmannVelocities(unittest.TestCase):
    """Test Suite for boltzmann_velocities function"""

    def setUp(self):
        self.natom = 4
        self.temperature = 300.0
        self.seed = 42
        # masses for 4 atoms, repeated for each DOF
        atom_masses = np.array([16.0, 1.0, 1.0, 12.0])
        self.mass_flat = np.repeat(atom_masses, 3)  # shape (12,)
        self.mass_2d = np.column_stack([atom_masses] * 3)  # shape (4, 3)
        # coordinates for angular momentum removal
        self.coords_flat = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ])
        self.coords_2d = self.coords_flat.reshape((4, 3))

    def test_flat_input_returns_flat_output(self):
        """Flat mass input (ndof,) should return flat velocities (ndof,)"""
        v = boltzmann_velocities(self.mass_flat, self.temperature, seed=self.seed)
        self.assertEqual(v.shape, self.mass_flat.shape)

    def test_2d_input_returns_2d_output(self):
        """2D mass input (natom, 3) should return 2D velocities (natom, 3)"""
        v = boltzmann_velocities(self.mass_2d, self.temperature, seed=self.seed)
        self.assertEqual(v.shape, self.mass_2d.shape)

    def test_flat_and_2d_give_same_velocities(self):
        """Flat and 2D inputs with same seed should produce equivalent velocities"""
        v_flat = boltzmann_velocities(self.mass_flat, self.temperature, seed=self.seed)
        v_2d = boltzmann_velocities(self.mass_2d, self.temperature, seed=self.seed)
        np.testing.assert_allclose(v_flat, v_2d.reshape(-1), atol=1e-12)

    def test_com_removed(self):
        """Center of mass momentum should be zero after removal"""
        v = boltzmann_velocities(self.mass_flat, self.temperature,
                                 remove_translation=True, seed=self.seed)
        v3 = v.reshape((-1, 3))
        atom_masses = self.mass_flat.reshape((-1, 3))[:, 0]
        com_momentum = np.sum(v3 * atom_masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(com_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_removed(self):
        """Angular momentum should be approximately zero after removal"""
        v = boltzmann_velocities(self.mass_flat, self.temperature,
                                 remove_translation=True,
                                 coords=self.coords_flat,
                                 remove_rotation=True,
                                 seed=self.seed)
        v3 = v.reshape((-1, 3))
        atom_masses = self.mass_flat.reshape((-1, 3))[:, 0]
        momentum = v3 * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(self.coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_removed_2d(self):
        """Angular momentum removal should work with 2D mass and coords"""
        v = boltzmann_velocities(self.mass_2d, self.temperature,
                                 remove_translation=True,
                                 coords=self.coords_2d,
                                 remove_rotation=True,
                                 seed=self.seed)
        self.assertEqual(v.shape, self.mass_2d.shape)
        atom_masses = self.mass_2d[:, 0]
        momentum = v * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(self.coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_angular_momentum_auto_enabled_with_coords(self):
        """Rotation removal should be auto-enabled when coords are provided"""
        v = boltzmann_velocities(self.mass_flat, self.temperature,
                                 coords=self.coords_flat, seed=self.seed)
        v3 = v.reshape((-1, 3))
        atom_masses = self.mass_flat.reshape((-1, 3))[:, 0]
        momentum = v3 * atom_masses[:, np.newaxis]
        angular_momentum = np.cross(self.coords_2d, momentum).sum(axis=0)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-10)

    def test_remove_rotation_without_coords_raises(self):
        """Requesting rotation removal without coords should raise ValueError"""
        with self.assertRaises(ValueError):
            boltzmann_velocities(self.mass_flat, self.temperature,
                                 remove_rotation=True, seed=self.seed)


if __name__ == '__main__':
    unittest.main()
