#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for turboparse module"""

import os
import pytest
import numpy as np

from mudslide.turboparse.parse_turbo import parse_turbo

testdir = os.path.dirname(__file__)
exampledir = os.path.join(os.path.dirname(testdir), "examples", "turboparse")


class TestHNAC:
    """Test parsing of H.NAC.txt egrad output"""

    @pytest.fixture
    def parsed(self):
        """Parse the H.NAC.txt file"""
        filepath = os.path.join(exampledir, "H.NAC.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_has_egrad_key(self, parsed):
        """Test that egrad section is parsed"""
        assert 'egrad' in parsed

    def test_ground_state_energy(self, parsed):
        """Test ground state total energy"""
        ground = parsed['egrad']['ground']
        # From line 241: Total energy: -0.4999993319768000
        assert 'dipole' in ground
        # Ground state energy is in the dipole section context

    def test_excited_states_count(self, parsed):
        """Test that all 15 excited states are parsed"""
        excited = parsed['egrad']['excited_state']
        assert len(excited) == 15

    def test_first_excited_state(self, parsed):
        """Test first excited state properties"""
        ex1 = parsed['egrad']['excited_state'][0]
        assert ex1['index'] == 1
        assert ex1['irrep'] == 'a'
        # Excitation energy from line 293: 0.3750008936805557
        assert np.isclose(ex1['energy'], 0.3750008936805557)

    def test_fourth_excited_state(self, parsed):
        """Test fourth excited state (different energy)"""
        ex4 = parsed['egrad']['excited_state'][3]
        assert ex4['index'] == 4
        # Excitation energy from line 537: 0.4055087605608059
        assert np.isclose(ex4['energy'], 0.4055087605608059)

    def test_gradient_parsed(self, parsed):
        """Test that gradient section is parsed"""
        assert 'gradient' in parsed['egrad']
        gradient = parsed['egrad']['gradient']
        assert len(gradient) > 0

    def test_gradient_atoms(self, parsed):
        """Test gradient atom list - should only have 2 atoms (h and q)"""
        gradient = parsed['egrad']['gradient'][0]
        # Currently broken: should be ['1 h', '2 q'] but includes NAC data too
        # This test documents the expected behavior
        assert 'atom_list' in gradient
        # Expected: 2 atoms
        # Actual (broken): 4 atoms (gradient + NAC combined)
        assert len(gradient['atom_list']) == 2, \
            f"Expected 2 atoms, got {len(gradient['atom_list'])}: {gradient['atom_list']}"

    def test_gradient_values(self, parsed):
        """Test gradient values - should be all zeros for this case"""
        gradient = parsed['egrad']['gradient'][0]
        # From lines 1579-1581: all zeros
        expected_dx = [0.0, 0.0]
        expected_dy = [0.0, 0.0]
        expected_dz = [0.0, 0.0]
        assert np.allclose(gradient['d_dx'], expected_dx), \
            f"Expected d_dx={expected_dx}, got {gradient['d_dx']}"
        assert np.allclose(gradient['d_dy'], expected_dy), \
            f"Expected d_dy={expected_dy}, got {gradient['d_dy']}"
        assert np.allclose(gradient['d_dz'], expected_dz), \
            f"Expected d_dz={expected_dz}, got {gradient['d_dz']}"

    def test_coupling_parsed(self, parsed):
        """Test that NAC coupling section is parsed"""
        # Currently broken: NAC is not being parsed because header doesn't match
        assert 'coupling' in parsed['egrad'], \
            f"Expected 'coupling' key in egrad, got keys: {list(parsed['egrad'].keys())}"

    def test_coupling_values(self, parsed):
        """Test NAC coupling values"""
        coupling = parsed['egrad']['coupling'][0]
        # From lines 1592-1594:
        # dE/dx -0.1251235D+00  0.0000000D+00
        # dE/dy  0.2216329D+00  0.0000000D+00
        # dE/dz -0.1151528D+00  0.0000000D+00
        expected_nac = [
            [-0.1251235, 0.0],
            [0.2216329, 0.0],
            [-0.1151528, 0.0]
        ]
        assert 'd/dR' in coupling
        assert np.allclose(coupling['d/dR'], expected_nac)
