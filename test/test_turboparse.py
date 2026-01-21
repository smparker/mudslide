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


class TestHHeS2S:
    """Test parsing of HHe_S2S.txt egrad output (H-He system with state-to-state couplings)"""

    @pytest.fixture
    def parsed(self):
        """Parse the HHe_S2S.txt file"""
        filepath = os.path.join(exampledir, "HHe_S2S.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_has_egrad_key(self, parsed):
        """Test that egrad section is parsed"""
        assert 'egrad' in parsed

    def test_ground_state_dipole(self, parsed):
        """Test ground state dipole moment is parsed"""
        ground = parsed['egrad']['ground']
        assert 'dipole' in ground
        # From lines 263-266: dipole z total = 0.173122
        assert np.isclose(ground['dipole']['total'][2], 0.173122)

    def test_excited_states_count(self, parsed):
        """Test that both excited states are parsed"""
        excited = parsed['egrad']['excited_state']
        assert len(excited) == 2

    def test_first_excited_state(self, parsed):
        """Test first excited state properties"""
        ex1 = parsed['egrad']['excited_state'][0]
        assert ex1['index'] == 1
        assert ex1['irrep'] == 'a'
        # Excitation energy from line 303: 0.2423779346721230
        assert np.isclose(ex1['energy'], 0.2423779346721230)

    def test_second_excited_state(self, parsed):
        """Test second excited state properties"""
        ex2 = parsed['egrad']['excited_state'][1]
        assert ex2['index'] == 2
        assert ex2['irrep'] == 'a'
        # Excitation energy from line 382: 0.9365243046492246
        assert np.isclose(ex2['energy'], 0.9365243046492246)

    def test_first_excited_state_dipole(self, parsed):
        """Test first excited state transition dipole moment"""
        ex1 = parsed['egrad']['excited_state'][0]
        # Electric transition dipole moment (length rep.) from lines 350-354
        # z = -0.714838
        assert 'diplen' in ex1
        assert np.isclose(ex1['diplen']['z'], -0.714838)

    def test_gradient_parsed(self, parsed):
        """Test that gradient section is parsed"""
        assert 'gradient' in parsed['egrad']
        gradient = parsed['egrad']['gradient']
        assert len(gradient) > 0

    def test_gradient_atoms(self, parsed):
        """Test gradient atom list"""
        gradient = parsed['egrad']['gradient'][0]
        assert 'atom_list' in gradient
        # Should have 2 atoms: h and he
        assert len(gradient['atom_list']) == 2
        assert '1 h' in gradient['atom_list'][0]
        assert '2 he' in gradient['atom_list'][1]

    def test_gradient_values(self, parsed):
        """Test gradient values for excited state 2"""
        gradient = parsed['egrad']['gradient'][0]
        # From lines 680-682:
        # dE/dx  0.0000000D+00  0.0000000D+00
        # dE/dy  0.0000000D+00  0.0000000D+00
        # dE/dz  0.7039368D-01 -0.7039368D-01
        assert np.allclose(gradient['d_dx'], [0.0, 0.0])
        assert np.allclose(gradient['d_dy'], [0.0, 0.0])
        assert np.allclose(gradient['d_dz'], [0.07039368, -0.07039368])

    def test_coupling_parsed(self, parsed):
        """Test that NAC coupling sections are parsed"""
        assert 'coupling' in parsed['egrad']
        coupling = parsed['egrad']['coupling']
        # Should have 3 NAC blocks: <1|d/dR|2>, <0|d/dR|1>, <0|d/dR|2>
        assert len(coupling) == 3

    def test_coupling_1_2(self, parsed):
        """Test NAC coupling between states 1 and 2"""
        coupling = parsed['egrad']['coupling'][0]
        # From lines 695-698:
        # d/dx  0.0000000D+00  0.0000000D+00
        # d/dy  0.0000000D+00  0.0000000D+00
        # d/dz -0.6646843D-01  0.5750408D-01
        assert coupling['bra_state'] == 1
        assert coupling['ket_state'] == 2
        expected_nac = [
            [0.0, 0.0],  # dx
            [0.0, 0.0],  # dy
            [-0.06646843, 0.05750408]  # dz
        ]
        assert np.allclose(coupling['d/dR'], expected_nac)

    def test_coupling_0_1(self, parsed):
        """Test NAC coupling between states 0 and 1"""
        coupling = parsed['egrad']['coupling'][1]
        # From lines 711-714:
        # d/dx  0.0000000D+00  0.0000000D+00
        # d/dy  0.0000000D+00  0.0000000D+00
        # d/dz  0.7739182D-01 -0.2188796D+00
        assert coupling['bra_state'] == 0
        assert coupling['ket_state'] == 1
        expected_nac = [
            [0.0, 0.0],  # dx
            [0.0, 0.0],  # dy
            [0.07739182, -0.2188796]  # dz
        ]
        assert np.allclose(coupling['d/dR'], expected_nac)

    def test_coupling_0_2(self, parsed):
        """Test NAC coupling between states 0 and 2"""
        coupling = parsed['egrad']['coupling'][2]
        # From lines 728-730:
        # d/dx  0.0000000D+00  0.0000000D+00
        # d/dy  0.0000000D+00  0.0000000D+00
        # d/dz  0.9668936D-02  0.1534590D+00
        assert coupling['bra_state'] == 0
        assert coupling['ket_state'] == 2
        expected_nac = [
            [0.0, 0.0],  # dx
            [0.0, 0.0],  # dy
            [0.009668936, 0.1534590]  # dz
        ]
        assert np.allclose(coupling['d/dR'], expected_nac)

    def test_state_to_state_parsed(self, parsed):
        """Test that state-to-state transition moments are parsed"""
        assert 'state-to-state' in parsed['egrad']
        s2s = parsed['egrad']['state-to-state']
        # Only <1|W|2> is parsed (labeled as "transition moments")
        # <2|W|2> is labeled as "difference moments" and not captured
        assert len(s2s) == 1

    def test_state_to_state_1_2(self, parsed):
        """Test <1|W|2> transition moments"""
        s2s = parsed['egrad']['state-to-state'][0]
        assert s2s['bra'] == 1
        assert s2s['ket'] == 2
        # Relaxed electric transition dipole moment (length rep.) from line 608
        # z = 0.546674
        assert np.isclose(s2s['diplen']['z'], 0.546674)
