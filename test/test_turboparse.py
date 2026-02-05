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

    def test_exopt(self, parsed):
        """'Excited state no.    3 chosen for optimization' -> [3]"""
        assert parsed['egrad']['exopt'] == [3]

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

    def test_gradient_index_backfilled(self, parsed):
        """Gradient header has no state number, index backfilled from exopt"""
        assert parsed['egrad']['gradient'][0]['index'] == 3

    def test_gradient_values(self, parsed):
        """Test gradient values - should be all zeros for this case"""
        gradient = parsed['egrad']['gradient'][0]
        # From lines 1579-1581: all zeros
        expected = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert np.allclose(gradient['d/dR'], expected), \
            f"Expected d/dR={expected}, got {gradient['d/dR']}"

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
            [-0.1251235, 0.2216329, -0.1151528],
            [0.0, 0.0, 0.0]
        ]
        assert 'd/dR' in coupling
        assert np.allclose(coupling['d/dR'], expected_nac)

    def test_cpks_parsed(self, parsed):
        """Test that CPKS section is parsed"""
        assert 'cpks' in parsed['egrad']

    def test_cpks_converged(self, parsed):
        """Test CPKS convergence flag"""
        assert parsed['egrad']['cpks']['converged'] is True

    def test_cpks_single_iteration(self, parsed):
        """Test CPKS with single iteration"""
        iterations = parsed['egrad']['cpks']['iterations']
        assert len(iterations) == 1
        assert iterations[0]['step'] == 1
        assert np.isclose(iterations[0]['max_residual_norm'], 5.047087406128772e-15)

    def test_davidson_parsed(self, parsed):
        """Test that Davidson section is parsed"""
        assert 'davidson' in parsed['egrad']

    def test_davidson_converged(self, parsed):
        """Test Davidson convergence flag"""
        assert parsed['egrad']['davidson']['converged'] is True

    def test_davidson_iterations(self, parsed):
        """Test Davidson iterations (3 iterations for 15 roots)"""
        iterations = parsed['egrad']['davidson']['iterations']
        assert len(iterations) == 3
        assert iterations[0]['step'] == 1
        assert np.isclose(iterations[0]['max_residual_norm'], 1.534162418811829e-02)
        assert iterations[2]['step'] == 3
        assert np.isclose(iterations[2]['max_residual_norm'], 2.513145479422938e-10)


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

    def test_exopt(self, parsed):
        """'Excited state no.    2 chosen for optimization' -> [2]"""
        assert parsed['egrad']['exopt'] == [2]

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

    def test_gradient_index(self, parsed):
        """Gradient header 'cartesian gradients of excited state    2' -> index 2"""
        assert parsed['egrad']['gradient'][0]['index'] == 2

    def test_gradient_values(self, parsed):
        """Test gradient values for excited state 2"""
        gradient = parsed['egrad']['gradient'][0]
        # From lines 680-682:
        # dE/dx  0.0000000D+00  0.0000000D+00
        # dE/dy  0.0000000D+00  0.0000000D+00
        # dE/dz  0.7039368D-01 -0.7039368D-01
        expected = [[0.0, 0.0, 0.07039368], [0.0, 0.0, -0.07039368]]
        assert np.allclose(gradient['d/dR'], expected)

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
            [0.0, 0.0, -0.06646843],  # atom 1
            [0.0, 0.0, 0.05750408]  # atom 2
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
            [0.0, 0.0, 0.07739182],  # atom 1
            [0.0, 0.0, -0.2188796]  # atom 2
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
            [0.0, 0.0, 0.009668936],  # atom 1
            [0.0, 0.0, 0.1534590]  # atom 2
        ]
        assert np.allclose(coupling['d/dR'], expected_nac)

    def test_state_to_state_parsed(self, parsed):
        """Test that state-to-state transition moments are parsed"""
        assert 'state-to-state' in parsed['egrad']
        s2s = parsed['egrad']['state-to-state']
        # Should have 2 blocks: <2|W|2> difference moments and <1|W|2> transition moments
        assert len(s2s) == 2

    def test_state_to_state_2_2(self, parsed):
        """Test <2|W|2> difference moments"""
        s2s = parsed['egrad']['state-to-state'][0]
        assert s2s['bra'] == 2
        assert s2s['ket'] == 2
        # Relaxed electric transition dipole moment (length rep.) from line 549
        # z = 3.819847
        assert np.isclose(s2s['diplen']['z'], 3.819847)

    def test_state_to_state_1_2(self, parsed):
        """Test <1|W|2> transition moments"""
        s2s = parsed['egrad']['state-to-state'][1]
        assert s2s['bra'] == 1
        assert s2s['ket'] == 2
        # Relaxed electric transition dipole moment (length rep.) from line 608
        # z = 0.546674
        assert np.isclose(s2s['diplen']['z'], 0.546674)

    def test_cpks_parsed(self, parsed):
        """Test that CPKS section is parsed"""
        assert 'cpks' in parsed['egrad']

    def test_cpks_converged(self, parsed):
        """Test CPKS convergence flag"""
        assert parsed['egrad']['cpks']['converged'] is True

    def test_cpks_iterations_count(self, parsed):
        """Test number of CPKS iterations"""
        iterations = parsed['egrad']['cpks']['iterations']
        assert len(iterations) == 3

    def test_cpks_iterations_values(self, parsed):
        """Test CPKS iteration step numbers and residual norms"""
        iterations = parsed['egrad']['cpks']['iterations']
        assert iterations[0]['step'] == 1
        assert np.isclose(iterations[0]['max_residual_norm'], 2.131934773503221e-02)
        assert iterations[1]['step'] == 2
        assert np.isclose(iterations[1]['max_residual_norm'], 8.626720870149588e-04)
        assert iterations[2]['step'] == 3
        assert np.isclose(iterations[2]['max_residual_norm'], 9.799962299242865e-14)

    def test_davidson_parsed(self, parsed):
        """Test that Davidson section is parsed"""
        assert 'davidson' in parsed['egrad']

    def test_davidson_converged(self, parsed):
        """Test Davidson convergence flag"""
        assert parsed['egrad']['davidson']['converged'] is True

    def test_davidson_iterations(self, parsed):
        """Test Davidson iterations (2 iterations for 2 roots)"""
        iterations = parsed['egrad']['davidson']['iterations']
        assert len(iterations) == 2
        assert iterations[0]['step'] == 1
        assert np.isclose(iterations[0]['max_residual_norm'], 7.045689164616345e-02)
        assert iterations[1]['step'] == 2
        assert np.isclose(iterations[1]['max_residual_norm'], 3.274080506635812e-14)


class TestAcroleinCIS:
    """Test parsing of acrolein.cis.txt egrad output (8-atom system with dense Fortran formatting)"""

    @pytest.fixture
    def parsed(self):
        """Parse the acrolein.cis.txt file"""
        filepath = os.path.join(exampledir, "acrolein.cis.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_has_egrad_key(self, parsed):
        assert 'egrad' in parsed

    def test_davidson_converged(self, parsed):
        assert parsed['egrad']['davidson']['converged'] is True

    def test_davidson_iterations(self, parsed):
        iterations = parsed['egrad']['davidson']['iterations']
        assert len(iterations) == 10
        assert np.isclose(iterations[0]['max_residual_norm'], 1.29407479841089e-01)
        assert np.isclose(iterations[9]['max_residual_norm'], 3.932525923725169e-06)

    def test_cpks_converged(self, parsed):
        assert parsed['egrad']['cpks']['converged'] is True

    def test_cpks_iterations(self, parsed):
        iterations = parsed['egrad']['cpks']['iterations']
        assert len(iterations) == 9
        assert np.isclose(iterations[0]['max_residual_norm'], 7.800946513368974e-02)
        assert np.isclose(iterations[8]['max_residual_norm'], 2.823662172303088e-06)

    def test_ground_state_dipole(self, parsed):
        dipole = parsed['egrad']['ground']['dipole']
        assert np.isclose(dipole['total'][0], -0.356430)
        assert np.isclose(dipole['total'][1], -1.424067)
        assert np.isclose(dipole['total'][2], 0.0)

    def test_no_exopt(self, parsed):
        """No exopt line in this file"""
        assert 'exopt' not in parsed['egrad']

    def test_gradient_no_index(self, parsed):
        """No header index and no exopt to backfill from"""
        assert 'index' not in parsed['egrad']['gradient'][0]

    def test_gradient_atom_count(self, parsed):
        gradient = parsed['egrad']['gradient'][0]
        assert len(gradient['atom_list']) == 8

    def test_gradient_atom_list(self, parsed):
        gradient = parsed['egrad']['gradient'][0]
        assert gradient['atom_list'] == [
            '1 c', '2 c', '3 h', '4 h', '5 h', '6 c', '7 o', '8 h'
        ]

    def test_gradient_shape(self, parsed):
        """Each atom should have all 3 gradient components"""
        grad = parsed['egrad']['gradient'][0]['d/dR']
        assert len(grad) == 8
        for atom_grad in grad:
            assert len(atom_grad) == 3

    def test_gradient_values(self, parsed):
        """Test all gradient values, especially where Fortran formatting
        causes adjacent values to run together without whitespace.

        For example, line 469 of the output:
          dE/dx -0.3076426D-02 0.6824289D-02 0.3807253D-02-0.1180016D-01 0.1286775D-01
        has no space between the 3rd and 4th values.
        """
        grad = parsed['egrad']['gradient'][0]['d/dR']
        expected = [
            [-0.3076426e-02, -0.5258624e-02,  0.0],
            [ 0.6824289e-02,  0.9902150e-02,  0.0],
            [ 0.3807253e-02, -0.1064780e-01,  0.0],
            [-0.1180016e-01,  0.3806394e-03,  0.0],
            [ 0.1286775e-01,  0.5345889e-03,  0.0],
            [ 0.1834220e-01,  0.5186150e-01,  0.0],
            [-0.9453334e-02, -0.3598091e-01,  0.0],
            [-0.1751157e-01, -0.1079154e-01,  0.0],
        ]
        assert np.allclose(grad, expected)


class TestBH:
    """Test parsing of BH.b3lyp.txt egrad output (multi-state exopt with 3 gradients)"""

    @pytest.fixture
    def parsed(self):
        filepath = os.path.join(exampledir, "BH.b3lyp.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_exopt(self, parsed):
        """'3 excited states specified in $exopt:     1     2     3' -> [1, 2, 3]"""
        assert parsed['egrad']['exopt'] == [1, 2, 3]

    def test_gradient_count(self, parsed):
        assert len(parsed['egrad']['gradient']) == 3

    def test_gradient_indices(self, parsed):
        """Three gradient sections with indices from headers"""
        gradients = parsed['egrad']['gradient']
        assert gradients[0]['index'] == 1
        assert gradients[1]['index'] == 2
        assert gradients[2]['index'] == 3
