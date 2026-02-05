#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for freeh_parser module"""

import os
import pytest
import numpy as np

from mudslide.turboparse.parse_turbo import parse_turbo

testdir = os.path.dirname(__file__)
exampledir = os.path.join(os.path.dirname(testdir), "examples", "turboparse")


class TestP7H3FreeH:
    """Test parsing of P7H3_freeh.txt output (P7H3 molecule with multiple T/P data)"""

    @pytest.fixture
    def parsed(self):
        """Parse the P7H3_freeh.txt file"""
        filepath = os.path.join(exampledir, "P7H3_freeh.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_has_freeh_key(self, parsed):
        """Test that freeh section is parsed"""
        assert 'freeh' in parsed

    def test_has_data_key(self, parsed):
        """Test that freeh contains data"""
        assert 'data' in parsed['freeh']

    def test_has_units_key(self, parsed):
        """Test that freeh contains units"""
        assert 'units' in parsed['freeh']

    def test_units_temperature(self, parsed):
        """Test temperature units"""
        assert parsed['freeh']['units']['T'] == 'K'

    def test_units_pressure(self, parsed):
        """Test pressure units"""
        assert parsed['freeh']['units']['P'] == 'MPa'

    def test_units_energy(self, parsed):
        """Test energy units"""
        assert parsed['freeh']['units']['energy'] == 'kJ/mol'

    def test_units_entropy(self, parsed):
        """Test entropy units"""
        assert parsed['freeh']['units']['entropy'] == 'kJ/mol/K'

    def test_units_heat_capacity(self, parsed):
        """Test heat capacity units"""
        assert parsed['freeh']['units']['Cv'] == 'kJ/mol-K'
        assert parsed['freeh']['units']['Cp'] == 'kJ/mol-K'

    def test_data_count(self, parsed):
        """Test number of data points - 41 temps * 10 pressures = 410

        Note: The parser captures the last freeh block (multi=False in FreeHData).
        This file has two freeh runs; only the second (100-500K sweep) is captured.
        """
        data = parsed['freeh']['data']
        assert len(data) == 410

    def test_first_data_point_temperature(self, parsed):
        """Test first data point temperature (100 K from second block)"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['T'], 100.0)

    def test_first_data_point_pressure(self, parsed):
        """Test first data point pressure (0.1 MPa)"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['P'], 0.1)

    def test_first_data_point_qtrans(self, parsed):
        """Test first data point ln(qtrans)"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['qtrans'], 15.96)

    def test_first_data_point_qrot(self, parsed):
        """Test first data point ln(qrot)"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['qrot'], 11.05)

    def test_first_data_point_qvib(self, parsed):
        """Test first data point ln(qvib)"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['qvib'], 0.18)

    def test_first_data_point_chem_pot(self, parsed):
        """Test first data point chemical potential"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['chem.pot.'], 112.81)

    def test_first_data_point_energy(self, parsed):
        """Test first data point energy"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['energy'], 138.41)

    def test_first_data_point_entropy(self, parsed):
        """Test first data point entropy"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['entropy'], 0.26430)

    def test_first_data_point_Cv(self, parsed):
        """Test first data point heat capacity at constant volume"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['Cv'], 0.0423454)

    def test_first_data_point_Cp(self, parsed):
        """Test first data point heat capacity at constant pressure"""
        first = parsed['freeh']['data'][0]
        assert np.isclose(first['Cp'], 0.0506597)

    def test_second_data_point(self, parsed):
        """Test second data point (100 K, 0.2 MPa)"""
        point = parsed['freeh']['data'][1]
        assert np.isclose(point['T'], 100.0)
        assert np.isclose(point['P'], 0.2)
        assert np.isclose(point['qtrans'], 15.27)
        assert np.isclose(point['qrot'], 11.05)
        assert np.isclose(point['qvib'], 0.18)
        assert np.isclose(point['chem.pot.'], 113.39)
        assert np.isclose(point['energy'], 138.41)
        assert np.isclose(point['entropy'], 0.25853)

    def test_high_temperature_point(self, parsed):
        """Test a high temperature point (500 K)"""
        # Find a 500 K data point
        data = parsed['freeh']['data']
        high_t_points = [d for d in data if np.isclose(d['T'], 500.0)]
        assert len(high_t_points) > 0
        # Check first 500 K point (at 0.1 MPa)
        point = high_t_points[0]
        assert np.isclose(point['T'], 500.0)
        assert np.isclose(point['P'], 0.1)

    def test_high_pressure_point(self, parsed):
        """Test a high pressure point (1.0 MPa)"""
        data = parsed['freeh']['data']
        high_p_points = [d for d in data if np.isclose(d['P'], 1.0)]
        assert len(high_p_points) > 0


class TestGraphyneFreeH:
    """Test parsing of graphyne_freeh.txt output (large graphyne molecule)"""

    @pytest.fixture
    def parsed(self):
        """Parse the graphyne_freeh.txt file"""
        filepath = os.path.join(exampledir, "graphyne_freeh.txt")
        with open(filepath, 'r') as f:
            return parse_turbo(f)

    def test_has_freeh_key(self, parsed):
        """Test that freeh section is parsed"""
        assert 'freeh' in parsed

    def test_has_data_key(self, parsed):
        """Test that freeh contains data"""
        assert 'data' in parsed['freeh']

    def test_data_count(self, parsed):
        """Test number of data points - single T/P point"""
        data = parsed['freeh']['data']
        assert len(data) == 1

    def test_temperature(self, parsed):
        """Test temperature value (298.15 K)"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['T'], 298.15)

    def test_pressure(self, parsed):
        """Test pressure value (0.1 MPa)"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['P'], 0.1)

    def test_qtrans(self, parsed):
        """Test ln(qtrans) for large molecule"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['qtrans'], 20.36)

    def test_qrot(self, parsed):
        """Test ln(qrot) for large molecule"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['qrot'], 18.76)

    def test_qvib(self, parsed):
        """Test ln(qvib) for large molecule (many vibrational modes)"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['qvib'], 34.82)

    def test_chem_pot(self, parsed):
        """Test chemical potential"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['chem.pot.'], 1168.12)

    def test_energy(self, parsed):
        """Test energy"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['energy'], 1453.67)

    def test_entropy(self, parsed):
        """Test entropy (high value for large molecule)"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['entropy'], 0.96606)

    def test_Cv(self, parsed):
        """Test heat capacity at constant volume"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['Cv'], 0.6684565)

    def test_Cp(self, parsed):
        """Test heat capacity at constant pressure"""
        point = parsed['freeh']['data'][0]
        assert np.isclose(point['Cp'], 0.6767708)

    def test_enthalpy(self, parsed):
        """Test enthalpy (graphyne file has enthalpy column)"""
        point = parsed['freeh']['data'][0]
        assert 'H' in point
        assert np.isclose(point['H'], 1456.15)
